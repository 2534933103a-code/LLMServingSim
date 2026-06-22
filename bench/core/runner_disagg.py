"""HTTP-based benchmark runner for PD-separated vLLM.

Sends streaming HTTP requests to the disagg proxy (which routes through
prefill → decode) and records per-request end-to-end timing.  Also polls
the prefill/decode instances' ``/metrics`` endpoints to collect exact
scheduler-level throughput and running/waiting stats for ``timeseries.csv``.

The runner reads a LLMServingSim-format JSONL workload and replays every
request through the proxy with its ``input_tok_ids`` and ``output_toks``
pinned, so results are comparable to the colocated bench and the simulator.

Output: ``<output-dir>/{meta.json, requests.jsonl, timeseries.csv}``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from bench.core import logger as log
from bench.core import recorder


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id (passed in request body).")
    p.add_argument("--dataset", required=True,
                   help="Path to a LLMServingSim-format JSONL workload.")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory (meta.json / requests.jsonl).")
    p.add_argument("--proxy-url", required=True, dest="proxy_url",
                   default="http://localhost:8000",
                   help="URL of the disagg proxy server (default: http://localhost:8000).")
    p.add_argument("--prefill-url", dest="prefill_url",
                   default=None,
                   help="URL of the prefill vLLM instance for /metrics polling "
                        "(default: none — skip scheduler-level timeseries).")
    p.add_argument("--decode-url", dest="decode_url",
                   default=None,
                   help="URL of the decode vLLM instance for /metrics polling "
                        "(default: none — skip scheduler-level timeseries).")
    p.add_argument("--tick-seconds", type=float, default=1.0,
                   dest="tick_seconds",
                   help="Stat logger downsample interval (timeseries.csv row spacing).")
    p.add_argument("--num-reqs", type=int, default=0,
                   dest="num_reqs",
                   help="Cap on number of requests from the dataset (0 = all).")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   dest="request_timeout",
                   help="Per-request HTTP timeout in seconds (default: 600).")


def run(args: argparse.Namespace) -> int:
    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Bench (PD Disagg)",
        f"HTTP-based PD-separated vLLM run -> {args.output_dir}",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests:
        raise ValueError(f"No requests loaded from {args.dataset}")
    log.info("Loaded %d requests from %s", len(requests), args.dataset)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    records, metrics_samples = asyncio.run(_drive(args, requests))
    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    # Persist outputs (same format as colocated bench).
    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version="pd-disagg-via-proxy",
        engine_kwargs={
            "proxy_url": args.proxy_url,
            "mode": "pd_disaggregated",
            "prefill_url": args.prefill_url or "n/a",
            "decode_url": args.decode_url or "n/a",
        },
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
    )
    recorder.write_requests(output_dir, records)

    # Timeseries: use /metrics polling if available, else fall back to client-side approx.
    header, rows = _build_timeseries(records, metrics_samples, args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)

    log.success(
        "%d requests, %d timeseries rows -> %s",
        len(records), len(rows), output_dir,
    )
    return 0


# ---------------------------------------------------------------------------
# Dataset loading (shared with runner.py)
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> list[dict]:
    """Read a LLMServingSim-format JSONL workload."""
    requests: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sub_requests" in row:
                continue  # agentic sessions: not supported
            if "input_tok_ids" not in row or not row["input_tok_ids"]:
                raise ValueError(
                    f"Row missing input_tok_ids in {path}; regenerate the "
                    f"dataset with `python -m workloads.generators`."
                )
            requests.append(row)
            if cap and len(requests) >= cap:
                break
    return requests


# ---------------------------------------------------------------------------
# Async HTTP driver
# ---------------------------------------------------------------------------

async def _drive(
    args: argparse.Namespace, requests: list[dict],
) -> tuple[list[dict], list[dict] | None]:
    proxy_url = args.proxy_url.rstrip("/")
    endpoint = f"{proxy_url}/v1/completions"
    tick_seconds = args.tick_seconds

    # Start /metrics polling if URLs are provided
    poll_task: asyncio.Task | None = None
    metrics_samples: list[dict] | None = None
    if args.prefill_url and args.decode_url:
        metrics_samples = []
        poll_task = asyncio.create_task(
            _poll_metrics(
                args.prefill_url.rstrip("/"),
                args.decode_url.rstrip("/"),
                tick_seconds,
                metrics_samples,
            )
        )

    with log.stage(f"Submitting {len(requests)} requests via HTTP"):
        records = await _submit_all(
            endpoint, args.model, requests,
            timeout=args.request_timeout,
        )

    # Stop polling and collect final samples
    if poll_task is not None:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

    return records, metrics_samples


async def _submit_all(
    endpoint: str,
    model: str,
    requests: list[dict],
    timeout: float = 600.0,
) -> list[dict]:
    """Schedule each request at its arrival offset, gather timing via SSE."""
    import aiohttp

    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    completed = [0]  # boxed for inner closure

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            n_out = int(req["output_toks"])

            payload = {
                "model": model,
                "prompt": list(req["input_tok_ids"]),
                "max_tokens": n_out,
                "min_tokens": n_out,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": True,
            }

            target = t0_loop + req["arrival_time_ns"] / 1e9
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            arrival = loop.time()
            first_token_ts: float | None = None
            last_token_ts: float | None = None
            num_tokens_seen = 0
            prefill_ttft_s: float | None = None  # prefill-only timing (injected by proxy)

            client_timeout = aiohttp.ClientTimeout(total=timeout)
            try:
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.post(endpoint, json=payload) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            log.warning(
                                "Request %d: HTTP %d: %s",
                                idx, resp.status, error_text[:200],
                            )
                            raise RuntimeError(
                                f"Request {idx}: HTTP {resp.status}"
                            )

                        async for line_bytes in resp.content:
                            line = line_bytes.decode("utf-8", errors="replace").strip()
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Proxy-injected prefill timing event (not a token)
                            if "prefill_ttft_s" in data:
                                prefill_ttft_s = data["prefill_ttft_s"]
                                continue

                            t_now = loop.time()
                            if first_token_ts is None:
                                first_token_ts = t_now
                            last_token_ts = t_now

                            choices = data.get("choices", [])
                            if choices:
                                token_ids = choices[0].get("token_ids")
                                if token_ids is not None:
                                    num_tokens_seen += len(token_ids)

            except asyncio.TimeoutError:
                log.warning("Request %d: timeout after %.1fs", idx, timeout)
            except Exception as exc:
                log.warning("Request %d: %s", idx, exc)

            completed[0] += 1
            bar.advance()

            return {
                "request_id": f"bench-disagg-{idx}",
                "input_toks": int(req["input_toks"]),
                "output_toks": n_out,
                "arrival_time": arrival,
                "queued_ts": None,
                "scheduled_ts": None,
                "first_token_ts": first_token_ts,
                "last_token_ts": last_token_ts,
                "tokens_received": num_tokens_seen,
                "prefill_ttft_s": prefill_ttft_s,
            }

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# /metrics polling — Prometheus text format → exact scheduler stats
# ---------------------------------------------------------------------------

# Metric names in vLLM v0.19.0.  In multiprocess mode they gain a
# ``_total`` prefix; we match both.
_METRIC_RE = re.compile(
    r"^(?P<name>vllm(?:_total)?_(?P<base>num_requests_running|num_requests_waiting"
    r"|prompt_tokens|generation_tokens|kv_cache_usage_perc))"
    r"\{.*\}\s+(?P<value>[0-9.e+\-]+)",
    re.MULTILINE,
)


async def _poll_metrics(
    prefill_url: str,
    decode_url: str,
    tick_seconds: float,
    samples_out: list[dict],
) -> None:
    """Background task: poll /metrics every tick_seconds, aggregate
    prefill + decode Prometheus gauges/counters into samples_out."""
    import aiohttp

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    prev_prompt_tokens: float | None = None
    prev_gen_tokens: float | None = None

    timeout = aiohttp.ClientTimeout(total=5)

    while True:
        t = loop.time() - t0
        running = 0
        waiting = 0
        prompt_tokens = 0.0
        gen_tokens = 0.0
        kv_cache_pct = 0.0
        n_kv = 0

        # Fetch from both instances
        for url in (prefill_url, decode_url):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{url}/metrics") as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()
            except Exception:
                continue

            for m in _METRIC_RE.finditer(text):
                base = m.group("base")
                val = float(m.group("value"))
                if base == "num_requests_running":
                    running += int(val)
                elif base == "num_requests_waiting":
                    waiting += int(val)
                elif base == "prompt_tokens":
                    prompt_tokens += val
                elif base == "generation_tokens":
                    gen_tokens += val
                elif base == "kv_cache_usage_perc":
                    kv_cache_pct += val
                    n_kv += 1

        # Compute throughput from cumulative token counters
        prompt_tput = 0.0
        gen_tput = 0.0
        if prev_prompt_tokens is not None:
            prompt_tput = max(0.0, (prompt_tokens - prev_prompt_tokens) / tick_seconds)
        if prev_gen_tokens is not None:
            gen_tput = max(0.0, (gen_tokens - prev_gen_tokens) / tick_seconds)
        prev_prompt_tokens = prompt_tokens
        prev_gen_tokens = gen_tokens

        samples_out.append({
            "t": round(t, 2),
            "prompt_throughput": round(prompt_tput, 1),
            "gen_throughput": round(gen_tput, 1),
            "running": running,
            "waiting": waiting,
            "kv_cache_pct": round(kv_cache_pct / max(1, n_kv), 2),
        })

        await asyncio.sleep(tick_seconds)


# ---------------------------------------------------------------------------
# Timeseries builder
# ---------------------------------------------------------------------------

def _build_timeseries(
    records: list[dict],
    metrics_samples: list[dict] | None,
    tick_seconds: float,
) -> tuple[list[str], list[list]]:
    """Build timeseries rows from /metrics polling (preferred) or
    fall back to client-side approximation."""
    header = ["t", "prompt_throughput", "gen_throughput",
              "running", "waiting", "kv_cache_pct"]

    if metrics_samples and len(metrics_samples) >= 2:
        # Use exact scheduler stats from /metrics polling
        rows = [
            [s["t"], s["prompt_throughput"], s["gen_throughput"],
             s["running"], s["waiting"], s["kv_cache_pct"]]
            for s in metrics_samples
            if s["running"] + s["waiting"] > 0  # skip idle ticks
              or s["prompt_throughput"] + s["gen_throughput"] > 0
        ]
        if rows:
            return header, rows

    # Fallback: client-side approximation
    return _compute_timeseries_client(records, tick_seconds)


def _compute_timeseries_client(
    records: list[dict], tick_seconds: float,
) -> tuple[list[str], list[list]]:
    """Client-side approximation when /metrics is unavailable."""
    valid = [r for r in records
             if r.get("arrival_time") is not None
             and r.get("first_token_ts") is not None
             and r.get("last_token_ts") is not None]
    if not valid:
        return (["t", "prompt_throughput", "gen_throughput",
                 "running", "waiting", "kv_cache_pct"], [])

    header = ["t", "prompt_throughput", "gen_throughput",
              "running", "waiting", "kv_cache_pct"]

    t_base = min(r["arrival_time"] for r in valid)
    t_end = max(r["last_token_ts"] for r in valid)
    num_ticks = max(1, int((t_end - t_base) / tick_seconds) + 1)
    rows: list[list] = []

    for i in range(num_ticks):
        t_start = t_base + i * tick_seconds
        t_stop = t_start + tick_seconds
        t_mid = (t_start + t_stop) / 2.0

        prompt_tokens = 0
        gen_tokens = 0
        running = 0
        waiting = 0

        for r in valid:
            first = r["first_token_ts"]
            last = r["last_token_ts"]
            arr = r["arrival_time"]

            if t_start <= first < t_stop:
                prompt_tokens += int(r.get("input_toks", 0))
            if t_start <= last < t_stop:
                gen_tokens += int(r.get("output_toks", 0))

            if first <= t_mid < last:
                running += 1
            elif arr <= t_mid < first:
                waiting += 1

        rows.append([
            round(t_mid - t_base, 2),
            round(prompt_tokens / tick_seconds, 1),
            round(gen_tokens / tick_seconds, 1),
            running,
            waiting,
            0.0,
        ])

    return header, rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
