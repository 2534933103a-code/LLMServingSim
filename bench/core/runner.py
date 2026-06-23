"""vLLM benchmark runner — strict replay of an existing dataset.

The runner reads a LLMServingSim-format JSONL workload (the same format
``python -m workloads.generators sharegpt`` produces and ``python -m serving
--dataset`` consumes) and replays every request through vLLM with its
``input_tok_ids`` and ``output_toks`` pinned, so the run is bit-for-bit
comparable to the simulator's view of the same workload.

White-room implementation against ``../vllm``:
  * ``vllm.v1.engine.async_llm.AsyncLLM`` — async engine, ``generate()``
    yields ``RequestOutput`` per chunk, ``RequestOutput.metrics`` carries
    per-request ``RequestStateStats`` (arrival_time / queued_ts /
    scheduled_ts / first_token_ts / last_token_ts).
  * ``vllm.v1.metrics.loggers.StatLoggerBase`` — pluggable per-engine stat
    logger; we hook it via ``BenchStatLogger`` to capture per-iteration
    scheduler/iteration stats for ``timeseries.csv``.

Output: ``<output-dir>/{meta.json, requests.jsonl, timeseries.csv}``.
The dataset itself is not modified — generation lives in
``workloads/generators``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path

from bench.core import logger as log
from bench.core import recorder


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id passed verbatim to vllm.AsyncLLM.")
    p.add_argument("--dataset", required=True,
                   help="Path to a LLMServingSim-format JSONL workload "
                        "(produced by `python -m workloads.generators`).")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory for this run "
                        "(meta.json/requests.jsonl/timeseries.csv).")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   dest="tensor_parallel_size",
                   help="vLLM tensor_parallel_size.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   dest="data_parallel_size",
                   help="vLLM data_parallel_size (DP across engines).")
    p.add_argument("--pipeline-parallel-size", type=int, default=1,
                   dest="pipeline_parallel_size",
                   help="vLLM pipeline_parallel_size (PP, layers split across GPUs).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   dest="enable_expert_parallel", default=False,
                   help="vLLM enable_expert_parallel for MoE models.")
    p.add_argument("--max-num-seqs", type=int, default=128,
                   dest="max_num_seqs",
                   help="vLLM scheduler max_num_seqs (per-engine running cap).")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   dest="max_num_batched_tokens",
                   help="vLLM scheduler max_num_batched_tokens.")
    p.add_argument("--max-model-len", type=int, default=None,
                   dest="max_model_len",
                   help="vLLM max_model_len (None = model's max).")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   dest="kv_cache_dtype",
                   help="vLLM kv_cache_dtype.")
    p.add_argument("--load-format", default="dummy",
                   dest="load_format",
                   help="vLLM load_format. Default 'dummy' skips weight download.")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for vLLM.")
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
    p.add_argument("--enable-prefix-caching", action="store_true",
                   default=True, dest="enable_prefix_caching",
                   help="vLLM enable_prefix_caching (default: True).")
    p.add_argument("--no-enable-prefix-caching", action="store_false",
                   dest="enable_prefix_caching",
                   help="Disable vLLM prefix caching.")


def run(args: argparse.Namespace) -> int:
    from bench.core.stat_logger import BenchStatLogger

    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Bench",
        f"vLLM end-to-end run -> {args.output_dir}",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests:
        raise ValueError(f"No requests loaded from {args.dataset}")
    log.info("Loaded %d requests from %s", len(requests), args.dataset)

    BenchStatLogger.reset()
    asyncio.run(_drive(args, requests, output_dir))
    return 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> list[dict]:
    """Read a LLMServingSim-format JSONL workload.

    Flattens agentic sessions (rows with ``sub_requests``) into individual
    requests chained sequentially.
    """
    requests: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if "sub_requests" in row:
                # Flatten agentic session: sub-requests run sequentially
                session_arrival = row.get("arrival_time_ns", 0)
                for j, sr in enumerate(row["sub_requests"]):
                    if "input_tok_ids" not in sr or not sr["input_tok_ids"]:
                        raise ValueError(
                            f"Sub-request {j} missing input_tok_ids in {path}"
                        )
                    requests.append({
                        "input_tok_ids": sr["input_tok_ids"],
                        "input_toks": sr["input_toks"],
                        "output_tok_ids": sr.get("output_tok_ids", []),
                        "output_toks": sr["output_toks"],
                        "arrival_time_ns": session_arrival,
                        "_session_id": row.get("session_id"),
                        "_sub_index": j,
                        "_tool_delay_ns": sr.get("tool_duration_ns", 0),
                        "_is_first": (j == 0),
                    })
                continue

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
# Async driver
# ---------------------------------------------------------------------------

async def _drive(args: argparse.Namespace, requests: list[dict], output_dir: Path) -> None:
    # Imports deferred so `validate` / `--help` works without vLLM installed.
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    from bench.core.stat_logger import BenchStatLogger

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        seed=args.seed,
        load_format=args.load_format,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=False,
    )
    engine_kwargs_for_meta = _engine_kwargs_for_meta(engine_args)

    with log.stage("Booting AsyncLLM"):
        with log.capture_stdio():
            engine = AsyncLLM.from_engine_args(
                engine_args, stat_loggers=[BenchStatLogger]
            )

    # Persist vLLM runtime logs alongside results (scheduler / memory events).
    # Same principle as bench_nixl.sh's `vllm serve > "$logfile" 2>&1`:
    # redirect fd 2 directly to the file so ALL vLLM output (Python logging +
    # C-level prints) is captured without fighting Python logger hierarchy.
    vllm_log_path = str(output_dir / "vllm.log")
    saved_stderr = os.dup(2)
    vllm_log_fd = os.open(vllm_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(vllm_log_fd, 2)
    os.close(vllm_log_fd)
    # Also ensure Python-level vLLM loggers emit at INFO level (they may have
    # been clamped to ERROR by configure() during boot).
    _VLLM_LOGGERS = (
        "vllm", "vllm.engine", "vllm.worker", "vllm.executor",
        "vllm.config", "vllm.model_executor", "vllm.distributed",
        "vllm.v1", "vllm.core",
    )
    for _name in _VLLM_LOGGERS:
        logging.getLogger(_name).setLevel(logging.INFO)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    try:
        with log.stage(f"Submitting {len(requests)} requests"):
            records = await _submit_all(
                engine, requests, SamplingParams, TokensPrompt,
                max_model_len=args.max_model_len,
            )
    finally:
        # Restore fd 2 before shutdown so shutdown messages go to the terminal
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        with log.stage("Shutting AsyncLLM down"):
            engine.shutdown()

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Persist outputs.
    # ------------------------------------------------------------------
    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version=_vllm_version(),
        engine_kwargs=engine_kwargs_for_meta,
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
    )
    recorder.write_requests(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)
    log.success(
        "%d requests, %d timeseries rows -> %s",
        len(records), len(rows), output_dir,
    )


async def _submit_all(engine, requests: list[dict], SamplingParams, TokensPrompt,
                      max_model_len: int | None = None) -> list[dict]:
    """Schedule each request at its arrival offset, gather metrics."""
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    completed = [0]  # boxed so the inner closure can mutate

    # Build chaining events for agentic sessions
    session_events: dict[int, list[asyncio.Event]] = {}
    session_requests: dict[int, list[int]] = {}
    for i, r in enumerate(requests):
        sid = r.get("_session_id")
        if sid is not None:
            session_requests.setdefault(sid, []).append(i)
    for sid, idxs in session_requests.items():
        session_events[sid] = [asyncio.Event() for _ in idxs]

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            # Scheduling: handle agentic session chaining
            sid = req.get("_session_id")
            sub_idx = req.get("_sub_index")
            if sid is not None and sub_idx is not None and sub_idx > 0:
                await session_events[sid][sub_idx - 1].wait()
                tool_delay_s = req.get("_tool_delay_ns", 0) / 1e9
                if tool_delay_s > 0:
                    await asyncio.sleep(tool_delay_s)
            else:
                target = t0_loop + req["arrival_time_ns"] / 1e9
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)

            # Sliding window: truncate input_tok_ids from the left if
            # input + output exceeds max_model_len, keeping the most
            # recent context.  Reserve room for output tokens.
            tok_ids = list(req["input_tok_ids"])
            if max_model_len and len(tok_ids) + n_out > max_model_len:
                keep = max(1, max_model_len - n_out)
                tok_ids = tok_ids[-keep:]

            n_out = int(req["output_toks"])
            sp = SamplingParams(
                min_tokens=n_out,
                max_tokens=n_out,
                ignore_eos=True,
                temperature=0.0,
            )
            prompt = TokensPrompt(prompt_token_ids=tok_ids)
            request_id = f"bench-{idx}"

            last_metrics = None
            async for output in engine.generate(prompt, sp, request_id):
                if output.metrics is not None:
                    last_metrics = output.metrics

            if sid is not None and sub_idx is not None:
                evts = session_events.get(sid, [])
                if sub_idx < len(evts):
                    evts[sub_idx].set()

            completed[0] += 1
            bar.advance()
            return _record_from_metrics(idx, req, last_metrics)

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        return await asyncio.gather(*tasks)


def _record_from_metrics(idx: int, req: dict, metrics) -> dict:
    """Project ``RequestStateStats`` onto our flat per-request schema."""
    if metrics is None:
        return {
            "request_id": f"bench-{idx}",
            "input_toks": int(req["input_toks"]),
            "output_toks": int(req["output_toks"]),
            "arrival_time": None,
            "queued_ts": None,
            "scheduled_ts": None,
            "first_token_ts": None,
            "last_token_ts": None,
        }
    return {
        "request_id": f"bench-{idx}",
        "input_toks": int(req["input_toks"]),
        "output_toks": int(req["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "queued_ts": getattr(metrics, "queued_ts", None),
        "scheduled_ts": getattr(metrics, "scheduled_ts", None),
        "first_token_ts": getattr(metrics, "first_token_ts", None),
        "last_token_ts": getattr(metrics, "last_token_ts", None),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_kwargs_for_meta(engine_args) -> dict:
    fields = (
        "model", "tensor_parallel_size", "data_parallel_size",
        "pipeline_parallel_size", "enable_expert_parallel",
        "max_num_seqs", "max_num_batched_tokens",
        "max_model_len", "dtype", "kv_cache_dtype", "seed",
    )
    return {k: getattr(engine_args, k, None) for k in fields}


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
