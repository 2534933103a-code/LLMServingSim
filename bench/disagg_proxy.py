#!/usr/bin/env python3
"""PD disaggregation proxy server for vLLM.

Routes requests through prefill → decode stages for disaggregated prefilling.
The prefill instance generates 1 token and transfers KV cache; the decode
instance picks up from the KV cache to produce the full output.

Adapted from vllm/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py

Usage:
    python3 disagg_proxy.py --port 8000 --prefill-url http://localhost:8100 --decode-url http://localhost:8200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from urllib.parse import urlparse

import aiohttp
from quart import Quart, Response, make_response, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy server")
    parser.add_argument(
        "--timeout", type=float, default=6 * 60 * 60,
        help="Timeout for backend service requests in seconds (default: 21600)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--prefill-url", type=str, default="http://localhost:8100",
        help="Prefill service base URL",
    )
    parser.add_argument(
        "--decode-url", type=str, default="http://localhost:8200",
        help="Decode service base URL",
    )
    parser.add_argument(
        "--kv-host", type=str, default="localhost",
        help="Hostname/IP for KV transfer (default: localhost)",
    )
    parser.add_argument(
        "--prefill-kv-port", type=int, default=14579,
        help="Prefill KV port (default: 14579)",
    )
    parser.add_argument(
        "--decode-kv-port", type=int, default=14580,
        help="Decode KV port (default: 14580)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PORT = args.port

    PREFILL_KV_ADDR = f"{args.kv_host}:{args.prefill_kv_port}"
    DECODE_KV_ADDR = f"{args.kv_host}:{args.decode_kv_port}"

    logger.info(
        "Proxy resolved KV addresses -> prefill: %s, decode: %s",
        PREFILL_KV_ADDR, DECODE_KV_ADDR,
    )

    app = Quart(__name__)
    app.config.update({
        "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
        "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
        "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
        "PREFILL_KV_ADDR": PREFILL_KV_ADDR,
        "DECODE_KV_ADDR": DECODE_KV_ADDR,
    })

    def _normalize_base_url(url: str) -> str:
        return url.rstrip("/")

    def _get_host_port(url: str) -> str:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (80 if parsed.scheme == "http" else 443)
        return f"{host}:{port}"

    PREFILL_BASE = _normalize_base_url(PREFILL_SERVICE_URL)
    DECODE_BASE = _normalize_base_url(DECODE_SERVICE_URL)
    KV_TARGET = _get_host_port(DECODE_SERVICE_URL)

    def _build_headers(request_id: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-Request-Id": request_id,
            "X-KV-Target": KV_TARGET,
        }
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _run_prefill(
        request_path: str, payload: dict, headers: dict[str, str], request_id: str,
    ):
        url = f"{PREFILL_BASE}{request_path}"
        start_ts = time.perf_counter()
        logger.info("[prefill] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Prefill backend error {resp.status}: {error_text}")
                await resp.read()
                logger.info(
                    "[prefill] done request_id=%s status=%s elapsed=%.2fs",
                    request_id, resp.status, time.perf_counter() - start_ts,
                )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Prefill service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Prefill service unavailable at {url}") from exc

    async def _stream_decode(
        request_path: str, payload: dict, headers: dict[str, str], request_id: str,
    ):
        url = f"{DECODE_BASE}{request_path}"
        logger.info("[decode] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Decode backend error %s - %s", resp.status, error_text)
                    err_msg = (
                        '{"error": "Decode backend error ' + str(resp.status) + '"}'
                    )
                    yield err_msg.encode()
                    return
                logger.info(
                    "[decode] streaming response request_id=%s status=%s",
                    request_id, resp.status,
                )
                async for chunk_bytes in resp.content.iter_chunked(1024):
                    yield chunk_bytes
                logger.info("[decode] finished streaming request_id=%s", request_id)
        except asyncio.TimeoutError:
            logger.error("Decode service timeout at %s", url)
            yield b'{"error": "Decode service timeout"}'
        except aiohttp.ClientError as exc:
            logger.error("Decode service error at %s: %s", url, exc)
            yield b'{"error": "Decode service unavailable"}'

    async def process_request():
        try:
            original_request_data = await request.get_json()
            t_received = time.perf_counter()

            # Create prefill request (max_tokens=1 so prefill only generates 1 token)
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            prefill_request["min_tokens"] = 1           # must match max_tokens
            prefill_request["ignore_eos"] = False       # allow EOS after 1 token
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            request_id = (
                f"___prefill_addr_{PREFILL_KV_ADDR}___decode_addr_"
                f"{DECODE_KV_ADDR}_{uuid.uuid4().hex}"
            )

            headers = _build_headers(request_id)
            await _run_prefill(request.path, prefill_request, headers, request_id)
            t_prefill_done = time.perf_counter()
            prefill_ttft_s = round(t_prefill_done - t_received, 4)

            # Stream decode, prefixed with prefill timing event
            async def generator():
                first_event = json.dumps({"prefill_ttft_s": prefill_ttft_s})
                yield f"data: {first_event}\n\n".encode()
                async for chunk in _stream_decode(
                    request.path, original_request_data, headers, request_id,
                ):
                    yield chunk

            response = await make_response(generator())
            response.timeout = None  # Disable timeout for streaming response
            return response

        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/v1/completions", methods=["POST"])
    async def handle_request():
        try:
            return await process_request()
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    # Also support v1/chat/completions by passing through
    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_chat_request():
        try:
            return await process_request()
        except asyncio.CancelledError:
            logger.warning("Chat request cancelled")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    app.run(port=PORT)


if __name__ == "__main__":
    main()
