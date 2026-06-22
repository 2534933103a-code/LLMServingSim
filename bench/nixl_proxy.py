#!/usr/bin/env python3
"""NixlConnector PD disaggregation proxy server.

Routes requests through prefill → decode stages for Nixl-based KV cache transfer.
Supports X prefills + Y decodes (XpYd) with round-robin scheduling.

Key differences from P2pNcclConnector proxy:
  - Prefill response carries ``kv_transfer_params`` (NIXL connection info)
  - Decode step passes those params so the decoder can locate the KV cache
  - Round-robin across multiple prefill/decode instances
  - Uses httpx + fastapi instead of aiohttp + quart

Adapted from: vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

Usage:
    python3 nixl_proxy.py --port 8000 \
        --prefiller-hosts localhost --prefiller-ports 8100 \
        --decoder-hosts localhost --decoder-ports 8200
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NixlConnector P/D disaggregation proxy (XpYd)"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--prefiller-hosts", "--prefiller-host",
        type=str, nargs="+", default=["localhost"],
        help="Hostname(s) for prefill instance(s)",
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port",
        type=int, nargs="+", default=[8100],
        help="Port(s) for prefill instance(s)",
    )
    parser.add_argument(
        "--decoder-hosts", "--decoder-host",
        type=str, nargs="+", default=["localhost"],
        help="Hostname(s) for decode instance(s)",
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port",
        type=int, nargs="+", default=[8200],
        help="Port(s) for decode instance(s)",
    )

    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports"
        )

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create HTTP client pools. Shutdown: close them."""
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=base_url,
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    for i, (host, port) in enumerate(global_args.decoder_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=base_url,
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Initialized %d prefill client(s) and %d decode client(s).",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    for c in app.state.prefill_clients:
        await c["client"].aclose()
    for c in app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


def _next_client(app_state, service_type: str) -> dict:
    if service_type == "prefill":
        idx = next(app_state.prefill_iterator)
        return app_state.prefill_clients[idx]
    elif service_type == "decode":
        idx = next(app_state.decode_iterator)
        return app_state.decode_clients[idx]
    raise ValueError(f"Unknown service type: {service_type}")


async def _send_prefill(client_info: dict, endpoint: str, req_data: dict, request_id: str):
    """Send request to a prefill instance. Returns the response JSON."""
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1           # must match max_tokens
    payload["ignore_eos"] = False       # allow EOS after 1 token
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    if "stream_options" in payload:
        del payload["stream_options"]

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    # NIXL side-channel host override for cross-node
    if os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST"):
        headers["X-Nixl-Sidechannel-Host"] = os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"]

    response = await client_info["client"].post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    await response.aclose()
    return data


async def _stream_decode(client_info: dict, endpoint: str, req_data: dict, request_id: str):
    """Stream response from a decode instance."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        t_received = time.perf_counter()

        # 1. Send to prefill (max_tokens=1, returns kv_transfer_params)
        prefill_client = _next_client(request.app.state, "prefill")
        prefill_resp = await _send_prefill(prefill_client, api, req_data, request_id)
        t_prefill_done = time.perf_counter()

        # 2. Attach kv_transfer_params from prefill response to the decode request
        kv_transfer_params = prefill_resp.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # 3. Stream from decode, prefixed with prefill timing event
        decode_client = _next_client(request.app.state, "decode")
        logger.debug("Prefill=%s:%d  Decode=%s:%d",
                     prefill_client["host"], prefill_client["port"],
                     decode_client["host"], decode_client["port"])

        # Prefill-side TTFT: proxy receipt → prefill complete (token₁ ready).
        # This aligns with the simulator's definition (prefill processing only,
        # no KV transfer / proxy overhead included).
        prefill_ttft_s = round(t_prefill_done - t_received, 4)

        async def generate():
            # Inject prefill timing as the first SSE event
            first_event = json.dumps({"prefill_ttft_s": prefill_ttft_s})
            yield f"data: {first_event}\n\n".encode()
            async for chunk in _stream_decode(decode_client, api, req_data, request_id):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception:
        import sys
        import traceback
        logger.exception("Error in disagg nixl proxy - %s endpoint", api)
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
