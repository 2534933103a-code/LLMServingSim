#!/bin/bash
# Run a vLLM Nixl-connector PD-separation benchmark.
#
# Supports XpYd (X prefill instances + Y decode instances) with round-robin
# routing through the Nixl proxy. Can also be used for 1P1D.
#
# Usage:
#     ./bench/bench_nixl.sh
#
# Prerequisites:
#     - At least 2 GPUs (1P1D) or more (XpYd)
#     - Inside the vLLM Docker container (scripts/docker-vllm.sh)
#     - Dependencies: pip install nixl httpx uvicorn fastapi
#
# Architecture (1P1D example):
#     prefill (GPU 0, :8100) ──┐
#                               ├── nixl_proxy (:8000) ── bench client
#     decode  (GPU 1, :8200) ──┘
#
# KV transfer: NixlConnector via UCX RDMA (async send/recv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../bench
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# EDIT THESE
# =============================================================================
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-workloads/agent_trace_test.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-bench/results/$RUN_ID}"

# -- GPU / Port assignment (comma-separated for multiple instances) --
#    e.g. PREFILL_GPUS="0,1" DECODE_GPUS="2,3" for 2P2D
PREFILL_GPUS="${PREFILL_GPUS:-0}"
DECODE_GPUS="${DECODE_GPUS:-1}"
PREFILL_PORTS="${PREFILL_PORTS:-8100}"
DECODE_PORTS="${DECODE_PORTS:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"

# -- vLLM engine parameters (shared by all instances) --
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"  # 'auto' to download real weights
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
SEED="${SEED:-42}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"  # 1 = use --enforce-eager (recommended for Nixl)
NO_ENABLE_PREFIX_CACHING="${NO_ENABLE_PREFIX_CACHING:-0}"  # 1 = use --no-enable-prefix-caching
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # 1 = skip HuggingFace network check, use local cache

# -- NIXL parameters --
NIXL_SIDE_CHANNEL_BASE_PORT="${NIXL_SIDE_CHANNEL_BASE_PORT:-5600}"
NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-}"  # empty = localhost; set for cross-node
KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY:-fail}"  # fail | recompute
# UCX transport: leave empty to auto-select; override if needed.
# Common values: "cuda_ipc,cuda_copy" (GPU-only), "rc" (IB RDMA), "tcp" (fallback)
UCX_TLS="${UCX_TLS:-}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

# -- Benchmark parameters --
TICK_SECONDS="${TICK_SECONDS:-0.5}"
NUM_REQS="${NUM_REQS:-0}"            # 0 => replay the full dataset
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# -- Host IP (for cross-node, set to the node's real IP) --
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

kill_gpu_processes() {
    echo "Cleaning up GPU processes..."
    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f "nixl_proxy" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    for port in $(echo "$PREFILL_PORTS" "$DECODE_PORTS" "$PROXY_PORT" | tr ',' ' '); do
        lsof -ti:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    sleep 1
    echo "Cleanup done."
}

SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-60}"  # max seconds to wait for vLLM to start

wait_for_server() {
    local port=$1
    local label=$2
    local pid=$3        # background vLLM PID (0 = skip liveness check)
    local logfile=$4    # log file to tail on failure

    echo "Waiting for $label at port $port (timeout: ${SERVER_START_TIMEOUT}s)..."

    local elapsed=0
    while [[ $elapsed -lt $SERVER_START_TIMEOUT ]]; do
        # Check if port is responding
        if curl -s -o /dev/null -w '%{http_code}' "localhost:${port}/v1/models" 2>/dev/null | grep -q '200'; then
            echo "$label on port $port is ready."
            return 0
        fi

        # Check if background process died unexpectedly
        if [[ $pid -ne 0 ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "ERROR: $label (PID $pid) died during startup."
            echo "--- Last 30 lines of $logfile ---"
            tail -30 "$logfile" 2>/dev/null || true
            echo "--- End of $logfile ---"
            kill_gpu_processes
            return 1
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Timeout
    echo ""
    echo "ERROR: $label on port $port did not become ready within ${SERVER_START_TIMEOUT}s"
    echo "--- Last 30 lines of $logfile ---"
    tail -30 "$logfile" 2>/dev/null || true
    echo "--- End of $logfile ---"
    kill_gpu_processes
    return 1
}

cleanup() {
    local exit_code=$?
    echo ""
    echo "Shutting down (exit code: $exit_code)..."
    kill_gpu_processes
    echo "All processes stopped."
    exit $exit_code
}
trap cleanup INT TERM EXIT

# =============================================================================
# BUILD KV-TRANSFER-CONFIG JSON (Nixl uses kv_both for all instances)
# =============================================================================

build_nixl_kv_config() {
    # NixlConnector uses kv_role=kv_both for both prefill and decode.
    # The actual role (producer/consumer) is determined by the proxy.
    local extra=""
    if [[ -n "${NIXL_EXTRA_CONFIG:-}" ]]; then
        extra=",\"kv_connector_extra_config\":${NIXL_EXTRA_CONFIG}"
    fi
    echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"${KV_LOAD_FAILURE_POLICY}\"${extra}}"
}

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

echo "Checking dependencies..."
python3 -c "import nixl" 2>/dev/null || {
    echo "Installing nixl..."
    python3 -m pip install nixl
}
python3 -c "import httpx" 2>/dev/null || python3 -m pip install httpx
python3 -c "import uvicorn" 2>/dev/null || python3 -m pip install uvicorn fastapi
echo "Dependencies OK."

# =============================================================================
# LAUNCH vLLM INSTANCES
# =============================================================================

mkdir -p "$OUTPUT_DIR"

# Parse comma-separated lists into arrays
IFS=',' read -ra PREFILL_GPU_ARR <<< "$PREFILL_GPUS"
IFS=',' read -ra DECODE_GPU_ARR <<< "$DECODE_GPUS"
IFS=',' read -ra PREFILL_PORT_ARR <<< "$PREFILL_PORTS"
IFS=',' read -ra DECODE_PORT_ARR <<< "$DECODE_PORTS"

NIXL_KV_CONFIG=$(build_nixl_kv_config)

echo ""
echo "=============================================="
echo "  Launching Nixl PD-separated vLLM"
echo "  Model:      $MODEL"
echo "  Prefill:    ${#PREFILL_GPU_ARR[@]}x GPU [${PREFILL_GPUS}], ports [${PREFILL_PORTS}]"
echo "  Decode:     ${#DECODE_GPU_ARR[@]}x GPU [${DECODE_GPUS}], ports [${DECODE_PORTS}]"
echo "  Proxy:      port $PROXY_PORT"
echo "  Connector:  NixlConnector"
echo "  UCX_TLS:    $UCX_TLS"
echo "  Output dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

PRE_ARGS_HOST=()
PRE_ARGS_PORT=()
PRE_PIDS=()
DEC_ARGS_HOST=()
DEC_ARGS_PORT=()
DEC_PIDS=()

# -- Launch prefill instances --
echo "Starting ${#PREFILL_GPU_ARR[@]} prefill instance(s)..."
for i in "${!PREFILL_GPU_ARR[@]}"; do
    gpu="${PREFILL_GPU_ARR[$i]}"
    port="${PREFILL_PORT_ARR[$i]}"
    nixl_port=$((NIXL_SIDE_CHANNEL_BASE_PORT + i))

    # vLLM engine args
    engine_args=(
        --host 0.0.0.0
        --port "$port"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TP"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --dtype "$DTYPE"
        --kv-cache-dtype "$KV_CACHE_DTYPE"
        --load-format "$LOAD_FORMAT"
        --seed "$SEED"
        --trust-remote-code
        --kv-transfer-config "$NIXL_KV_CONFIG"
    )
    [[ "$ENFORCE_EAGER" == "1" ]] && engine_args+=(--enforce-eager)
    [[ "$NO_ENABLE_PREFIX_CACHING" == "1" ]] && engine_args+=(--no-enable-prefix-caching)

    # Build environment prefix dynamically
    env_prefix=(
        env CUDA_VISIBLE_DEVICES="$gpu"
        VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port"
        VLLM_NIXL_SIDE_CHANNEL_HOST="$VLLM_HOST_IP"
    )
    [[ "$HF_HUB_OFFLINE" == "1" ]] && env_prefix+=(HF_HUB_OFFLINE=1)
    [[ -n "${UCX_TLS:-}" ]] && env_prefix+=(UCX_TLS="$UCX_TLS")
    [[ -n "${UCX_NET_DEVICES:-}" ]] && env_prefix+=(UCX_NET_DEVICES="$UCX_NET_DEVICES")

    logfile="$OUTPUT_DIR/prefill${i}.log"
    echo "  Prefill[$i]: GPU=$gpu port=$port nixl_port=$nixl_port  log=$logfile"
    "${env_prefix[@]}" vllm serve "$MODEL" "${engine_args[@]}" \
        > "$logfile" 2>&1 &
    PRE_PIDS+=($!)

    PRE_ARGS_HOST+=("$VLLM_HOST_IP")
    PRE_ARGS_PORT+=("$port")
done

# -- Launch decode instances --
echo "Starting ${#DECODE_GPU_ARR[@]} decode instance(s)..."
for i in "${!DECODE_GPU_ARR[@]}"; do
    gpu="${DECODE_GPU_ARR[$i]}"
    port="${DECODE_PORT_ARR[$i]}"
    # Decode NIXL ports start after the last prefill port
    nixl_port=$((NIXL_SIDE_CHANNEL_BASE_PORT + ${#PREFILL_GPU_ARR[@]} + i))

    engine_args=(
        --host 0.0.0.0
        --port "$port"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TP"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --dtype "$DTYPE"
        --kv-cache-dtype "$KV_CACHE_DTYPE"
        --load-format "$LOAD_FORMAT"
        --seed "$SEED"
        --trust-remote-code
        --kv-transfer-config "$NIXL_KV_CONFIG"
    )
    [[ "$ENFORCE_EAGER" == "1" ]] && engine_args+=(--enforce-eager)
    [[ "$NO_ENABLE_PREFIX_CACHING" == "1" ]] && engine_args+=(--no-enable-prefix-caching)

    # Build environment prefix dynamically
    env_prefix=(
        env CUDA_VISIBLE_DEVICES="$gpu"
        VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port"
        VLLM_NIXL_SIDE_CHANNEL_HOST="$VLLM_HOST_IP"
    )
    [[ "$HF_HUB_OFFLINE" == "1" ]] && env_prefix+=(HF_HUB_OFFLINE=1)
    [[ -n "${UCX_TLS:-}" ]] && env_prefix+=(UCX_TLS="$UCX_TLS")
    [[ -n "${UCX_NET_DEVICES:-}" ]] && env_prefix+=(UCX_NET_DEVICES="$UCX_NET_DEVICES")

    logfile="$OUTPUT_DIR/decode${i}.log"
    echo "  Decode[$i]: GPU=$gpu port=$port nixl_port=$nixl_port  log=$logfile"
    "${env_prefix[@]}" vllm serve "$MODEL" "${engine_args[@]}" \
        > "$logfile" 2>&1 &
    DEC_PIDS+=($!)

    DEC_ARGS_HOST+=("$VLLM_HOST_IP")
    DEC_ARGS_PORT+=("$port")
done

# -- Wait for all servers --
echo ""
for i in "${!PREFILL_PORT_ARR[@]}"; do
    wait_for_server "${PREFILL_PORT_ARR[$i]}" "Prefill[$i]" \
        "${PRE_PIDS[$i]:-0}" "$OUTPUT_DIR/prefill${i}.log" || exit 1
done
for i in "${!DECODE_PORT_ARR[@]}"; do
    wait_for_server "${DECODE_PORT_ARR[$i]}" "Decode[$i]" \
        "${DEC_PIDS[$i]:-0}" "$OUTPUT_DIR/decode${i}.log" || exit 1
done

# -- Launch Nixl proxy --
echo ""
echo "Starting Nixl proxy server (port $PROXY_PORT)..."
# Build proxy args with all values after a single flag (nargs="+" style)
PROXY_CMD=(
    python3 "$SCRIPT_DIR/nixl_proxy.py"
    --port "$PROXY_PORT"
    --host 0.0.0.0
    --prefiller-hosts "${PRE_ARGS_HOST[@]}"
    --prefiller-ports "${PRE_ARGS_PORT[@]}"
    --decoder-hosts "${DEC_ARGS_HOST[@]}"
    --decoder-ports "${DEC_ARGS_PORT[@]}"
)

VLLM_NIXL_SIDE_CHANNEL_HOST="$VLLM_HOST_IP" \
    "${PROXY_CMD[@]}" \
    > "$OUTPUT_DIR/proxy.log" 2>&1 &

PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"
sleep 2

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: Proxy failed to start. Check $OUTPUT_DIR/proxy.log"
    cat "$OUTPUT_DIR/proxy.log"
    exit 1
fi
echo "Proxy is running. Health check:"
curl -s "http://localhost:${PROXY_PORT}/healthcheck" 2>/dev/null || echo "(healthcheck not available yet)"

# =============================================================================
# RUN BENCHMARK (reuses the same HTTP-based runner)
# =============================================================================

echo ""
echo "=============================================="
echo "  Running benchmark (via Nixl proxy)"
echo "  Dataset: $DATASET"
echo "  Proxy:   http://localhost:$PROXY_PORT"
echo "=============================================="
echo ""

cmd=(python3 -m bench run_disagg
    --model "$MODEL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --proxy-url "http://localhost:${PROXY_PORT}"
    --tick-seconds "$TICK_SECONDS"
    --num-reqs "$NUM_REQS"
    --log-level "$LOG_LEVEL"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
BENCH_EXIT_CODE=$?

echo ""
echo "Benchmark exited with code: $BENCH_EXIT_CODE"
echo "Done. Results in: $OUTPUT_DIR"
echo "  - meta.json"
echo "  - requests.jsonl"
echo "  - prefill*.log / decode*.log / proxy.log"

exit $BENCH_EXIT_CODE