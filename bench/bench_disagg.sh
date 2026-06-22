#!/bin/bash
# Run a vLLM PD-separation benchmark and write results under bench/results/<run_id>/.
#
# This script mirrors bench/bench.sh but launches vLLM in disaggregated
# prefill (P/D) mode: one prefill instance + one decode instance, connected
# via P2pNcclConnector, plus a proxy that coordinates prefill → decode handoff.
#
# Usage:
#     ./bench/bench_disagg.sh
#
# Prerequisites:
#     - 2 GPUs (prefill on GPU 0, decode on GPU 1 by default)
#     - Inside the vLLM Docker container (launched via scripts/docker-vllm.sh)
#     - Quart must be installed: pip install quart aiohttp
#
# Architecture (ports):
#     prefill (8100) ──┐
#                      ├── proxy (8000) ── client (bench client)
#     decode  (8200) ──┘
#
# KV transfer: P2pNcclConnector over NCCL (ranks 0→1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../bench
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# EDIT THESE
# =============================================================================
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-workloads/myllama-2-7b.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-bench/results/$RUN_ID}"

# -- GPU assignment --
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"

# -- Port assignment --
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"

# -- vLLM engine parameters (shared by prefill & decode) --
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
DTYPE="${DTYPE:-float16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"  # 'auto' to download real weights
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
SEED="${SEED:-42}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # 1 = skip HuggingFace network check

# -- KV transfer parameters --
KV_CONNECTOR="${KV_CONNECTOR:-P2pNcclConnector}"
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-1e9}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-14579}"
DECODE_KV_PORT="${DECODE_KV_PORT:-14580}"
KV_PARALLEL_SIZE="${KV_PARALLEL_SIZE:-2}"

# -- Benchmark parameters --
TICK_SECONDS="${TICK_SECONDS:-1.0}"
NUM_REQS="${NUM_REQS:-0}"            # 0 => replay the full dataset
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# -- Host IP (can be overridden) --
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"

# -- Extra KV connector config (JSON object, optional) --
# For P2pNcclConnector you may need:
#   KV_EXTRA_CONFIG='{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8100","send_type":"PUT_ASYNC"}'
KV_EXTRA_CONFIG="${KV_EXTRA_CONFIG:-}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

kill_gpu_processes() {
    echo "Cleaning up GPU processes..."
    # 1. Kill by process name patterns
    pgrep -f "^VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f "disagg_proxy" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f "bench_disagg" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    # 2. Kill by port
    for port in "$PROXY_PORT" "$PREFILL_PORT" "$DECODE_PORT"; do
        lsof -ti:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    # 3. nvidia-smi safety net: kill any remaining GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 2
    echo "Cleanup done."
}

SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-60}"  # max seconds to wait for vLLM to start

wait_for_server() {
    local port=$1
    local label=$2
    echo "Waiting for $label at port $port (timeout: ${SERVER_START_TIMEOUT}s)..."
    timeout "$SERVER_START_TIMEOUT" bash -c "
        until curl -s -o /dev/null -w '%{http_code}' localhost:\"${port}\"/v1/models 2>/dev/null | grep -q '200'; do
            sleep 1
        done
    " && echo "$label on port $port is ready." || {
        echo "ERROR: $label on port $port did not become ready within ${SERVER_START_TIMEOUT}s"
        kill_gpu_processes
        return 1
    }
}

# Cleanup on exit / Ctrl+C
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
# BUILD KV-TRANSFER-CONFIG JSON
# =============================================================================

build_kv_config() {
    local role="$1"    # kv_producer or kv_consumer
    local rank="$2"    # 0 for prefill, 1 for decode
    local kv_port="$3" # 14579 or 14580

    local config="{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"${role}\",\"kv_rank\":${rank},\"kv_parallel_size\":${KV_PARALLEL_SIZE},\"kv_buffer_size\":\"${KV_BUFFER_SIZE}\",\"kv_port\":\"${kv_port}\""

    # Append kv_connector_extra_config if set
    if [[ -n "${KV_EXTRA_CONFIG:-}" ]]; then
        config+=",\"kv_connector_extra_config\":${KV_EXTRA_CONFIG}"
    fi

    config+="}"
    echo "$config"
}

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

echo "Checking dependencies..."
python3 -c "import quart" 2>/dev/null || {
    echo "Installing quart..."
    python3 -m pip install quart aiohttp
}
echo "Dependencies OK."

# =============================================================================
# LAUNCH vLLM INSTANCES
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "  Launching PD-separated vLLM"
echo "  Model:      $MODEL"
echo "  Prefill:    GPU $PREFILL_GPU, port $PREFILL_PORT"
echo "  Decode:     GPU $DECODE_GPU, port $DECODE_PORT"
echo "  Proxy:      port $PROXY_PORT"
echo "  KV host:    $VLLM_HOST_IP"
echo "  Connector:  $KV_CONNECTOR"
echo "  Output dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# -- Launch prefill instance (KV producer) --
echo "Starting prefill instance (GPU $PREFILL_GPU)..."
PREFILL_KV_CONFIG=$(build_kv_config "kv_producer" 0 "$PREFILL_KV_PORT")

CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
    ${HF_HUB_OFFLINE:+HF_HUB_OFFLINE=$HF_HUB_OFFLINE} \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TP" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --load-format "$LOAD_FORMAT" \
    --seed "$SEED" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    > "$OUTPUT_DIR/prefill.log" 2>&1 &

PREFILL_PID=$!
echo "  Prefill PID: $PREFILL_PID"

# -- Launch decode instance (KV consumer) --
echo "Starting decode instance (GPU $DECODE_GPU)..."
DECODE_KV_CONFIG=$(build_kv_config "kv_consumer" 1 "$DECODE_KV_PORT")

CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
    ${HF_HUB_OFFLINE:+HF_HUB_OFFLINE=$HF_HUB_OFFLINE} \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TP" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --load-format "$LOAD_FORMAT" \
    --seed "$SEED" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    > "$OUTPUT_DIR/decode.log" 2>&1 &

DECODE_PID=$!
echo "  Decode PID: $DECODE_PID"

# -- Wait for both servers --
echo ""
wait_for_server "$PREFILL_PORT" "Prefill instance" || {
    echo "Prefill instance failed to start. Check $OUTPUT_DIR/prefill.log"
    tail -50 "$OUTPUT_DIR/prefill.log"
    exit 1
}
wait_for_server "$DECODE_PORT" "Decode instance" || {
    echo "Decode instance failed to start. Check $OUTPUT_DIR/decode.log"
    tail -50 "$OUTPUT_DIR/decode.log"
    exit 1
}

# -- Launch proxy server --
echo ""
echo "Starting disagg proxy server (port $PROXY_PORT)..."
python3 "$SCRIPT_DIR/disagg_proxy.py" \
    --port "$PROXY_PORT" \
    --prefill-url "http://localhost:${PREFILL_PORT}" \
    --decode-url "http://localhost:${DECODE_PORT}" \
    --kv-host "$VLLM_HOST_IP" \
    --prefill-kv-port "$PREFILL_KV_PORT" \
    --decode-kv-port "$DECODE_KV_PORT" \
    > "$OUTPUT_DIR/proxy.log" 2>&1 &

PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"
sleep 2

# Verify proxy is running
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: Proxy failed to start. Check $OUTPUT_DIR/proxy.log"
    cat "$OUTPUT_DIR/proxy.log"
    exit 1
fi
echo "Proxy is running."

# =============================================================================
# RUN BENCHMARK
# =============================================================================

echo ""
echo "=============================================="
echo "  Running benchmark"
echo "  Dataset: $DATASET"
echo "  Proxy:   http://localhost:$PROXY_PORT"
echo "=============================================="
echo ""

cmd=(python3 -m bench run_disagg
    --model "$MODEL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --proxy-url "http://localhost:${PROXY_PORT}"
    --prefill-url "http://localhost:${PREFILL_PORT}"
    --decode-url "http://localhost:${DECODE_PORT}"
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
echo "  - prefill.log / decode.log / proxy.log"

# cleanup() runs automatically via trap on exit
exit $BENCH_EXIT_CODE
