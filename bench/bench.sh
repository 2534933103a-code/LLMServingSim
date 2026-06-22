#!/bin/bash
# Run a vLLM benchmark and write results under bench/results/<run_id>/.
#
# This is a thin host-side wrapper that:
#   1. Resolves the repo root.
#   2. Launches python -m bench run inside the vLLM Docker container
#      (or the local uv venv from scripts/install-vllm.sh).
#
# Edit the variables below for your run, then execute:
#
#     ./bench/bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# EDIT THESE
# =============================================================================
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-workloads/myllama-2-7b-heavy.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-bench/results/$RUN_ID}"

TP="${TP:-2}"
DP="${DP:-1}"
PP="${PP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"   # blank => use the model default
DTYPE="${DTYPE:-float16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"  # 'auto' to download real weights
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-0.5}"
NUM_REQS="${NUM_REQS:-0}"            # 0 => replay the full dataset
LOG_LEVEL="${LOG_LEVEL:-INFO}"

EXPERT_PARALLEL="${EXPERT_PARALLEL:-0}"   # 1 to enable for MoE

# =============================================================================
# EXECUTE
# =============================================================================
mkdir -p "$OUTPUT_DIR"

cmd=(python3 -m bench run
    --model "$MODEL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --pipeline-parallel-size "$PP"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --load-format "$LOAD_FORMAT"
    --seed "$SEED"
    --tick-seconds "$TICK_SECONDS"
    --num-reqs "$NUM_REQS"
    --log-level "$LOG_LEVEL"
    # --no-enable-prefix-caching
)

[[ -n "$MAX_MODEL_LEN" ]] && cmd+=(--max-model-len "$MAX_MODEL_LEN")
[[ "$EXPERT_PARALLEL" == "1" ]] && cmd+=(--enable-expert-parallel)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Results in: $OUTPUT_DIR"