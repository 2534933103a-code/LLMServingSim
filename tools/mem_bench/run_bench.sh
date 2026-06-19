#!/usr/bin/env bash
# ==========================================================================
# Memory & Communication Benchmark Runner
# 一键运行 CPU 内存、GPU 显存、CPU-GPU 通信的带宽和延迟测试
#
# 用法:
#   bash tools/mem_bench/run_bench.sh            # 运行全部测试
#   bash tools/mem_bench/run_bench.sh --cpu-only # 仅 CPU 测试
#   bash tools/mem_bench/run_bench.sh --gpu-only # 仅 GPU 测试
#   bash tools/mem_bench/run_bench.sh --export-config ../configs/cluster/my.json  # 导出到 config
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/../bin"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/bench_${TIMESTAMP}.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

CPU_ONLY=false
GPU_ONLY=false
EXPORT_CONFIG=""

for arg in "$@"; do
    case "$arg" in
        --cpu-only) CPU_ONLY=true ;;
        --gpu-only) GPU_ONLY=true ;;
        --export-config)
            EXPORT_CONFIG="next"
            ;;
        --help|-h)
            echo "Usage: $0 [--cpu-only] [--gpu-only] [--export-config <path>] [--help]"
            echo ""
            echo "  (no flag)              Run all benchmarks (CPU + GPU)"
            echo "  --cpu-only             Run CPU memory bandwidth & latency only"
            echo "  --gpu-only             Run GPU memory + CPU-GPU communication only"
            echo "  --export-config <path> After benchmarks, export results to cluster config JSON"
            exit 0
            ;;
        *)
            if [ "$EXPORT_CONFIG" = "next" ]; then
                EXPORT_CONFIG="$arg"
            else
                echo "Unknown option: $arg"; exit 1
            fi
            ;;
    esac
done

mkdir -p "$LOG_DIR"

# ---- banner ----
banner() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "  Memory & Communication Benchmark Suite"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo -e "${NC}"
}

# ---- system info ----
print_sysinfo() {
    echo -e "${YELLOW}[System Info]${NC}"
    echo "  Hostname: $(hostname)"
    echo "  Kernel:   $(uname -r)"
    if [ -f /proc/cpuinfo ]; then
        CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
        echo "  CPU:      ${CPU_MODEL}"
        echo "  Cores:    $(grep -c processor /proc/cpuinfo)"
    fi
    if [ -f /proc/meminfo ]; then
        MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1e6}')
        echo "  Memory:   ${MEM_TOTAL}"
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | \
            while IFS=, read -r idx name mem; do
                echo "  GPU ${idx}:     ${name} (${mem})"
            done
    fi
    echo ""
}

# ---- build ----
build() {
    echo -e "${YELLOW}[Build]${NC}"
    if ! make -C "$SCRIPT_DIR" -j all 2>&1; then
        echo -e "${RED}Build failed. Please check CUDA/OpenMP installation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}  Build OK${NC}"
    echo ""
}

# ---- run one bench and tee to log ----
run_bench() {
    local name="$1"
    local bin="$2"
    echo -e "${CYAN}------------------------------------------------------------${NC}"
    echo -e "${CYAN}[Running: ${name}]${NC}"
    echo ""
    if [ -x "$bin" ]; then
        "$bin" 2>&1 | tee -a "$LOG_FILE"
    else
        echo -e "${RED}  Binary not found: ${bin}${NC}"
    fi
    echo ""
}

# ---- main ----
banner | tee "$LOG_FILE"
print_sysinfo | tee -a "$LOG_FILE"
build | tee -a "$LOG_FILE"

if $GPU_ONLY; then
    run_bench "GPU Memory & CPU-GPU Communication" "${BIN_DIR}/gpu_bench"
elif $CPU_ONLY; then
    run_bench "CPU Memory Bandwidth (STREAM)"   "${BIN_DIR}/stream"
    run_bench "CPU Memory Latency"              "${BIN_DIR}/latency"
else
    run_bench "CPU Memory Bandwidth (STREAM)"   "${BIN_DIR}/stream"
    run_bench "CPU Memory Latency"              "${BIN_DIR}/latency"
    run_bench "GPU Memory & CPU-GPU Communication" "${BIN_DIR}/gpu_bench"
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  All benchmarks completed.${NC}"
echo -e "${GREEN}  Log saved to: ${LOG_FILE}${NC}"
echo -e "${GREEN}============================================================${NC}"

# ---- export to config ----
if [ -n "$EXPORT_CONFIG" ]; then
    echo ""
    echo -e "${YELLOW}[Export to Config]${NC}"
    python3 "${SCRIPT_DIR}/export_to_config.py" --log "$LOG_FILE" --config "$EXPORT_CONFIG"
    echo ""
    echo -e "${YELLOW}Review the values above, then run with --in-place to apply:${NC}"
    echo -e "  python3 ${SCRIPT_DIR}/export_to_config.py --log ${LOG_FILE} --config ${EXPORT_CONFIG} --in-place"
fi
