#!/bin/bash
# -----------------------------------------------------------------------------
# 单次运行性能分析脚本。
#
# 此脚本设计为直接编辑：更改下方的变量为你当前想要分析的对象，然后执行：
#
#     ./profiler/profile.sh
#
# 分析器会根据模型配置中的 ``model_type`` 字段自动解析架构——你无需在此指定。
# 请确保在运行前，对应的架构 yaml 文件存在于 ``profiler/models/`` 目录下。
# -----------------------------------------------------------------------------

set -euo pipefail

# =============================================================================
# 编辑以下内容（必需）
# =============================================================================

# HuggingFace 风格的模型 ID。在 LLMServingSim 根目录下的
# ``configs/model/<MODEL>.json`` 路径中，必须存在原始的 HuggingFace config.json 文件。
# 分析器会从该配置中读取 model_type，以选择 profiler/models/ 目录下的架构 yaml 文件。
# MODEL="meta-llama/Llama-3.1-8B"
MODEL="Qwen/Qwen3-8B"

# GPU 标识符，将用作 ``perf/`` 目录下的输出文件夹名称。
# 自由格式 — 为你的硬件选择一个有意义的名称。
HARDWARE="A100-40GB"

# =============================================================================
# 编辑以下内容（可选 — 根据需要取消注释并调整）
# =============================================================================

# --- 张量并行度（TP）扫描 ------------------------------------------------------
# 逗号分隔的列表；必须包含 1。
TP_DEGREES="1,2"

# --- 引擎参数 ----------------------------------------------------------
# DTYPE 通常从模型配置的 ``torch_dtype`` 字段推断
# （目前 configs/model/ 下的所有模型都使用 bfloat16）。
# 只有在需要强制使用不同的权重数据类型时才显式设置它。
# KV_CACHE_DTYPE 默认为 "auto"，即继承 DTYPE。
# DTYPE="bfloat16"                 # bfloat16 / float16 / float32 / fp8
# KV_CACHE_DTYPE="fp8"             # auto / fp8 / fp16 / bf16
MAX_NUM_BATCHED_TOKENS=2048      # vLLM 的 --max-num-batched-tokens 参数
MAX_NUM_SEQS=256                 # vLLM 的 --max-num-seqs 参数

# --- 注意力网格参数 ---------------------------------------------------------
# kv_prefill / kv_decode 轴的上限。网格从 512 开始，按几何级数增长，
# 直至 min(此值, max_model_len)。
ATTENTION_MAX_KV=40960 #max-model-len
# prefill_chunk 轴的几何增长因子（从 16 增长到 MAX_NUM_BATCHED_TOKENS）。
# 2.0 表示加倍；使用更小的值可以在二次方代价区域获得更密集的采样，
# 但会延长分析时间。
ATTENTION_CHUNK_FACTOR=2.0 #表示从16开始增加，16，32，64，，，，MAX_NUM_BATCHED_TOKENS
# kv_prefill / kv_decode 轴的几何增长因子。2.0 表示加倍；
# 使用更小的值可以在长上下文场景获得更密集的覆盖。
ATTENTION_KV_FACTOR=2.0 #表示从512开始，512，1024，2048，，，ATTENTION_MAX_KV

# --- 测量平均次数 --------------------------------------------------
# 每次测试的计时前向传播次数（通过 vLLM 的 layerwise_profile 及其调用次数进行平均）。
# 由于 DVFS / 睿频时钟抖动，单次采样在大型 GEMM 上可能会有 15-25% 的波动；
# N=3（默认值）可将波动减少到约 5%，但分析时间会增加约 3 倍。
MEASUREMENT_ITERATIONS=3

# --- 偏斜（Skew）性能分析 ---------------------------------------------------------
# 在均匀注意力网格之后，还会分析异构的解码 KV 批次（每个 TP 需要 1-2 小时）。
# 这是模拟器用于预测偏斜批次所需的 alpha 公式拟合所必需的。
# 设置 SKIP_SKEW=1 可禁用此功能。
# SKIP_SKEW=1
#
# 偏斜扫描的每个轴的几何增长因子。2.0（默认值）表示加倍。
# 对于你不关心的轴，可以调高此值（例如 kvs / kp 设为 4.0）以粗略化采样，减少分析时间。
# 如果需要在需要更高精度的区域进行更密集的采样，则使用更小的值。
SKEW_N_FACTOR=4.0
SKEW_PC_FACTOR=4.0
SKEW_KP_FACTOR=4.0
SKEW_KVS_FACTOR=4.0

# --- 恢复模式 vs 强制重新分析 -------------------------------------------------------
# 默认：恢复模式。现有的 CSV 文件会被预加载，只有键值尚不存在的测试项才会被执行。
# 这让你可以在更改可行性条件后（例如添加 pc=2048 的情况），用几分钟而不是几小时来扩展现有的分析结果。
# 适用于所有类别以及偏斜分析。
# 设置 FORCE=1 会清空每个 CSV 文件并从头开始重新分析。
# FORCE=1

# --- 输出命名 ----------------------------------------------------------
# 如果省略，变体文件夹将根据实际的 DTYPE + KV_CACHE_DTYPE 自动命名 —
# 例如 "bf16"（默认）、"bf16-kvfp8"（FP8 KV cache）、"fp8-kvfp8"（两者均为 FP8）。
# 当 DTYPE 未设置时，会从模型的 ``torch_dtype`` 中获取，因此你无需设置任何值即可获得有意义的名称。
# 仅为指定名称的运行（awq、gptq……）覆盖 VARIANT 变量。
# VARIANT="my_experiment"

# --- 详细程度 --------------------------------------------------------------
# 默认为 INFO（显示进度 + TP 限制）。取消注释以下选项之一进行更改：
# VERBOSITY="--silent"             # 仅显示警告
# VERBOSITY="--verbose"            # 显示 DEBUG + vLLM 的标准输出

# =============================================================================
# 执行 — 通常不需要修改下面这一行之后的内容。
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 构建 Python 命令，仅包含已设置的标志。
cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")

[[ -n "${TP_DEGREES:-}" ]]             && cmd+=(--tp "$TP_DEGREES")
[[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "${MAX_NUM_SEQS:-}" ]]           && cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -n "${ATTENTION_MAX_KV:-}" ]]       && cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
[[ -n "${ATTENTION_CHUNK_FACTOR:-}" ]] && cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
[[ -n "${ATTENTION_KV_FACTOR:-}" ]]    && cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
[[ -n "${MEASUREMENT_ITERATIONS:-}" ]] && cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
[[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
[[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
[[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
[[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
[[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
[[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)

"${cmd[@]}"