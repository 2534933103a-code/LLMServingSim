#!/bin/bash

# 启动用于性能分析 / 基准测试 / 验证的 vLLM Docker 容器。
#
# 将 LLMServingSim 仓库根目录挂载为 /workspace，使性能分析器、
# 基准测试工具、数据集生成器以及共享的模型配置都可见：
#
#     /workspace/profiler/            性能分析包 + 脚本
#     /workspace/bench/               基准测试 + 验证
#     /workspace/workloads/           工作负载 JSONL 文件及生成器
#     /workspace/configs/model/       HuggingFace 模型配置
#
# 工作目录默认为 /workspace，因此任何模块都可以通过
# ``python -m profiler``、``python -m bench`` 等命令运行。
#
# 官方 vllm/vllm-openai 镜像已内置 vllm、pydantic、pyyaml、
# rich 和 huggingface_hub — 无需额外安装 pip 包。

set -euo pipefail

# 解析仓库根目录，无论此脚本从何处调用都能正确获取路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}" # 这个token只有llama2能用

docker run --name vllm_docker \
  --gpus all \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.19.0 \
  -c "pip install datasets matplotlib && exec bash"  -c "pip install datasets matplotlib && exec bash"
