# bench

端到端 vLLM 基准测试 + 仿真器验证。运行真实 vLLM serving 负载，
采集每个请求的时序和每 tick 的调度器状态，并将结果与仿真器对
同一数据集的输出进行对比。

## 目录结构

```
bench/                          Python 包 — `python -m bench ...`
├── __init__.py                 包标记 + 模块映射
├── __main__.py                 CLI 调度（run / validate）
├── core/                       内部实现
│   ├── runner.py               AsyncLLM 驱动，采集 RequestStateStats
│   ├── recorder.py             写入 meta.json / requests.jsonl / timeseries.csv
│   ├── stat_logger.py          自定义 vLLM StatLoggerBase，填充 timeseries
│   ├── validate.py             bench-vs-sim 对比入口
│   ├── plots.py                吞吐量 / running-waiting / 延迟-CDF 绘图工具
│   └── logger.py               Rich 日志 + 标准输出捕获
├── bench.sh                    主机侧 ``python -m bench run`` 包装脚本
├── validate.sh                 主机侧 ``python -m bench validate`` 包装脚本
├── examples/                   标准端到端运行示例（已提交的产物）
│   ├── configs/<model>.json    仿真器侧使用的 cluster config
│   ├── <model>/vllm/           vLLM bench 产物（meta.json, requests.jsonl, timeseries.csv）
│   ├── <model>/outputs/        仿真器输出（sim.csv, sim.log）
│   ├── <model>/validation/     `bench validate` 输出（PDF + summary.txt）
│   ├── run.sh                  重跑部分或全部示例的仿真器侧
│   └── validate.sh             重跑部分或全部示例的验证步骤
└── results/                    临时运行输出根目录：bench/results/<run_id>/
```

## 用法

### `bench run` — 对已有数据集进行严格回放

Runner 读取 LLMServingSim 格式的 JSONL（与 `python -m workloads.generators`
生成、`python -m serving --dataset` 消费的格式相同）。每个请求的
`input_tok_ids` 和 `output_toks` 通过
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)` 固定，
确保 vLLM 运行结果与仿真器对同一负载的视角可以逐位对比。

```bash
# 在 vLLM 容器内（scripts/docker-vllm.sh）。
./bench/bench.sh
# 或直接调用模块并显式传参：
python -m bench run \
    --model <hf-id-or-path> \
    --dataset workloads/<workload>.jsonl \
    --output-dir bench/results/<run_id> \
    --tensor-parallel-size 1 --data-parallel-size 1 \
    --max-num-seqs 128 --max-num-batched-tokens 2048 \
    --dtype bfloat16 --kv-cache-dtype auto
```

### `bench validate` — 将已完成的 bench 运行结果与仿真器输出对比

加载 bench 产物以及同一负载的仿真器 `sim.csv` / `sim.log`，
在统一定义下计算双方的 TTFT / TPOT / 端到端延迟，
并将图表和数值汇总写入 bench 运行的子目录中。

```bash
./bench/validate.sh \
    bench/results/<run_id> \
    outputs/<sim-run>/sim.csv \
    outputs/<sim-run>/sim.log \
    [prefix]
```

## 输出格式（单次 bench run）

```
bench/results/<run_id>/
  meta.json            运行元数据（模型、vLLM 版本、engine 参数、
                       数据集哈希、挂钟开始/结束时间）
  requests.jsonl       每个请求的时序 — request_id, input_toks,
                       output_toks, arrival_time, queued_ts, scheduled_ts,
                       first_token_ts, last_token_ts
  timeseries.csv       每 tick 聚合 — t, prompt_throughput,
                       gen_throughput, running, waiting, kv_cache_pct
  validation/          （由 `bench validate` 创建）
    <prefix>_throughput.png
    <prefix>_requests.png
    <prefix>_latency.png
    <prefix>_summary.txt
```

## 延迟定义（sim ↔ bench）

双方基于相同的参考点上报 TTFT、TPOT 和端到端延迟，
因此 diff% 具有可比意义：

| 指标 | 定义 |
| --- | --- |
| `TTFT`     | `first_token_ts - arrival_time`（含排队等待） |
| `TPOT`     | `(last_token_ts - first_token_ts) / max(1, output_toks - 1)` |
| `Latency`  | `last_token_ts - arrival_time` |

仿真器的 `sim.csv` 直接暴露 `arrival`、`end_time` 以及每个 token
的 ITL 列表；bench 从 vLLM 的 `RequestStateStats`
（`vllm/v1/metrics/stats.py`）计算相同字段。

## 标准示例（`bench/examples/`）

`bench/examples/` 下提交了三组端到端验证运行，覆盖 dense 单 GPU
基线、TP=2 dense 运行以及 DP+EP MoE 运行。每个示例包含 vLLM bench
产物、仿真器输出以及对应的 `bench validate` 汇总 + 图表。

| 示例 | 并行策略 | 负载（300 请求） | TTFT 均值 | TPOT 均值 | Latency 均值 |
| --- | --- | --- | --- | --- | --- |
| `Llama-3.1-8B`                | TP=1 dense              | `sharegpt-llama-3.1-8b-300-sps10.jsonl`     | -2.8% | -0.3% | -1.0% |
| `Qwen3-32B`                   | TP=2 dense              | `sharegpt-qwen3-32b-300-sps10.jsonl`        | -0.7% | -0.3% | -0.4% |
| `Qwen3-30B-A3B-Instruct-2507` | DP=2, EP=2 MoE          | `sharegpt-qwen3-30b-a3b-300-sps10.jsonl`    | -2.9% | +0.6% | +0.4% |

Diff% 公式为 `(sim - vLLM) / vLLM × 100`。三组运行均在 RTXPRO6000
上进行，使用 `bf16` 权重，`max_num_seqs=128`，`max_num_batched_tokens=2048`，
`block_size=16`，负载由 `python -m workloads.generators` 生成
（ShareGPT，单轮对话，vLLM 自由生成模式）。各百分位细分数据
（P50 / P90 / P95 / P99）见各
`bench/examples/<model>/validation/summary.txt`。

复现标准示例：

```bash
# 在仿真器容器内：
./bench/examples/run.sh                       # 全部三个示例
./bench/examples/run.sh Qwen3-30B-A3B-Instruct-2507   # 单个示例

# 然后与已提交的 vLLM 产物进行验证：
./bench/examples/validate.sh
./bench/examples/validate.sh Qwen3-30B-A3B-Instruct-2507
```

`run.sh` 读取每个示例的 `meta.json`（engine 参数 + 数据集路径）
以及 `bench/examples/configs/` 下对应的 cluster config，因此
仿真器能够针对与原始 vLLM bench 完全相同的负载和 engine 配置
运行。如需从头重新生成 vLLM 侧数据，在 vLLM 容器内使用
`bench/bench.sh`（或 `python -m bench run`）。
