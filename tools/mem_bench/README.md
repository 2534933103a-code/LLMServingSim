# Memory & Communication Benchmark Tools

本目录包含 CPU 内存、GPU 显存、CPU-GPU 通信的带宽和延迟测试工具。

## 测试覆盖

| 测试项 | 工具 | 方法 |
|--------|------|------|
| **CPU 内存带宽（顺序）** | `stream` | STREAM 基准测试 (COPY/SCALE/ADD/TRIAD)，4.8GB 工作集确保到 DRAM |
| **CPU 内存带宽（随机）** | `stream` | 随机指针追逐吞吐——预取器失效时的真实 DRAM 带宽 |
| **CPU 内存延迟** | `latency` | 指针追逐 (pointer chasing)，覆盖 L1→L2→L3→DRAM |
| **GPU 显存带宽** | `gpu_bench` | device-to-device 拷贝内核 |
| **GPU 显存延迟** | `gpu_bench` | GPU 端指针追逐，覆盖 L2→HBM |
| **CPU↔GPU 通信带宽** | `gpu_bench` | `cudaMemcpy` H2D / D2H 大块传输 |
| **CPU↔GPU 通信延迟** | `gpu_bench` | `cudaMemcpy` 小包传输及往返延迟 (RTT) |

## 构建

### 前置条件

- **CPU 工具** (`stream`, `latency`)：GCC/Clang + OpenMP
- **GPU 工具** (`gpu_bench`)：NVIDIA CUDA Toolkit (`nvcc`)

### 编译

```bash
cd tools/mem_bench
make all
```

编译产物输出到 `tools/bin/`：

```
tools/bin/
├── stream       # CPU 内存带宽
├── latency      # CPU 内存延迟
└── gpu_bench    # GPU 显存带宽/延迟 + CPU-GPU 通信带宽/延迟
```

### 清理

```bash
make clean
```

## 使用

### 一键运行全部测试

```bash
bash tools/mem_bench/run_bench.sh
```

### 导出到 Cluster Config（推荐）

测量结果可以直接映射到仿真器的 cluster config JSON：

```bash
# 方式 1：一步完成（运行 benchmark + 预览 config 更新）
bash tools/mem_bench/run_bench.sh --export-config configs/cluster/single_node_single_instance.json

# 方式 2：只做提取（不重新跑 benchmark，用已有日志）
python3 tools/mem_bench/export_to_config.py --log tools/mem_bench/logs/bench_20240608.log

# 方式 3：直接覆写 config 文件
python3 tools/mem_bench/export_to_config.py --config configs/cluster/my_cluster.json --in-place
```

**Benchmark → Config 字段映射：**

| Benchmark 测量值 | Config 字段 | 说明 |
|---|---|---|
| `stream` TRIAD (GB/s) | `cpu_mem.mem_bw` | CPU DRAM 带宽 |
| `latency` 4KB stride DRAM (ns) | `cpu_mem.mem_latency` | CPU DRAM 延迟 |
| `gpu_bench` d2d bandwidth (GB/s) | `npu_mem.mem_bw` | GPU 显存带宽 |
| `gpu_bench` pointer chase HBM (ns) | `npu_mem.mem_latency` | GPU 显存延迟 |
| `gpu_bench` H2D bandwidth (GB/s) | `link_bw` *参考* | GPU↔GPU 通信带宽 |
| `gpu_bench` 4B RTT (ns) | `link_latency` *参考* | GPU↔GPU 通信延迟 |

> **注意**：`link_bw`/`link_latency` 表示 GPU↔GPU 通信（NVLink/PCIe），不是 CPU↔GPU。
> PCIe-only 多 GPU 系统可参考 H2D 带宽和 RTT；有 NVLink 的系统请使用 `nccl-tests` 测试。

### 单独运行各工具

```bash
# CPU 内存带宽 (STREAM: COPY/SCALE/ADD/TRIAD)
./tools/bin/stream

# CPU 内存延迟 (pointer chasing, 显示各级缓存延迟)
./tools/bin/latency

# GPU 显存 + CPU-GPU 通信 (需要 NVIDIA GPU)
./tools/bin/gpu_bench
```

## 输出解读

### CPU 内存带宽 (`stream`)

`stream` 输出两部分：

**STREAM（顺序访问，受益于预取器）：**

```
Operation    Time(ms)       MB/s        GB/s
-----------------------------------------------------
COPY          76.738    41700.3      41.70
SCALE         73.230    43697.9      43.70
ADD          100.863    47589.3      47.59
TRIAD        100.328    47843.1      47.84
```

- **TRIAD**（2 读 1 写）：最接近真实 HPC 负载，是 `cpu_mem.mem_bw` 的**推荐取值**
- 4.8 GB 总工作集保证数据在 DRAM 而非缓存中

**随机访问（预取器失效）：**

```
Size            Lat(ns)         GB/s      M ops/s
-----------------------------------------------------
   268 MB       102.5         0.62          9.8
  4295 MB       146.1         0.44          6.8
```

- 随机指针追逐吞吐，**比顺序 STREAM 慢 100 倍**
- 反映不规则访问模式（hash 查表、指针追逐）的真实 DRAM 瓶颈
- 仿真器用不到这个值，仅作为对照参考
- 结果反映 CPU → DRAM 的实际可用带宽（受内存控制器、通道数、频率影响）

### CPU 内存延迟 (`latency`)

```
Sequential access (64B stride):
     16 KB    4.2 cycles    1.2 ns    ← L1 cache
     64 KB    4.3 cycles    1.2 ns    ← L1 cache (末尾)
    256 KB   12.1 cycles    3.5 ns    ← L2 cache
    512 KB   12.5 cycles    3.6 ns    ← L2 cache
   2048 KB   42.0 cycles   12.0 ns    ← L3 cache
  16384 KB   45.0 cycles   12.8 ns    ← L3 cache (末尾)
  32768 KB  105.0 cycles   30.0 ns    ← DRAM
 262144 KB  110.0 cycles   31.4 ns    ← DRAM (稳定值)
```

- 不同 buffer 大小反映不同层级缓存的延迟
- 32KB 以内 → L1，128KB-512KB → L2，2MB-16MB → L3，>32MB → DRAM
- DRAM 延迟通常在 60-120 ns（DDR4）或 80-140 ns（DDR5）

### GPU 显存带宽 (`gpu_bench`)

```
GPU Device-to-Device Bandwidth:
    16 MB:     0.023 ms     1400.0 GB/s    ← HBM 带宽
    64 MB:     0.089 ms     1438.0 GB/s
   256 MB:     0.356 ms     1438.0 GB/s
   512 MB:     0.710 ms     1442.0 GB/s
  1024 MB:     1.421 ms     1440.0 GB/s

GPU Device Memory Latency:
    32 KB:    28 cycles    20.0 ns    ← L1 cache
   128 KB:    28 cycles    20.0 ns
  1024 KB:    88 cycles    63.0 ns    ← L2 cache
  4096 KB:    90 cycles    64.0 ns
 16384 KB:   280 cycles   200.0 ns    ← HBM
 65536 KB:   310 cycles   221.0 ns
262144 KB:   315 cycles   225.0 ns
```

- HBM 带宽典型值：A100 约 1555 GB/s，H100 约 2039 GB/s，RTX 3090 约 936 GB/s
- HBM 延迟通常 200-400 ns（远高于 DRAM，但带宽极高）

### CPU-GPU 通信 (`gpu_bench`)

```
CPU-GPU Communication Bandwidth:
  Size        H2D ms     H2D GB/s     D2H ms     D2H GB/s
----------------------------------------------------------------
    1 MB      0.082        12.2        0.045        22.2
    8 MB      0.625        12.8        0.332        24.1
   64 MB      4.998        12.8        2.510        25.5
  256 MB     20.012        12.8       10.044        25.5

CPU-GPU Communication Latency:
  Bytes       H2D us       D2H us       RTT us
-------------------------------------------------------
  4            5.20         3.80         9.50
  64           5.30         3.90         9.60
  1024         5.80         4.20        10.50
  16384       15.20        10.30        26.00
  65536       60.50        30.20        91.00
```

- **H2D/D2H 带宽** 受 PCIe 限制：Gen4 x16 ≈ 25 GB/s，Gen5 x16 ≈ 50 GB/s
- **小包延迟**：4 字节传输约 5-10 μs（PCIe 链路延迟 + 内核开销）
- **RTT**（往返延迟）：H2D + D2H 顺序执行的总延迟，是评估 CPU-GPU 同步开销的核心指标
- 注意 D2H 带宽通常略高于 H2D，这是 PCIe 不对称特性

## 技术细节

### CPU 延迟测试方法
使用 pointer chasing（指针追逐）技术：在 buffer 内构建链表，通过 `p = *p` 强制串行依赖，
阻止 CPU 预取和乱序执行，测量真实的访存延迟。

### GPU 延迟测试方法
GPU 端同样使用 pointer chasing，用 `clock64()` 读取 GPU 时钟周期数，转换为纳秒。
不同 buffer 大小观测 L1 / L2 / HBM 的延迟层级。

### 带宽测试注意事项
- 关闭 CPU 频率调节 (cpufreq governor → performance) 可获得稳定结果
- GPU 测试前确保没有其他进程占用 GPU (`nvidia-smi`)
- 多次运行取最优值可以排除 OS 调度抖动的影响
