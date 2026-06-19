#!/usr/bin/env python3
"""
Extract benchmark results and map to cluster config fields.

Usage:
    # Run benchmarks and print config-ready values
    python tools/mem_bench/export_to_config.py

    # Run benchmarks and patch an existing config file
    python tools/mem_bench/export_to_config.py --config configs/cluster/my_cluster.json

    # Patch in-place (overwrites the file)
    python tools/mem_bench/export_to_config.py --config configs/cluster/my_cluster.json --in-place

    # Dry-run: parse existing benchmark log without re-running
    python tools/mem_bench/export_to_config.py --log tools/mem_bench/logs/bench_20240608_120000.log

Values extracted and their config field mapping:
    stream  TRIAD GB/s          → cpu_mem.mem_bw
    latency 4KB-stride DRAM ns  → cpu_mem.mem_latency
    gpu_bench d2d bandwidth     → npu_mem.mem_bw
    gpu_bench HBM pointer chase → npu_mem.mem_latency
    gpu_bench H2D bandwidth     → (reference for link_bw on PCIe-only systems)
    gpu_bench 4B H2D+D2H RTT   → (reference for link_latency on PCIe-only systems)

Note: link_bw / link_latency represent inter-GPU (NVLink/PCIe) communication
bandwidth/latency, NOT CPU↔GPU PCIe. For PCIe-only multi-GPU systems, use the
H2D bandwidth and RTT as approximations.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR.parent / "bin"
LOG_DIR = SCRIPT_DIR / "logs"

# ── benchmark runner ──────────────────────────────────────────────────────

def run_bench(binary: str, args: list = None) -> str:
    """Run a benchmark binary and return its stdout."""
    bin_path = BIN_DIR / binary
    if not bin_path.exists():
        sys.exit(f"Binary not found: {bin_path}\nRun 'make -C {SCRIPT_DIR} all' first.")

    cmd = [str(bin_path)] + (args or [])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Warning: {binary} exited with code {result.returncode}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        return result.stdout
    except subprocess.TimeoutExpired:
        sys.exit(f"Timeout running {binary}")
    except FileNotFoundError:
        sys.exit(f"Cannot execute {binary}. Check dependencies (CUDA, OpenMP).")


# ── parsers ───────────────────────────────────────────────────────────────

def parse_stream(output: str) -> dict:
    """Parse STREAM output → {operation: GB/s}."""
    results = {}
    for line in output.splitlines():
        # "TRIAD              3.178   302088.1     302.09"
        m = re.match(r'^(COPY|SCALE|ADD|TRIAD)\s+[\d.]+\s+[\d.]+\s+([\d.]+)', line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results


def parse_latency(output: str) -> dict:
    """Parse latency output → {stride_label: {size_kb: {cycles, ns}}}.

    Returns the DRAM latency (ns) for the largest 4KB-stride buffer.
    """
    results = {}
    current_stride = None
    for line in output.splitlines():
        # "  Sequential access (64B stride) (stride=64, freq=2.10 GHz):"
        m = re.match(r'\s+(.+?)\s+\(stride=(\d+),', line)
        if m:
            current_stride = m.group(1)
            results[current_stride] = {}
            continue

        # "        16 KB       4.3 cycles     2.1 ns"
        m = re.match(r'\s+(\d+)\s+KB\s+([\d.]+)\s+cycles\s+([\d.]+)\s+ns', line)
        if m and current_stride:
            size_kb = int(m.group(1))
            results[current_stride][size_kb] = {
                "cycles": float(m.group(2)),
                "ns": float(m.group(3)),
            }
    return results


def parse_gpu_bench(output: str) -> dict:
    """Parse gpu_bench output → dict of structured results."""
    results = {
        "gpu_name": "",
        "d2d_bandwidth": {},
        "gpu_latency": {},
        "h2d_d2h_bandwidth": {},
        "h2d_d2h_latency": {},
    }

    # GPU name
    m = re.search(r'GPU:\s+(.+)', output)
    if m:
        results["gpu_name"] = m.group(1).strip()

    # D2D bandwidth: "      17 MB:     0.046 ms     729.2 GB/s"
    for m in re.finditer(r'^\s+([\d.]+)\s+MB:\s+([\d.]+)\s+ms\s+([\d.]+)\s+GB/s', output, re.MULTILINE):
        size_mb = float(m.group(1))
        bw = float(m.group(3))
        results["d2d_bandwidth"][size_mb] = bw

    # GPU latency: "      32 KB:    248 cycles    177.1 ns"
    for m in re.finditer(r'^\s+([\d.]+)\s+KB:\s+(\d+)\s+cycles\s+([\d.]+)\s+ns', output, re.MULTILINE):
        size_kb = float(m.group(1))
        cycles = int(m.group(2))
        ns = float(m.group(3))
        results["gpu_latency"][size_kb] = {"cycles": cycles, "ns": ns}

    # H2D/D2H bandwidth: "      1 MB        0.106        9.9      0.202        5.2"
    for m in re.finditer(
        r'^\s+([\d.]+)\s+MB\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        output, re.MULTILINE
    ):
        size_mb = float(m.group(1))
        h2d_ms = float(m.group(2))
        h2d_bw = float(m.group(3))
        d2h_ms = float(m.group(4))
        d2h_bw = float(m.group(5))
        results["h2d_d2h_bandwidth"][size_mb] = {
            "h2d_ms": h2d_ms, "h2d_gbps": h2d_bw,
            "d2h_ms": d2h_ms, "d2h_gbps": d2h_bw,
        }

    # H2D/D2H latency: "  4               3.81       5.01       6.91"
    for m in re.finditer(r'^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', output, re.MULTILINE):
        # Skip the header line "Bytes ..."
        if 'Bytes' in m.string.splitlines()[0]:
            continue
        try:
            size_bytes = int(m.group(1))
            h2d_us = float(m.group(2))
            d2h_us = float(m.group(3))
            rtt_us = float(m.group(4))
            results["h2d_d2h_latency"][size_bytes] = {
                "h2d_us": h2d_us, "d2h_us": d2h_us, "rtt_us": rtt_us,
            }
        except (ValueError, IndexError):
            continue

    return results


# ── mapping: benchmark results → config fields ────────────────────────────

def _parse_random_bw(output: str) -> float | None:
    """Parse the random-access bandwidth from the new stream output.

    Looks for the largest-size entry in the random-access table:
       4295 MB       146.1       0.44        6.8
    Returns the GB/s value (0.44).
    """
    best_bw = None
    best_size = 0
    # Match lines like "  4295 MB       146.1       0.44        6.8"
    # Only after the "Random-Access DRAM Bandwidth" header
    in_section = False
    for line in output.splitlines():
        if "Random-Access DRAM Bandwidth" in line:
            in_section = True
            continue
        if in_section and line.startswith("---"):
            continue
        if in_section:
            m = re.match(r'\s+(\d+)\s+MB\s+[\d.]+\s+([\d.]+)\s+[\d.]+', line)
            if m:
                size_mb = int(m.group(1))
                bw = float(m.group(2))
                if size_mb > best_size:
                    best_size = size_mb
                    best_bw = bw
    return best_bw


def build_config_values(stream_out: str, latency_out: str, gpu_out: str) -> dict:
    """Extract config-ready values from benchmark outputs."""
    stream_data = parse_stream(stream_out)
    latency_data = parse_latency(latency_out)
    gpu_data = parse_gpu_bench(gpu_out)

    values = {
        "_meta": {
            "gpu_name": gpu_data.get("gpu_name", "unknown"),
            "note": "Measured by tools/mem_bench. Review before using in simulation.",
        },
        "cpu_mem": {
            "mem_bw": None,
            "mem_latency": None,
        },
        "npu_mem": {
            "mem_bw": None,
            "mem_latency": None,
        },
        "link": {
            "bw": None,
            "latency": None,
        },
    }

    # ── cpu_mem.mem_bw: STREAM TRIAD GB/s (sequential DRAM bandwidth) ──
    if "TRIAD" in stream_data:
        bw = stream_data["TRIAD"]
        if bw > 200:
            print(f"Warning: STREAM TRIAD = {bw:.1f} GB/s is unusually high for DRAM.",
                  file=sys.stderr)
            print("  Working set may still fit in cache. Increase array size.", file=sys.stderr)
        values["cpu_mem"]["mem_bw"] = round(bw, 1)

    # Also parse random-access bandwidth for reference
    rand_bw = _parse_random_bw(stream_out)
    if rand_bw is not None:
        values["_meta"]["random_access_bw_gbps"] = round(rand_bw, 2)
        values["_meta"]["note"] += (
            f" Random-access BW: {rand_bw:.2f} GB/s (prefetcher-proof). "
            "Use STREAM sequential BW for cpu_mem.mem_bw.")

    # ── cpu_mem.mem_latency: 4KB-stride DRAM latency ──
    for label, data in latency_data.items():
        if "4KB" in label or "4096" in label or "page" in label.lower():
            if data:
                # Take the largest buffer (true DRAM latency, bypasses cache)
                max_size = max(data.keys())
                values["cpu_mem"]["mem_latency"] = round(data[max_size]["ns"], 1)
            break

    # ── npu_mem.mem_bw: largest successful D2D bandwidth ──
    if gpu_data["d2d_bandwidth"]:
        max_size = max(gpu_data["d2d_bandwidth"].keys())
        values["npu_mem"]["mem_bw"] = round(gpu_data["d2d_bandwidth"][max_size], 1)

    # ── npu_mem.mem_latency: HBM latency (largest buffer that fits in HBM) ──
    if gpu_data["gpu_latency"]:
        sizes = sorted(gpu_data["gpu_latency"].keys())
        # Skip L1/L2 sizes (< 2048 KB), take the HBM plateau
        hbm_sizes = [s for s in sizes if s >= 1024]
        if hbm_sizes:
            # Average the HBM entries (they should plateau)
            hbm_lat = sum(gpu_data["gpu_latency"][s]["ns"] for s in hbm_sizes) / len(hbm_sizes)
            values["npu_mem"]["mem_latency"] = round(hbm_lat, 1)

    # ── link_bw: H2D bandwidth (PCIe representative) ──
    if gpu_data["h2d_d2h_bandwidth"]:
        max_size = max(gpu_data["h2d_d2h_bandwidth"].keys())
        values["link"]["bw"] = round(gpu_data["h2d_d2h_bandwidth"][max_size]["h2d_gbps"], 1)

    # ── link_latency: 4B RTT (PCIe representative) ──
    if gpu_data["h2d_d2h_latency"]:
        if 4 in gpu_data["h2d_d2h_latency"]:
            # Convert microseconds to nanoseconds
            values["link"]["latency"] = round(gpu_data["h2d_d2h_latency"][4]["rtt_us"] * 1000, 0)

    return values


# ── output / patching ─────────────────────────────────────────────────────

def print_values(values: dict):
    """Pretty-print extracted values with field mapping."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Benchmark → Cluster Config Mapping                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    def p(label, value, unit, field):
        if value is not None:
            print(f"║  {label:<12} {value:>10.1f} {unit:<6} → {field:<30} ║")
        else:
            print(f"║  {label:<12} {'N/A':>10}        → {field:<30} ║")

    p("cpu_mem_bw",   values["cpu_mem"]["mem_bw"],      "GB/s",  "cpu_mem.mem_bw")
    p("cpu_mem_lat",  values["cpu_mem"]["mem_latency"], "ns",    "cpu_mem.mem_latency")
    p("npu_mem_bw",   values["npu_mem"]["mem_bw"],      "GB/s",  "npu_mem.mem_bw")
    p("npu_mem_lat",  values["npu_mem"]["mem_latency"], "ns",    "npu_mem.mem_latency")
    p("link_bw *",    values["link"]["bw"],             "GB/s",  "link_bw (PCIe H2D ref)")
    p("link_lat *",   values["link"]["latency"],        "ns",    "link_latency (PCIe RTT ref)")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  * link_bw/link_latency = inter-GPU communication.          ║")
    print("║    For PCIe-only systems, H2D BW & RTT are shown as ref.    ║")
    print("║    For NVLink systems, measure with nccl-tests instead.     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Also print the raw JSON snippet
    print()
    print("── JSON snippet (for cluster config) ──")
    snippet = {
        "cpu_mem": {
            "mem_size": 512,  # placeholder — fill manually
            "mem_bw": values["cpu_mem"]["mem_bw"],
            "mem_latency": values["cpu_mem"]["mem_latency"],
        },
        "npu_mem": {
            "mem_size": 96,  # placeholder — fill manually
            "mem_bw": values["npu_mem"]["mem_bw"],
            "mem_latency": values["npu_mem"]["mem_latency"],
        },
    }
    print(json.dumps(snippet, indent=2, ensure_ascii=False))
    print()
    print(f"  GPU: {values['_meta']['gpu_name']}")
    print(f"  {values['_meta']['note']}")
    print()


def patch_config(values: dict, config_path: str, in_place: bool = False):
    """Apply measured values to a cluster config JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    updated = []
    for node in config.get("nodes", []):
        # Apply to CPU memory
        if "cpu_mem" in node:
            if values["cpu_mem"]["mem_bw"] is not None:
                old = node["cpu_mem"].get("mem_bw", "?")
                node["cpu_mem"]["mem_bw"] = values["cpu_mem"]["mem_bw"]
                updated.append(f"cpu_mem.mem_bw: {old} → {values['cpu_mem']['mem_bw']}")
            if values["cpu_mem"]["mem_latency"] is not None:
                old = node["cpu_mem"].get("mem_latency", "?")
                node["cpu_mem"]["mem_latency"] = values["cpu_mem"]["mem_latency"]
                updated.append(f"cpu_mem.mem_latency: {old} → {values['cpu_mem']['mem_latency']}")

        # Apply to each instance's NPU memory
        for inst in node.get("instances", []):
            if "npu_mem" in inst:
                if values["npu_mem"]["mem_bw"] is not None:
                    old = inst["npu_mem"].get("mem_bw", "?")
                    inst["npu_mem"]["mem_bw"] = values["npu_mem"]["mem_bw"]
                    updated.append(f"instance[{inst.get('model_name','?')}].npu_mem.mem_bw: {old} → {values['npu_mem']['mem_bw']}")
                if values["npu_mem"]["mem_latency"] is not None:
                    old = inst["npu_mem"].get("mem_latency", "?")
                    inst["npu_mem"]["mem_latency"] = values["npu_mem"]["mem_latency"]
                    updated.append(f"instance[{inst.get('model_name','?')}].npu_mem.mem_latency: {old} → {values['npu_mem']['mem_latency']}")

    if not updated:
        print("No fields updated. Config structure might not match expected format.")
        return

    print("Fields updated:")
    for u in updated:
        print(f"  {u}")

    if in_place:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f"\nConfig written to: {config_path}")
    else:
        print("\n── Updated config preview ──")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"\nRun with --in-place to overwrite: {config_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract benchmark results → cluster config fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/mem_bench/export_to_config.py
  python tools/mem_bench/export_to_config.py --config configs/cluster/my_cluster.json
  python tools/mem_bench/export_to_config.py --config configs/cluster/my_cluster.json --in-place
  python tools/mem_bench/export_to_config.py --log tools/mem_bench/logs/bench_20240608.log
        """,
    )
    parser.add_argument("--config", "-c", help="Path to cluster config JSON to patch")
    parser.add_argument("--in-place", "-i", action="store_true",
                        help="Overwrite config file (default: preview only)")
    parser.add_argument("--log", help="Parse existing benchmark log instead of re-running")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmarks")
    args = parser.parse_args()

    if args.log:
        # Parse from existing log file
        with open(args.log, 'r') as f:
            log_text = f.read()
        # The log has all three outputs concatenated — parse each section
        stream_out = log_text
        latency_out = log_text
        gpu_out = log_text
    else:
        # Run benchmarks
        print("Running CPU memory bandwidth benchmark (stream)...")
        stream_out = run_bench("stream")
        print("Running CPU memory latency benchmark (latency)...")
        latency_out = run_bench("latency")
        if args.skip_gpu:
            gpu_out = ""
        else:
            print("Running GPU memory & communication benchmark (gpu_bench)...")
            gpu_out = run_bench("gpu_bench")

    values = build_config_values(stream_out, latency_out, gpu_out)
    print_values(values)

    if args.config:
        patch_config(values, args.config, args.in_place)


if __name__ == "__main__":
    main()
