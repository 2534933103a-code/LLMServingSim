import json
import csv
import statistics
from collections import Counter

# 输入输出文件路径
input_jsonl = "workloads/myllama-2-7b.jsonl"
output_csv = "/home/vipuser/wlvllm/myllama-2-7b.csv"

output_toks_list = []
input_toks_list = []

with open(input_jsonl, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

    writer = csv.writer(outfile)
    # 写入表头（与原CSV一致）
    writer.writerow(["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"])

    # timestamp = 1  # 从1开始递增，可随意
    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        req = data["input_toks"]
        resp = data["output_toks"]
        total = req + resp
        output_toks_list.append(resp)
        input_toks_list.append(req)

        # 如需使用 arrival_time_ns 作为时间戳（秒），可改为：
        timestamp = data["arrival_time_ns"] // 1_000_000_000  # 转换为秒
        # 但样例中数值很小(58658511)，看起来像是纳秒但实际可能不是绝对时间，故按顺序递增更简单

        writer.writerow([timestamp, "GPT-4", req, resp, total, "Conversation log"])
        # timestamp += 1

print(f"转换完成，输出文件：{output_csv}")

# --- 输入/输出长度分布 ---
def print_dist(name, data):
    n = len(data)
    if n == 0:
        print(f"\n{name}: 无数据")
        return
    s = sorted(data)
    print(f"\n========== {name} 长度分布 ==========")
    print(f"样本数: {n}")
    print(f"均值:   {statistics.mean(data):.1f}")
    print(f"中位数: {statistics.median(data):.1f}")
    print(f"最小值: {min(data)}")
    print(f"最大值: {max(data)}")
    print(f"标准差: {statistics.stdev(data):.1f}" if n > 1 else "标准差: N/A")

    def percentile(d, p):
        k = (len(d) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(d):
            return d[f] + c * (d[f + 1] - d[f])
        return d[f]

    print("--- 分位数 ---")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2d}: {percentile(s, p):.0f}")

    print("--- 直方图 (10 桶) ---")
    lo, hi = min(data), max(data)
    bin_width = max((hi - lo) / 10, 1)
    bins = [0] * 10
    for v in data:
        idx = min(int((v - lo) / bin_width), 9)
        bins[idx] += 1
    for i in range(10):
        low = int(lo + i * bin_width)
        high = int(lo + (i + 1) * bin_width)
        bar = "█" * max(1, bins[i] * 40 // max(bins))
        print(f"  [{low:>5d}, {high:>5d}]: {bins[i]:>5d}  {bar}")

print_dist("Input tokens", input_toks_list)
print_dist("Output tokens (Response)", output_toks_list)