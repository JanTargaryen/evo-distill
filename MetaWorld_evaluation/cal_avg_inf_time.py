import os
import re
import argparse
import glob
import numpy as np

def parse_log_file(file_path):
    latencies = []
    success_rates = []
    
    # 正则匹配 latency=123.45ms 和 success_rate=1.000
    latency_pattern = re.compile(r'latency=([\d\.]+)ms')
    success_pattern = re.compile(r'success_rate=([\d\.]+)')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取 Latency
                lat_match = latency_pattern.search(line)
                if lat_match:
                    latencies.append(float(lat_match.group(1)))
                
                # 提取 Success Rate (用于辅助验证)
                succ_match = success_pattern.search(line)
                if succ_match:
                    success_rates.append(float(succ_match.group(1)))
                    
    except Exception as e:
        print(f"[Error] Reading {file_path}: {e}")
        return None

    if not latencies:
        return None

    return {
        "file": os.path.basename(file_path),
        "count": len(latencies),
        "avg_latency": np.mean(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "avg_success": np.mean(success_rates) if success_rates else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate average latency from MetaWorld logs.")
    parser.add_argument("path", type=str, help="Path to a log file or a directory containing log files")
    args = parser.parse_args()

    # 确定要处理的文件列表
    files = []
    if os.path.isfile(args.path):
        files.append(args.path)
    elif os.path.isdir(args.path):
        # 递归查找所有 .txt 文件
        files = glob.glob(os.path.join(args.path, "**/*.txt"), recursive=True)
        # 也可以包含 .log 文件
        files += glob.glob(os.path.join(args.path, "**/*.log"), recursive=True)
    else:
        print(f"Invalid path: {args.path}")
        return

    all_latencies_summary = []
    
    print(f"{'File Name':<40} | {'Count':<5} | {'Avg Latency (ms)':<18} | {'Avg Success':<12}")
    print("-" * 85)

    for file_path in sorted(files):
        stats = parse_log_file(file_path)
        if stats:
            print(f"{stats['file']:<40} | {stats['count']:<5} | {stats['avg_latency']:.2f} ms{'':<8} | {stats['avg_success']:.3f}")
            # 保存该文件的平均值，用于计算总平均
            # 注意：这里我们应该保存*加权*平均，或者直接把所有raw latency存起来算总平均
            # 简单起见，这里重新解析一次或者假设每个任务权重一样。
            # 为了精确，我们记录 (sum, count)
            all_latencies_summary.append((stats['avg_latency'] * stats['count'], stats['count']))

    print("-" * 85)
    
    if all_latencies_summary:
        total_time = sum(x[0] for x in all_latencies_summary)
        total_count = sum(x[1] for x in all_latencies_summary)
        grand_avg = total_time / total_count if total_count > 0 else 0
        print(f"TOTAL PROCESSED FILES: {len(all_latencies_summary)}")
        print(f"TOTAL TASKS EVALUATED: {total_count}")
        print(f"OVERALL AVERAGE LATENCY: \033[92m{grand_avg:.2f} ms\033[0m")
    else:
        print("No valid log data found.")

if __name__ == "__main__":
    main()