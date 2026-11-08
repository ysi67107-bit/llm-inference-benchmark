
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    result_dir = Path("results")
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 找到最新的实验矩阵 CSV
    files = sorted(result_dir.glob("matrix_*.csv"))

    if not files:
        raise FileNotFoundError("No matrix CSV files found")

    csv_path = files[-1]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    concurrency = [
        int(row["concurrency"]) for row in rows
    ]

    throughput = [
        float(row["request_throughput"]) for row in rows
    ]

    latency = [
        float(row["mean_latency"]) for row in rows
    ]

    # 绘制吞吐量曲线
    plt.figure(figsize=(8, 5))
    plt.plot(concurrency, throughput, marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("Request Throughput (req/s)")
    plt.title("Ollama Throughput vs Concurrency")
    plt.xticks(concurrency)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    throughput_path = figure_dir / "matrix_throughput.png"
    plt.savefig(throughput_path, dpi=200)
    plt.close()

    # 绘制平均延迟曲线
    plt.figure(figsize=(8, 5))
    plt.plot(concurrency, latency, marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("Mean Request Latency (s)")
    plt.title("Ollama Latency vs Concurrency")
    plt.xticks(concurrency)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    latency_path = figure_dir / "matrix_latency.png"
    plt.savefig(latency_path, dpi=200)
    plt.close()

    print(f"Source: {csv_path}")
    print(f"Saved: {throughput_path}")
    print(f"Saved: {latency_path}")


if __name__ == "__main__":
    main()
