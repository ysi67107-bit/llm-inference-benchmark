
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    result_dir = Path("results")
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(result_dir.glob("matrix_summary_*.csv"))

    if not files:
        raise FileNotFoundError("No summary CSV found")

    csv_path = files[-1]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    concurrency = [int(r["concurrency"]) for r in rows]
    throughput = [float(r["mean_throughput"]) for r in rows]
    std = [float(r["std_throughput"]) for r in rows]

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        concurrency,
        throughput,
        yerr=std,
        marker="o",
        capsize=5,
        linewidth=2,
    )

    plt.xlabel("Concurrency")
    plt.ylabel("Request Throughput (req/s)")
    plt.title("Ollama Concurrency Benchmark")
    plt.xticks(concurrency)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = figure_dir / "matrix_repeated_throughput.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Source: {csv_path}")
    print(f"Saved: {output_path}")


    # Mean latency with standard deviation
    latency = [
        float(r["mean_latency"])
        for r in rows
    ]

    latency_std = [
        float(r["std_latency"])
        for r in rows
    ]

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        concurrency,
        latency,
        yerr=latency_std,
        marker="o",
        capsize=5,
        linewidth=2,
    )

    plt.xlabel("Concurrency")
    plt.ylabel("Mean Request Latency (s)")
    plt.title("Ollama Latency vs Concurrency")
    plt.xticks(concurrency)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    latency_path = figure_dir / "matrix_repeated_latency.png"
    plt.savefig(latency_path, dpi=200)
    plt.close()

    print(f"Saved: {latency_path}")



if __name__ == "__main__":
    main()
