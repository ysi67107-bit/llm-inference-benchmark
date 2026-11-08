
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def load_results(filename):
    groups = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["success"] != "True":
                continue

            target = int(row["target_output_tokens"])

            groups[target].append({
                "actual": int(row["eval_count"]),
                "decode_ms": float(
                    row["eval_duration"]
                ) * 1000,
                "speed": float(
                    row["decode_throughput"]
                )
            })

    summary = []

    for target, rows in sorted(groups.items()):
        summary.append({
            "target": target,
            "actual": mean(
                r["actual"] for r in rows
            ),
            "decode_ms": mean(
                r["decode_ms"] for r in rows
            ),
            "speed": mean(
                r["speed"] for r in rows
            )
        })

    return summary


def plot_decode(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = [r["actual"] for r in summary]
    durations = [r["decode_ms"] for r in summary]
    speeds = [r["speed"] for r in summary]

    # 图一：输出长度与 Decode 耗时
    plt.figure(figsize=(9, 5))

    plt.plot(tokens, durations, marker="o", linewidth=2)

    plt.xlabel("Output Tokens")
    plt.ylabel("Decode Duration (ms)")
    plt.title("Output Length vs Decode Duration")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    path1 = output_dir / "decode_duration.png"
    plt.savefig(path1, dpi=200)
    plt.close()

    # 图二：输出长度与 Decode 速度
    plt.figure(figsize=(9, 5))

    plt.plot(tokens, speeds, marker="o", linewidth=2)

    plt.xlabel("Output Tokens")
    plt.ylabel("Decode Throughput (tokens/s)")
    plt.title("Output Length vs Decode Throughput")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    path2 = output_dir / "decode_throughput.png"
    plt.savefig(path2, dpi=200)
    plt.close()

    print(f"Saved: {path1}")
    print(f"Saved: {path2}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/figures"
    )

    args = parser.parse_args()

    summary = load_results(args.input)

    if not summary:
        raise ValueError("No successful results found.")

    plot_decode(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
