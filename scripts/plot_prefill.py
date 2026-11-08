
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def load_results(filename):
    results = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["success"] != "True":
                continue

            results.append({
                "target_length": int(row["target_length"]),
                "prompt_tokens": int(row["prompt_eval_count"]),
                "cached_tokens": int(
                    row["prompt_eval_cached_count"] or 0
                ),
                "prefill_seconds": float(
                    row["prompt_eval_duration"]
                ),
                "decode_seconds": float(
                    row["eval_duration"]
                ),
                "output_tokens": int(row["eval_count"]),
            })

    return results


def summarize(results):
    groups = defaultdict(list)

    for row in results:
        groups[row["target_length"]].append(row)

    summary = []

    for target, rows in sorted(groups.items()):
        tokens = mean(
            r["prompt_tokens"] for r in rows
        )

        prefill = mean(
            r["prefill_seconds"] for r in rows
        )

        cached = mean(
            r["cached_tokens"] for r in rows
        )

        output = mean(
            r["output_tokens"] for r in rows
        )

        summary.append({
            "target": target,
            "tokens": tokens,
            "prefill_ms": prefill * 1000,
            "cached": cached,
            "output": output,
            "throughput": tokens / prefill
        })

    return summary


def plot_results(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = [r["tokens"] for r in summary]
    prefill = [r["prefill_ms"] for r in summary]
    throughput = [r["throughput"] for r in summary]

    # 图一：输入 Token 数与 Prefill 耗时
    plt.figure(figsize=(9, 5))

    plt.plot(
        tokens,
        prefill,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Prompt Tokens")
    plt.ylabel("Prefill Duration (ms)")
    plt.title("Prompt Length vs Prefill Duration")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    path1 = output_dir / "prefill_duration.png"
    plt.savefig(path1, dpi=200)
    plt.close()

    # 图二：Prefill 处理速率
    plt.figure(figsize=(9, 5))

    plt.plot(
        tokens,
        throughput,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Prompt Tokens")
    plt.ylabel("Nominal Prefill Throughput (tokens/s)")
    plt.title("Prompt Length vs Prefill Throughput")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    path2 = output_dir / "prefill_throughput.png"
    plt.savefig(path2, dpi=200)
    plt.close()

    print(f"Saved: {path1}")
    print(f"Saved: {path2}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures"
    )

    args = parser.parse_args()

    results = load_results(args.input)
    summary = summarize(results)

    print("\n===== Prefill Performance =====")

    for row in summary:
        print(
            f"Prompt Tokens={row['tokens']:.0f}, "
            f"Cached={row['cached']:.1f}, "
            f"Output={row['output']:.1f}, "
            f"Prefill={row['prefill_ms']:.2f}ms, "
            f"Throughput={row['throughput']:.2f} tok/s"
        )

    plot_results(
        summary,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
