
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    files = sorted(
        Path("results").glob(
            f"native_concurrent_c{args.concurrency}_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError("No matching CSV files")

    # 使用最新一次实验结果
    csv_path = files[-1]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fig, ax = plt.subplots(figsize=(12, 8))

    for row in rows:
        request_id = int(row["request_id"])

        created = float(row["created_offset_ms"]) / 1000
        started = float(row["start_offset_ms"]) / 1000
        ended = float(row["end_offset_ms"]) / 1000

        # 灰色：客户端 Semaphore 等待
        if started > created:
            ax.barh(
                request_id,
                started - created,
                left=created,
                height=0.65,
                color="lightgray",
            )

        # 蓝色：获得 Semaphore 后的请求耗时
        ax.barh(
            request_id,
            ended - started,
            left=started,
            height=0.65,
            color="steelblue",
        )

        ax.plot(ended, request_id, "o", color="darkorange")

    ax.set_xlabel("Time Since Experiment Start (s)")
    ax.set_ylabel("Request ID")
    ax.set_title(
        f"Ollama Request Timeline - C{args.concurrency}"
    )

    ax.set_yticks(range(len(rows)))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Patch(color="lightgray", label="Client Queue"),
            Patch(color="steelblue", label="In-flight Request"),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color="darkorange",
                label="Completion",
            ),
        ]
    )

    fig.tight_layout()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"request_timeline_c{args.concurrency}.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Source: {csv_path}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
