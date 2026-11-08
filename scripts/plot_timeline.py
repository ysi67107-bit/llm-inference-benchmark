
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]


def load_results(filename):
    results = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_fields = [
            "queue_start_time",
            "semaphore_acquired_time",
            "request_start_time",
            "first_token_time",
            "request_end_time",
        ]

        missing = [
            field for field in required_fields
            if field not in (reader.fieldnames or [])
        ]

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for row in reader:
            if row["success"] != "True":
                continue

            if any(
                not row.get(field)
                for field in required_fields
            ):
                continue

            for field in required_fields:
                row[field] = float(row[field])

            row["request_id"] = int(row["request_id"])

            results.append(row)

    return sorted(
        results,
        key=lambda r: r["request_id"]
    )


def plot_timeline(results, title, output_file):
    if not results:
        raise ValueError("No valid requests found.")

    origin = min(
        r["queue_start_time"]
        for r in results
    )

    fig, ax = plt.subplots(figsize=(13, 9))

    for row in results:
        request_id = row["request_id"]

        queue_start = (
            row["queue_start_time"] - origin
        )

        acquired = (
            row["semaphore_acquired_time"] - origin
        )

        first_token = (
            row["first_token_time"] - origin
        )

        request_end = (
            row["request_end_time"] - origin
        )

        queue_duration = acquired - queue_start

        ttft_duration = first_token - acquired

        streaming_duration = request_end - first_token

        # 阶段一：客户端 Semaphore 排队
        ax.barh(
            request_id,
            queue_duration,
            left=queue_start,
            height=0.65,
            color="#64748B"
        )

        # 阶段二：等待首块文本
        ax.barh(
            request_id,
            ttft_duration,
            left=acquired,
            height=0.65,
            color="#E69F00"
        )

        # 阶段三：流式输出
        ax.barh(
            request_id,
            streaming_duration,
            left=first_token,
            height=0.65,
            color="#009E73"
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Time since first task (seconds)")
    ax.set_ylabel("Request ID")

    ax.set_yticks([
        r["request_id"] for r in results
    ])

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Patch(
                facecolor="#64748B",
                label="Client Queue"
            ),
            Patch(
                facecolor="#E69F00",
                label="TTFT / Before First Content"
            ),
            Patch(
                facecolor="#009E73",
                label="Streaming Output"
            )
        ],
        loc="lower right"
    )

    plt.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to streaming benchmark CSV"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG file path"
    )

    args = parser.parse_args()

    input_file = Path(args.input)

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    results = load_results(input_file)

    output_file = (
        Path(args.output)
        if args.output
        else input_file.with_name(
            input_file.stem + "_lifecycle.png"
        )
    )

    plot_timeline(
        results,
        title=f"Request Lifecycle: {input_file.stem}",
        output_file=output_file
    )


if __name__ == "__main__":
    main()
