
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    files = sorted(
        Path("results").glob(
            f"native_concurrent_c{args.concurrency}_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError("No matching CSV file")

    csv_path = files[-1]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []

    for row in rows:
        if row["success"] != "True":
            continue

        def metric(name):
            value = row.get(name)
            return float(value) if value else 0.0

        server_total = metric("server_total_ms")
        prefill = metric("prefill_ms")
        decode = metric("decode_ms")
        load = metric("load_ms")

        records.append({
            "request_id": int(row["request_id"]),
            "client_ms": metric("client_latency_ms"),
            "server_ms": server_total,
            "prefill_ms": prefill,
            "decode_ms": decode,
            "load_ms": load,
            "residual_ms": (
                server_total - prefill - decode - load
            ),
        })

    records.sort(key=lambda r: r["request_id"])

    print(f"\nSource: {csv_path}")
    print(
        f"{'ID':>3} {'Client':>10} {'Server':>10} "
        f"{'Prefill':>10} {'Decode':>10} "
        f"{'Load':>10} {'Residual':>10}"
    )

    for r in records:
        print(
            f"{r['request_id']:>3} "
            f"{r['client_ms']:>10.1f} "
            f"{r['server_ms']:>10.1f} "
            f"{r['prefill_ms']:>10.1f} "
            f"{r['decode_ms']:>10.1f} "
            f"{r['load_ms']:>10.1f} "
            f"{r['residual_ms']:>10.1f}"
        )

    ids = [r["request_id"] for r in records]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        ids,
        [r["prefill_ms"] for r in records],
        label="Prefill",
    )

    ax.bar(
        ids,
        [r["decode_ms"] for r in records],
        bottom=[r["prefill_ms"] for r in records],
        label="Decode",
    )

    ax.plot(
        ids,
        [r["server_ms"] for r in records],
        marker="o",
        label="Server Total",
    )

    ax.set_xlabel("Request ID")
    ax.set_ylabel("Duration (ms)")
    ax.set_title(
        f"Native Profiling - Concurrency {args.concurrency}"
    )
    ax.set_xticks(ids)
    ax.legend()
    fig.tight_layout()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"native_breakdown_c{args.concurrency}.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
