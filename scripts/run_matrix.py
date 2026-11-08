
import asyncio
import csv
from datetime import datetime
from pathlib import Path

from benchmark.runner import BenchmarkRunner


CONCURRENCY_LEVELS = [1, 2, 4, 8]
MODEL = "qwen2.5:3b"
REQUESTS = 20
MAX_TOKENS = 32


async def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    summary = []

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\nRunning concurrency={concurrency}")

        runner = BenchmarkRunner(
            model=MODEL,
            concurrency=concurrency,
            requests=REQUESTS,
            max_tokens=MAX_TOKENS,
        )

        results, duration = await runner.run()

        successful = [r for r in results if r["success"]]

        latencies = [r["latency"] for r in successful]

        total_tokens = sum(
            r["output_tokens"] or 0
            for r in successful
        )

        summary.append({
            "concurrency": concurrency,
            "requests": REQUESTS,
            "successful": len(successful),
            "duration": duration,
            "mean_latency": (
                sum(latencies) / len(latencies)
                if latencies else None
            ),
            "request_throughput": (
                len(successful) / duration
                if duration > 0 else None
            ),
            "output_throughput": (
                total_tokens / duration
                if duration > 0 else None
            ),
        })

        print(
            f"Completed: {len(successful)}/{REQUESTS}, "
            f"duration={duration:.2f}s"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"matrix_{timestamp}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nMatrix results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
