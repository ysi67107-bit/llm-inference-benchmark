import argparse
import asyncio
import csv
import statistics
from datetime import datetime
from pathlib import Path

from benchmark.runner import BenchmarkRunner


CONCURRENCY_LEVELS = [1, 2, 4, 8]
REPETITIONS = 3
REQUESTS = 20
MAX_TOKENS = 32
MODEL = "qwen2.5:3b"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server-parallel",
        type=int,
        required=True,
        choices=[1, 2, 4],
    )

    return parser.parse_args()

async def main():
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    round_results = []
    summary_results = []

    # 先预热一次
    warmup_runner = BenchmarkRunner(
        model=MODEL,
        concurrency=1,
        requests=REQUESTS,
        max_tokens=MAX_TOKENS,
    )
    await warmup_runner.warmup(warmup_requests=3)

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n=== Concurrency {concurrency} ===")

        throughputs = []
        latencies = []

        for repetition in range(1, REPETITIONS + 1):
            runner = BenchmarkRunner(
                model=MODEL,
                concurrency=concurrency,
                requests=REQUESTS,
                max_tokens=MAX_TOKENS,
            )

            results, duration = await runner.run()

            successful = [
                r for r in results if r["success"]
            ]

            request_throughput = (
                len(successful) / duration
                if duration > 0 else 0
            )

            request_latencies = [
                r["latency"] for r in successful
            ]

            mean_latency = (
                statistics.mean(request_latencies)
                if request_latencies else None
            )

            round_results.append({
                "server_parallel": args.server_parallel,
                "concurrency": concurrency,
                "repetition": repetition,
                "successful": len(successful),
                "duration": duration,
                "request_throughput": request_throughput,
                "mean_latency": mean_latency,
            })

            if successful:
                throughputs.append(request_throughput)
                latencies.append(mean_latency)

            print(
                f"Round {repetition}: "
                f"success={len(successful)}/{REQUESTS}, "
                f"throughput={request_throughput:.3f} req/s"
            )

        summary_results.append({
            "server_parallel": args.server_parallel,
            "concurrency": concurrency,
            "mean_throughput": (
                statistics.mean(throughputs)
                if throughputs else None
            ),
            "std_throughput": (
                statistics.stdev(throughputs)
                if len(throughputs) > 1 else None
            ),
            "mean_latency": (
                statistics.mean(latencies)
                if latencies else None
            ),
            "std_latency": (
                statistics.stdev(latencies)
                if len(latencies) > 1 else None
            ),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    round_path = output_dir / (
    f"matrix_p{args.server_parallel}_rounds_{timestamp}.csv"
    )

    summary_path = output_dir / (
        f"matrix_p{args.server_parallel}_summary_{timestamp}.csv"
    )

    with open(round_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=round_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(round_results)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_results)

    print("\n=== Experiment Completed ===")
    print(f"Rounds saved:  {round_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
