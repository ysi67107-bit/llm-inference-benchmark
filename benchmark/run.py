import csv
import json
from pathlib import Path
from datetime import datetime
import asyncio
import statistics
from benchmark.runner import BenchmarkRunner
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark CLI"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama"],
        help="Inference backend"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b",
        help="Model name"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests"
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Total number of requests"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum output tokens per request"
    )

    return parser.parse_args()


def percentile(values, p):
    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * p / 100

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def save_results(args, results, duration):
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    json_path = output_dir / f"benchmark_{timestamp}_config.json"

    # Save per-request results
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "success",
                "queue_time",
                "latency",
                "total_latency",
                "output_tokens",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    successful = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in successful]

    total_tokens = sum(
        r["output_tokens"] or 0 for r in successful
    )

    config = {
        "timestamp": timestamp,
        "backend": args.backend,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "duration_seconds": duration,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "mean_latency_seconds": (
            statistics.mean(latencies) if latencies else None
        ),
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "p99_latency_seconds": percentile(latencies, 99),
        "request_throughput": (
            len(successful) / duration if duration > 0 else None
        ),
        "output_throughput": (
            total_tokens / duration if duration > 0 else None
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nCSV saved to:    {csv_path}")
    print(f"Config saved to: {json_path}")


def main():
    args = parse_args()

    if args.concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    if args.requests < 1:
        raise ValueError("requests must be >= 1")

    if args.max_tokens < 1:
        raise ValueError("max-tokens must be >= 1")

    if args.backend != "ollama":
        raise ValueError("Only Ollama is supported currently")

    runner = BenchmarkRunner(
        model=args.model,
        concurrency=args.concurrency,
        requests=args.requests,
        max_tokens=args.max_tokens,
    )

    results, duration = asyncio.run(runner.run())

    successful = [
        result for result in results
        if result["success"]
    ]

    failed = len(results) - len(successful)

    latencies = [
        result["latency"]
        for result in successful
    ]

    total_tokens = sum(
        result["output_tokens"] or 0
        for result in successful
    )

    print("\n=== Benchmark Results ===")
    print(f"Total requests: {len(results)}")
    print(f"Successful:     {len(successful)}")
    print(f"Failed:         {failed}")
    print(f"Duration:       {duration:.3f} s")

    if successful:
        mean_latency = sum(latencies) / len(latencies)

        request_throughput = len(successful) / duration
        token_throughput = total_tokens / duration

        print(f"Mean latency:   {mean_latency:.3f} s")
        print(f"P50 latency:    {percentile(latencies, 50):.3f} s")
        print(f"P95 latency:    {percentile(latencies, 95):.3f} s")
        print(f"P99 latency:    {percentile(latencies, 99):.3f} s")
        print(f"Max latency:    {max(latencies):.3f} s")
        print(f"Request TPS:    {request_throughput:.3f} req/s")
        print(f"Output TPS:     {token_throughput:.3f} tok/s")

    for result in results:
        if not result["success"]:
            print(
                f"Request {result['request_id']} failed: "
                f"{result['error']}"
            )
    
    save_results(args, results, duration)


if __name__ == "__main__":
    main()
