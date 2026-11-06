
import sys
import csv
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.client import BenchmarkClient
from benchmark.metrics import calculate_metrics


def save_results(results):
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"benchmark_{timestamp}.csv"

    fieldnames = [
        "request_id",
        "latency",
        "output_tokens",
        "success",
        "error"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return filename


def main():
    client = BenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    prompt = "请用简单的语言解释 KV Cache 的作用。"

    warmup_requests = 2
    test_requests = 10

    print("===== Warmup =====")

    for i in range(warmup_requests):
        result = client.measure_latency(prompt)
        print(
            f"Warmup {i + 1}: "
            f"{result['latency']:.3f}s, "
            f"success={result['success']}"
        )

        if not result["success"]:
            print(f"Error: {result['error']}")
            print("Warmup failed. Benchmark stopped.")
            return

    print("\n===== Benchmark =====")

    results = []

    benchmark_start = time.perf_counter()

    for i in range(test_requests):
        result = client.measure_latency(prompt)

        result["request_id"] = i + 1
        results.append(result)

        print(
            f"Request {i + 1}: "
            f"{result['latency']:.3f}s, "
            f"tokens={result['output_tokens']}, "
            f"success={result['success']}"
        )

    benchmark_duration = (
        time.perf_counter() - benchmark_start
    )

    metrics = calculate_metrics(results)

    print("\n===== Summary =====")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Successful: {metrics['successful_requests']}")
    print(f"Failed: {metrics['failed_requests']}")

    if metrics["mean_latency"] is not None:
        print(f"Mean Latency: {metrics['mean_latency']:.3f}s")
        print(f"P50 Latency: {metrics['p50_latency']:.3f}s")
        print(f"P95 Latency: {metrics['p95_latency']:.3f}s")

    print(f"Total Benchmark Time: {benchmark_duration:.3f}s")

    filename = save_results(results)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()

