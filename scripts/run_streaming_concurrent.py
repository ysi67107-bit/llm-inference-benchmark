
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.streaming_client import StreamingBenchmarkClient


# 每组测试的总请求数
TOTAL_REQUESTS = 20

# 需要测试的客户端并发数
CONCURRENCY_LEVELS = [1, 2, 4, 8]

# 测试 Prompt
PROMPT = "请用简单的语言解释 KV Cache 的作用。"


def percentile(values, p):
    """使用线性插值计算百分位数。"""
    if not values:
        return None

    values = sorted(values)
    position = (len(values) - 1) * p

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


async def run_test(client, concurrency):
    semaphore = asyncio.Semaphore(concurrency)


    async def single_request(request_id):
        queue_start = time.perf_counter()

        async with semaphore:
            acquired_time = time.perf_counter()

            result = await client.measure_streaming(PROMPT)

        finish_time = time.perf_counter()

        result["request_id"] = request_id

        result["queue_start_time"] = queue_start
        result["semaphore_acquired_time"] = acquired_time

        result["queue_wait"] = acquired_time - queue_start

        result["end_to_end_latency"] = (
            finish_time - queue_start
        )

        return result


    start = time.perf_counter()

    tasks = [
        single_request(i + 1)
        for i in range(TOTAL_REQUESTS)
    ]

    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start

    return results, duration


def calculate_streaming_metrics(results, duration):
    successful = [
        r for r in results
        if r["success"]
    ]

    failed = len(results) - len(successful)

    if not successful:
        return {
            "successful": 0,
            "failed": failed,
            "duration": duration
        }

    latencies = [r["latency"] for r in successful]
    ttfts = [
        r["ttft"] for r in successful
        if r["ttft"] is not None
    ]
    tpots = [
        r["tpot"] for r in successful
        if r["tpot"] is not None
    ]

    # 只统计服务端实际返回的输出 Token 数
    total_tokens = sum(
        r["output_tokens"]
        for r in successful
        if r["output_tokens"] is not None
    )

    request_throughput = len(successful) / duration

    # 如果部分成功请求缺少 Token 数，不能计算准确的吞吐量
    all_token_counts_available = all(
        r["output_tokens"] is not None
        for r in successful
    )

    output_throughput = (
        total_tokens / duration
        if all_token_counts_available else None
    )

    return {
        "successful": len(successful),
        "failed": failed,
        "duration": duration,
        "mean_latency": mean(latencies),
        "p95_latency": percentile(latencies, 0.95),
        "mean_ttft": mean(ttfts) if ttfts else None,
        "p50_ttft": percentile(ttfts, 0.50),
        "p95_ttft": percentile(ttfts, 0.95),
        "p99_ttft": percentile(ttfts, 0.99),
        "mean_tpot": mean(tpots) if tpots else None,
        "p95_tpot": percentile(tpots, 0.95),
        "mean_queue_wait": mean(
            r["queue_wait"] for r in successful
        ),
        "mean_time_to_first_output": (
            mean(r["queue_wait"] + r["ttft"] for r in successful
                 if r["ttft"] is not None)
            if ttfts else None
        ),
        "mean_end_to_end_latency": mean(
            r["end_to_end_latency"] for r in successful
        ),
        "total_output_tokens": (
            total_tokens if all_token_counts_available else None
        ),
        "request_throughput": request_throughput,
        "output_throughput": output_throughput
    }


def save_csv(results, concurrency):
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = output_dir / (
        f"streaming_c{concurrency}_{timestamp}.csv"
    )

    fields = [
        "request_id",
        "queue_start_time",
        "semaphore_acquired_time",
        "request_start_time",
        "first_token_time",
        "request_end_time",
        "success",
        "latency",
        "ttft",
        "tpot",
        "output_tokens",
        "chunks",
        "queue_wait",
        "end_to_end_latency",
        "error"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    return filename


def format_metric(value, multiplier=1, suffix=""):
    if value is None:
        return "N/A"

    return f"{value * multiplier:.3f}{suffix}"


async def main():
    client = StreamingBenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    try:
        print("Warming up...")

        for _ in range(2):
            result = await client.measure_streaming(PROMPT)

            if not result["success"]:
                raise RuntimeError(result["error"])

        for concurrency in CONCURRENCY_LEVELS:
            print(
                f"\n===== Streaming Concurrency "
                f"{concurrency} ====="
            )

            results, duration = await run_test(
                client,
                concurrency
            )

            metrics = calculate_streaming_metrics(
                results,
                duration
            )

            print(f"Successful: {metrics['successful']}")
            print(f"Failed: {metrics['failed']}")

            if metrics["successful"] > 0:
                print(
                    "Mean Latency: "
                    + format_metric(
                        metrics["mean_latency"],
                        suffix="s"
                    )
                )

                print(
                    "Mean TTFT: "
                    + format_metric(
                        metrics["mean_ttft"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "P50 TTFT: "
                    + format_metric(
                        metrics["p50_ttft"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "P95 TTFT: "
                    + format_metric(
                        metrics["p95_ttft"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "P99 TTFT: "
                    + format_metric(
                        metrics["p99_ttft"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "Mean TPOT: "
                    + format_metric(
                        metrics["mean_tpot"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "P95 TPOT: "
                    + format_metric(
                        metrics["p95_tpot"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "Mean Queue Wait: "
                    + format_metric(
                        metrics["mean_queue_wait"],
                        suffix="s"
                    )
                )

                print(
                    "Mean Queue + TTFT: "
                    + format_metric(
                        metrics["mean_time_to_first_output"],
                        multiplier=1000,
                        suffix="ms"
                    )
                )

                print(
                    "Mean End-to-End Latency: "
                    + format_metric(
                        metrics["mean_end_to_end_latency"],
                        suffix="s"
                    )
                )

                print(
                    "Request Throughput: "
                    + format_metric(
                        metrics["request_throughput"],
                        suffix=" requests/s"
                    )
                )

                print(
                    "Output Throughput: "
                    + format_metric(
                        metrics["output_throughput"],
                        suffix=" tokens/s"
                    )
                )

                print(
                    "Total Output Tokens: "
                    f"{metrics['total_output_tokens']}"
                )

            print(f"Duration: {duration:.3f}s")

            filename = save_csv(results, concurrency)
            print(f"Saved: {filename}")

            print("\n=== Per-Request TTFT ===")
            for row in results:
                if not row["success"]:
                    print(f"R{row['request_id']:02d}: FAILED - {row.get('error')}")
                    continue
                ttft = row.get("ttft")
                if ttft is None:
                    print(f"R{row['request_id']:02d}: TTFT=N/A")
                    continue
                print(
                    f"R{row['request_id']:02d}: "
                    f"TTFT={ttft * 1000:.2f} ms, "
                    f"Queue={row['queue_wait'] * 1000:.2f} ms, "
                    f"Queue+TTFT={(row['queue_wait'] + ttft) * 1000:.2f} ms"
                )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
