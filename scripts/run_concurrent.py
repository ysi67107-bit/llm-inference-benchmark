
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.async_client import AsyncBenchmarkClient
from benchmark.metrics import calculate_metrics


async def run_test(client, prompt, concurrency, total_requests):
    semaphore = asyncio.Semaphore(concurrency)


    async def single_request(request_id):
        # 任务开始等待 Semaphore
        queue_start = time.perf_counter()

        async with semaphore:
            # 成功获取并发许可
            request_start = time.perf_counter()

            # 发起模型请求
            result = await client.measure_latency(prompt)

            # 模型请求结束
            request_end = time.perf_counter()

            result["request_id"] = request_id

            # 客户端排队时间
            result["queue_wait"] = request_start - queue_start

            # 实际 API 请求耗时
            result["request_latency"] = request_end - request_start

            # 从等待开始到响应结束的总耗时
            result["total_latency"] = request_end - queue_start

            return result


    start = time.perf_counter()

    tasks = [
        single_request(i + 1)
        for i in range(total_requests)
    ]

    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start

    return results, duration


def save_csv(results, concurrency):
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = output_dir / (
        f"concurrent_c{concurrency}_{timestamp}.csv"
    )

    fields = [
        "request_id",
        "latency",
        "queue_wait",
        "request_latency",
        "total_latency",
        "output_tokens",
        "success",
        "error"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    return filename


async def main():
    client = AsyncBenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    prompt = "请用简单的语言解释 KV Cache 的作用。"

    try:
        # 预热
        print("Warming up...")
        for _ in range(2):
            result = await client.measure_latency(prompt)
            if not result["success"]:
                raise RuntimeError(result["error"])

        # 不同并发配置
        for concurrency in [1, 2, 4, 8]:
            print(f"\n===== Concurrency {concurrency} =====")

            results, duration = await run_test(
                client=client,
                prompt=prompt,
                concurrency=concurrency,
                total_requests=20
            )

            metrics = calculate_metrics(results)

            successful = [
                r for r in results if r["success"]
            ]

            if successful:
                mean_queue_wait = sum(
                    r["queue_wait"] for r in successful
                ) / len(successful)

                mean_total_latency = sum(
                    r["total_latency"] for r in successful
                ) / len(successful)

                print(f"Mean Queue Wait: {mean_queue_wait:.3f}s")
                print(f"Mean Total Latency: {mean_total_latency:.3f}s")

            throughput = (
                metrics["successful_requests"] / duration
            )

            print(f"Successful: {metrics['successful_requests']}")
            print(f"Failed: {metrics['failed_requests']}")

            if metrics["mean_latency"] is not None:
                print(f"Mean: {metrics['mean_latency']:.3f}s")
                print(f"P50: {metrics['p50_latency']:.3f}s")
                print(f"P95: {metrics['p95_latency']:.3f}s")

            print(f"Throughput: {throughput:.3f} requests/s")
            print(f"Duration: {duration:.3f}s")

            filename = save_csv(results, concurrency)
            print(f"Saved: {filename}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
