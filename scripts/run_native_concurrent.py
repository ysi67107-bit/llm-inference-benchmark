
import argparse
import asyncio
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from benchmark.native_profiler import OllamaNativeProfiler


PROMPT = (
    "Explain the difference between prefill "
    "and decode in large language model inference."
)


async def run_experiment(concurrency, requests, max_tokens):
    profiler = OllamaNativeProfiler(model="qwen2.5:3b")

    # 限制客户端同时执行的请求数量
    semaphore = asyncio.Semaphore(concurrency)

    # 【新增 1】整个实验的统一起始时间
    # 所有请求的时间偏移都以这个时刻为基准
    experiment_start = time.perf_counter()

    async def worker(request_id):
        # 【新增 2】记录任务开始执行、准备等待信号量的时间
        created_at = time.perf_counter()

        async with semaphore:
            # 【新增 3】记录获得信号量的时间
            started_at = time.perf_counter()

            try:
                result = await profiler.profile(
                    prompt=PROMPT,
                    max_tokens=max_tokens,
                )

                # 【新增 4】记录请求完成时间
                finished_at = time.perf_counter()

                return {
                    "request_id": request_id,
                    "concurrency": concurrency,
                    "success": True,

                    # 相对于实验开始的时间偏移
                    "created_offset_ms": (
                        created_at - experiment_start
                    ) * 1000,

                    "start_offset_ms": (
                        started_at - experiment_start
                    ) * 1000,

                    "end_offset_ms": (
                        finished_at - experiment_start
                    ) * 1000,

                    # 客户端 Semaphore 等待时间
                    "queue_ms": (
                        started_at - created_at
                    ) * 1000,

                    # Ollama 原生 Profiling 结果
                    **result,

                    "error": "",
                }

            except Exception as exc:
                finished_at = time.perf_counter()

                return {
                    "request_id": request_id,
                    "concurrency": concurrency,
                    "success": False,

                    "created_offset_ms": (
                        created_at - experiment_start
                    ) * 1000,

                    "start_offset_ms": (
                        started_at - experiment_start
                    ) * 1000,

                    "end_offset_ms": (
                        finished_at - experiment_start
                    ) * 1000,

                    "queue_ms": (
                        started_at - created_at
                    ) * 1000,

                    "error": str(exc),
                }

    # 同时创建所有任务
    tasks = [
        worker(i)
        for i in range(requests)
    ]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)

    # 【修改】使用统一的实验起点计算总耗时
    duration = time.perf_counter() - experiment_start

    return results, duration


def mean_metric(results, key):
    values = [
        r[key]
        for r in results
        if r["success"] and r.get(key) is not None
    ]

    return statistics.mean(values) if values else None


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if min(
        args.concurrency,
        args.requests,
        args.max_tokens,
    ) < 1:
        parser.error("All numeric parameters must be >= 1")

    profiler = OllamaNativeProfiler(model="qwen2.5:3b")

    # 预热不计入正式实验时间
    print("Warming up...")
    await profiler.profile(
        prompt=PROMPT,
        max_tokens=32,
    )

    print(
        f"Running concurrency={args.concurrency}, "
        f"requests={args.requests}"
    )

    results, duration = await run_experiment(
        concurrency=args.concurrency,
        requests=args.requests,
        max_tokens=args.max_tokens,
    )

    successful = [
        r for r in results
        if r["success"]
    ]

    throughput = len(successful) / duration

    print("\n=== Native Concurrent Profile ===")
    print(f"Success: {len(successful)}/{args.requests}")
    print(f"Duration: {duration:.3f} s")
    print(f"Throughput: {throughput:.3f} req/s")

    for key in [
        "queue_ms",
        "client_latency_ms",
        "server_total_ms",
        "prefill_ms",
        "decode_ms",
        "decode_tokens_per_s",
    ]:
        value = mean_metric(results, key)

        if value is not None:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: N/A")

    # 输出文件
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = output_dir / (
        f"native_concurrent_c{args.concurrency}_"
        f"{timestamp}.csv"
    )

    # 自动收集所有 CSV 字段
    fieldnames = sorted({
        key
        for result in results
        for key in result.keys()
    })

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output_path}")

    # 输出每条请求的时间线
    print("\n=== Request Timeline ===")
    print(
        f"{'ID':>3} "
        f"{'Start(ms)':>12} "
        f"{'End(ms)':>12} "
        f"{'Queue(ms)':>12}"
    )

    for r in results:
        print(
            f"{r['request_id']:>3} "
            f"{r['start_offset_ms']:>12.1f} "
            f"{r['end_offset_ms']:>12.1f} "
            f"{r['queue_ms']:>12.1f}"
        )

    for result in results:
        if not result["success"]:
            print(
                f"Request {result['request_id']} failed: "
                f"{result['error']}"
            )


if __name__ == "__main__":
    asyncio.run(main())

