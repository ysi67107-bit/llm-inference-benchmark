
import asyncio
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from benchmark.streaming_client import StreamingBenchmarkClient


ROOT = Path(__file__).resolve().parents[1]

PROMPT = "请用简单的语言解释 KV Cache 的作用。"

ARRIVAL_RATES = [0.5, 1.0, 1.5, 2.0]
TOTAL_REQUESTS = 40


def percentile(values, p):
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


async def run_experiment(client, arrival_rate):
    """
    按固定时间间隔提交请求。

    与之前的并发测试不同：
    这里不使用 Semaphore 限制同时在途请求数。
    """

    experiment_start = time.perf_counter()

    async def worker(request_id, scheduled_offset):
        # 计划发起时间
        target_time = experiment_start + scheduled_offset

        # 等待到计划发起时间
        delay = target_time - time.perf_counter()

        if delay > 0:
            await asyncio.sleep(delay)

        actual_start = time.perf_counter()

        try:
            result = await client.measure_streaming(PROMPT)
        except Exception as exc:
            result = {
                "success": False,
                "error": str(exc),
            }

        finish_time = time.perf_counter()

        # TTFT 是从请求真正开始执行后计时
        ttft = result.get("ttft")

        return {
            "request_id": request_id,
            "arrival_rate": arrival_rate,
            "success": result.get("success", False),
            "scheduled_offset_s": scheduled_offset,
            "actual_start_offset_s": (
                actual_start - experiment_start
            ),
            "end_offset_s": (
                finish_time - experiment_start
            ),
            "schedule_delay_ms": max(
                0.0,
                (actual_start - target_time) * 1000
            ),
            "latency_s": result.get("latency"),
            "ttft_ms": (
                ttft * 1000
                if ttft is not None
                else None
            ),
            "tpot_ms": (
                result["tpot"] * 1000
                if result.get("tpot") is not None
                else None
            ),
            "output_tokens": result.get("output_tokens"),
            "error": result.get("error", ""),
        }

    # 每个请求的计划到达间隔 = 1 / arrival_rate
    interval = 1.0 / arrival_rate

    tasks = [
        asyncio.create_task(
            worker(
                request_id=i + 1,
                scheduled_offset=i * interval,
            )
        )
        for i in range(TOTAL_REQUESTS)
    ]

    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - experiment_start

    return results, duration


def summarize(results, duration, arrival_rate):
    successful = [
        r for r in results
        if r["success"]
    ]

    ttfts = [
        r["ttft_ms"]
        for r in successful
        if r["ttft_ms"] is not None
    ]

    tpots = [
        r["tpot_ms"]
        for r in successful
        if r["tpot_ms"] is not None
    ]

    latencies = [
        r["latency_s"]
        for r in successful
        if r["latency_s"] is not None
    ]

    tokens_available = all(
        r["output_tokens"] is not None
        for r in successful
    )

    total_tokens = sum(
        r["output_tokens"] or 0
        for r in successful
    )

    return {
        "arrival_rate": arrival_rate,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": duration,
        "mean_ttft_ms": (
            statistics.mean(ttfts) if ttfts else None
        ),
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "mean_tpot_ms": (
            statistics.mean(tpots) if tpots else None
        ),
        "mean_latency_s": (
            statistics.mean(latencies)
            if latencies else None
        ),
        "request_throughput": (
            len(successful) / duration
        ),
        "output_throughput": (
            total_tokens / duration
            if tokens_available else None
        ),
    }


def save_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def display_metric(label, value, unit=""):
    if value is None:
        print(f"{label}: N/A")
    else:
        print(f"{label}: {value:.3f} {unit}")


async def main():
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    client = StreamingBenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b",
    )

    summaries = []

    try:
        print("Warming up...")

        for _ in range(2):
            result = await client.measure_streaming(PROMPT)

            if not result["success"]:
                raise RuntimeError(result["error"])

        for rate in ARRIVAL_RATES:
            print(f"\n===== Arrival Rate {rate} req/s =====")

            results, duration = await run_experiment(
                client,
                arrival_rate=rate,
            )

            summary = summarize(
                results,
                duration,
                rate,
            )

            summaries.append(summary)

            print(
                f"Success: {summary['successful']}/"
                f"{TOTAL_REQUESTS}"
            )

            display_metric(
                "Mean TTFT",
                summary["mean_ttft_ms"],
                "ms",
            )
            display_metric(
                "P95 TTFT",
                summary["p95_ttft_ms"],
                "ms",
            )
            display_metric(
                "Mean TPOT",
                summary["mean_tpot_ms"],
                "ms",
            )
            display_metric(
                "Request Throughput",
                summary["request_throughput"],
                "req/s",
            )
            display_metric(
                "Duration",
                summary["duration_s"],
                "s",
            )

            filename = output_dir / (
                f"arrival_rate_{rate}_{timestamp}.csv"
            )

            save_csv(filename, results)
            print(f"Saved: {filename}")

        summary_path = output_dir / (
            f"arrival_rate_summary_{timestamp}.csv"
        )

        save_csv(summary_path, summaries)

        print(f"\nSummary saved: {summary_path}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
