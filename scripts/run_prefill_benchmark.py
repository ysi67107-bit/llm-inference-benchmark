
import asyncio
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.streaming_client import StreamingBenchmarkClient


# 实验配置
TARGET_LENGTHS = [128, 512, 1024, 2048]
REPEAT = 5
MAX_OUTPUT_TOKENS = 16

# 使用英文单词构造不同输入
WORDS = [
    "apple", "river", "computer", "science",
    "biology", "network", "system", "memory",
    "model", "learning", "database", "energy",
    "research", "protein", "signal", "language",
    "algorithm", "sequence", "hardware", "software"
]


def generate_prompt(target_length, seed):
    """
    生成长度近似 target_length 的输入。

    注意：这里的 target_length 是目标长度，
    并非模型分词器确认的精确 Token 数。
    """
    rng = random.Random(seed)

    words = [
        rng.choice(WORDS)
        for _ in range(target_length)
    ]

    content = " ".join(words)

    return (
        "Read the following text and reply "
        "with only the word OK.\n\n"
        + content
    )


async def run_benchmark():
    client = StreamingBenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    results = []

    try:
        print("Warming up...")

        for i in range(2):
            prompt = generate_prompt(128, seed=10000 + i)

            result = await client.measure_streaming(
                prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )

            if not result["success"]:
                raise RuntimeError(result["error"])

        # 将不同长度交错测试，减少时间顺序影响
        cases = [
            (length, repeat_id)
            for repeat_id in range(REPEAT)
            for length in TARGET_LENGTHS
        ]

        rng = random.Random(2026)
        rng.shuffle(cases)

        for index, (length, repeat_id) in enumerate(
            cases, start=1
        ):
            seed = length * 1000 + repeat_id

            prompt = generate_prompt(length, seed)

            result = await client.measure_streaming(
                prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )

            row = {
                "target_length": length,
                "repeat_id": repeat_id + 1,
                "success": result["success"],
                "ttft": result["ttft"],
                "latency": result["latency"],
                "tpot": result["tpot"],
                "output_tokens": result["output_tokens"],
                "error": result["error"]
            }

            results.append(row)

            if result["success"]:
                print(
                    f"[{index}/{len(cases)}] "
                    f"Target={length}, "
                    f"TTFT={result['ttft'] * 1000:.2f}ms, "
                    f"Latency={result['latency']:.3f}s, "
                    f"Output={result['output_tokens']}"
                )
            else:
                print(
                    f"[{index}/{len(cases)}] "
                    f"Target={length}, "
                    f"ERROR={result['error']}"
                )

    finally:
        await client.close()

    return results


def save_results(results):
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = output_dir / (
        f"prefill_benchmark_{timestamp}.csv"
    )

    fields = [
        "target_length",
        "repeat_id",
        "success",
        "ttft",
        "latency",
        "tpot",
        "output_tokens",
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

    print(f"\nSaved: {filename}")


def summarize(results):
    print("\n===== Prefill Benchmark Summary =====")

    print(
        f"{'Target':<10}"
        f"{'Success':<10}"
        f"{'Mean TTFT':<15}"
        f"{'Median TTFT':<15}"
        f"{'Mean Latency':<15}"
    )

    for length in TARGET_LENGTHS:
        group = [
            r for r in results
            if r["target_length"] == length
            and r["success"]
        ]

        if not group:
            print(f"{length:<10}No successful requests")
            continue

        ttfts = [r["ttft"] for r in group]
        latencies = [r["latency"] for r in group]

        print(
            f"{length:<10}"
            f"{len(group):<10}"
            f"{mean(ttfts) * 1000:<15.2f}"
            f"{median(ttfts) * 1000:<15.2f}"
            f"{mean(latencies):<15.3f}"
        )


async def main():
    results = await run_benchmark()

    save_results(results)
    summarize(results)


if __name__ == "__main__":
    asyncio.run(main())
