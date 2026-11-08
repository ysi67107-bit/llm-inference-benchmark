
import asyncio
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.ollama_native_client import OllamaNativeClient


OUTPUT_LENGTHS = [32, 64, 128, 256]
REPEAT = 5

PROMPT = (
    "Generate a very long sequence of integers "
    "starting from 1. Separate each integer with "
    "a comma and a space. Continue counting "
    "without explanations, headings, or stopping. "
    "Output only the sequence."
)


async def main():
    client = OllamaNativeClient()
    results = []

    try:
        print("Warming up...")

        for _ in range(2):
            result = await client.measure(
                PROMPT,
                max_tokens=32
            )

            if not result["success"]:
                raise RuntimeError(result["error"])

        cases = [
            (length, repeat_id)
            for repeat_id in range(REPEAT)
            for length in OUTPUT_LENGTHS
        ]

        rng = random.Random(2026)
        rng.shuffle(cases)

        for index, (length, repeat_id) in enumerate(
            cases, start=1
        ):
            result = await client.measure(
                PROMPT,
                max_tokens=length
            )

            row = {
                "target_output_tokens": length,
                "repeat_id": repeat_id + 1,
                **result
            }

            if result["success"]:
                output_tokens = result["eval_count"]
                decode_seconds = result["eval_duration"]

                decode_throughput = (
                    output_tokens / decode_seconds
                    if output_tokens is not None
                    and decode_seconds is not None
                    and decode_seconds > 0
                    else None
                )

                row["decode_throughput"] = decode_throughput

                print(
                    f"[{index}/{len(cases)}] "
                    f"Target={length}, "
                    f"Actual={output_tokens}, "
                    f"Decode={decode_seconds * 1000:.2f}ms, "
                    f"Speed={decode_throughput:.2f} tok/s, "
                    f"Reason={result['done_reason']}"
                )
            else:
                row["decode_throughput"] = None
                print(
                    f"[{index}/{len(cases)}] "
                    f"ERROR={result['error']}"
                )

            results.append(row)

    finally:
        await client.close()

    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / (
        "native_decode_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    fields = [
        "target_output_tokens",
        "repeat_id",
        "success",
        "client_latency",
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "prompt_eval_count",
        "prompt_eval_cached_count",
        "eval_count",
        "done_reason",
        "decode_throughput",
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

    print("\n===== Native Decode Summary =====")

    print(
        f"{'Target':<10}"
        f"{'Actual':<12}"
        f"{'Decode(ms)':<15}"
        f"{'Speed(tok/s)':<17}"
        f"{'Success':<10}"
    )

    for length in OUTPUT_LENGTHS:
        group = [
            r for r in results
            if r["target_output_tokens"] == length
            and r["success"]
        ]

        if not group:
            print(f"{length:<10}No successful requests")
            continue

        print(
            f"{length:<10}"
            f"{mean(r['eval_count'] for r in group):<12.1f}"
            f"{mean(r['eval_duration'] for r in group) * 1000:<15.2f}"
            f"{mean(r['decode_throughput'] for r in group):<17.2f}"
            f"{len(group):<10}"
        )


if __name__ == "__main__":
    asyncio.run(main())
