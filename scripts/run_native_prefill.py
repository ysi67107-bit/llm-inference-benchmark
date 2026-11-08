
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
from scripts.run_prefill_benchmark import generate_prompt


TARGET_LENGTHS = [128, 512, 1024, 2048]
REPEAT = 5
MAX_OUTPUT_TOKENS = 16


async def main():
    client = OllamaNativeClient()
    results = []

    try:
        print("Warming up...")

        for i in range(2):
            prompt = generate_prompt(
                128,
                seed=20000 + i
            )

            result = await client.measure(
                prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )

            if not result["success"]:
                raise RuntimeError(result["error"])

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
            prompt = generate_prompt(
                length,
                seed=length * 1000 + repeat_id
            )

            result = await client.measure(
                prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )

            row = {
                "target_length": length,
                "repeat_id": repeat_id + 1,
                **result
            }

            results.append(row)

            if result["success"]:
                print(
                    f"[{index}/{len(cases)}] "
                    f"Target={length}, "
                    f"PromptTokens="
                    f"{result['prompt_eval_count']}, "
                    f"Cached="
                    f"{result['prompt_eval_cached_count']}, "
                    f"Prefill="
                    f"{result['prompt_eval_duration'] * 1000:.2f}ms, "
                    f"Decode="
                    f"{result['eval_duration'] * 1000:.2f}ms"
                )
            else:
                print(
                    f"[{index}/{len(cases)}] "
                    f"ERROR={result['error']}"
                )

    finally:
        await client.close()

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / (
        "native_prefill_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    fields = [
        "target_length",
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

    print("\n===== Native Prefill Summary =====")
    print(
        f"{'Target':<10}"
        f"{'Tokens':<12}"
        f"{'Prefill(ms)':<15}"
        f"{'Decode(ms)':<15}"
    )

    for length in TARGET_LENGTHS:
        group = [
            r for r in results
            if r["target_length"] == length
            and r["success"]
        ]

        if not group:
            continue

        print(
            f"{length:<10}"
            f"{mean(r['prompt_eval_count'] for r in group):<12.1f}"
            f"{mean(r['prompt_eval_duration'] for r in group) * 1000:<15.2f}"
            f"{mean(r['eval_duration'] for r in group) * 1000:<15.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
