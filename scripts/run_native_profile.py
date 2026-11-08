
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

from benchmark.native_profiler import OllamaNativeProfiler


async def main():
    profiler = OllamaNativeProfiler(
        model="qwen2.5:3b"
    )

    prompt = (
        "Explain the difference between prefill "
        "and decode in large language model inference."
    )

    # Warmup
    print("Warming up...")
    await profiler.profile(prompt, max_tokens=32)

    # Formal measurements
    results = []

    for i in range(5):
        result = await profiler.profile(
            prompt,
            max_tokens=128,
        )

        results.append(result)

        print(
            f"Request {i + 1}: "
            f"client={result['client_latency_ms']:.2f} ms, "
            f"prefill={result['prefill_ms']} ms, "
            f"decode={result['decode_ms']} ms, "
            f"tokens={result['output_tokens']}"
        )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"native_profile_{timestamp}.csv"
    json_path = output_dir / f"native_profile_{timestamp}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": profiler.model,
                "prompt": prompt,
                "max_tokens": 128,
                "warmup_requests": 1,
                "measurements": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nCSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
