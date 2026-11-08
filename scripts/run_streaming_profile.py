
import asyncio
import statistics

from benchmark.streaming_profiler import OllamaStreamingProfiler


PROMPT = (
    "Explain what KV Cache is and why it improves "
    "large language model inference."
)


async def main():
    profiler = OllamaStreamingProfiler()

    print("Warming up...")
    await profiler.profile(PROMPT, max_tokens=32)

    results = []

    for i in range(5):
        result = await profiler.profile(
            prompt=PROMPT,
            max_tokens=128,
        )

        results.append(result)

        print(
            f"Request {i + 1}: "
            f"TTFT={result['ttft_ms']:.2f} ms, "
            f"TPOT={result['tpot_ms']:.2f} ms, "
            f"Latency={result['client_latency_ms']:.2f} ms, "
            f"Tokens={result['output_tokens']}, "
            f"Chunks={result['output_chunks']}"
        )

    print("\n=== Summary ===")

    for key in [
        "ttft_ms",
        "tpot_ms",
        "client_latency_ms",
    ]:
        values = [
            r[key] for r in results
            if r[key] is not None
        ]

        print(
            f"{key}: {statistics.mean(values):.2f} ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
