
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.streaming_client import StreamingBenchmarkClient


async def main():
    client = StreamingBenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    prompt = "请用简单的语言解释 KV Cache 的作用。"

    try:
        print("Warming up...")
        warmup = await client.measure_streaming(prompt)

        if not warmup["success"]:
            raise RuntimeError(warmup["error"])

        print("\n===== Streaming Benchmark =====")

        for i in range(5):
            result = await client.measure_streaming(prompt)

            print(f"\nRequest {i + 1}")
            print(f"Success: {result['success']}")

            if result["success"]:
                print(f"Latency: {result['latency']:.3f}s")
                print(f"TTFT: {result['ttft']:.3f}s")

                if result["tpot"] is not None:
                    print(f"TPOT: {result['tpot'] * 1000:.2f}ms")
                else:
                    print("TPOT: N/A")

                print(f"Output Tokens: {result['output_tokens']}")
                print(f"Chunks: {result['chunks']}")
            else:
                print(f"Error: {result['error']}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
