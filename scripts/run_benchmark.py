
import sys
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.client import BenchmarkClient


def main():
    client = BenchmarkClient(
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:3b"
    )

    prompt = "请用简单的语言解释 KV Cache 的作用。"

    result = client.measure_latency(prompt)

    print("===== Benchmark Result =====")
    print(f"Prompt: {result['prompt']}")
    print(f"Latency: {result['latency']:.3f} seconds")
    print(f"Output Tokens: {result['output_tokens']}")
    print(f"Response: {result['response']}")


if __name__ == "__main__":
    main()
