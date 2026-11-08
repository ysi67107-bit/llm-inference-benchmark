
import time
from openai import AsyncOpenAI


class StreamingBenchmarkClient:
    def __init__(self, base_url, model_name):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=120.0,
            max_retries=0
        )
        self.model_name = model_name

    async def measure_streaming(self, prompt, max_tokens=100):
        last_token_time = None
        start = time.perf_counter()
        first_token_time = None
        chunks = 0
        output_parts = []
        output_tokens = None

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in stream:
                now = time.perf_counter()

                if chunk.usage is not None:
                    output_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    if first_token_time is None:
                        first_token_time = now

                    last_token_time = now

                    chunks += 1
                    output_parts.append(content)

            end = time.perf_counter()

            total_latency = end - start

            if first_token_time is None:
                raise RuntimeError("No content received")

            ttft = first_token_time - start

            # 只有获取到可靠的输出 Token 数时才计算 TPOT

            if output_tokens is not None and output_tokens > 1:
                tpot = (
                    last_token_time - first_token_time
                ) / (output_tokens - 1)
            else:
                tpot = None

            return {
                "request_start_time": start,
                "first_token_time": first_token_time,
                "request_end_time": end,
                "success": True,
                "latency": total_latency,
                "ttft": ttft,
                "tpot": tpot,
                "output_tokens": output_tokens,
                "chunks": chunks,
                "error": ""
            }

        except Exception as e:
            return {
                "success": False,
                "latency": time.perf_counter() - start,
                "ttft": None,
                "tpot": None,
                "output_tokens": None,
                "chunks": chunks,
                "error": str(e)
            }

    async def close(self):
        await self.client.close()
