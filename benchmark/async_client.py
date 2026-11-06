
import time
from openai import AsyncOpenAI


class AsyncBenchmarkClient:
    def __init__(self, base_url, model_name):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=120.0,
            max_retries=0
        )

    async def measure_latency(self, prompt):
        start = time.perf_counter()

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )

            return {
                "latency": time.perf_counter() - start,
                "output_tokens": (
                    response.usage.completion_tokens
                    if response.usage else None
                ),
                "success": True,
                "error": ""
            }

        except Exception as e:
            return {
                "latency": time.perf_counter() - start,
                "output_tokens": None,
                "success": False,
                "error": str(e)
            }

    async def close(self):
        await self.client.close()
