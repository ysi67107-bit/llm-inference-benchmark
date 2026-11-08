
import asyncio
import time

from openai import AsyncOpenAI


class BenchmarkRunner:
    def __init__(
        self,
        model: str,
        concurrency: int,
        requests: int,
        max_tokens: int,
        base_url: str = "http://localhost:11434/v1",
    ):
        self.model = model
        self.concurrency = concurrency
        self.requests = requests
        self.max_tokens = max_tokens
        self.base_url = base_url

    
    async def warmup(self, warmup_requests: int = 3):
        print(f"Warmup: {warmup_requests} requests")

        async with AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=120.0,
        ) as client:

            for i in range(warmup_requests):
                await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Explain what a large language "
                                "model inference engine does."
                            ),
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0,
                )

        print("Warmup completed")


    async def run(self):
        semaphore = asyncio.Semaphore(self.concurrency)

        results = []

        async with AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=120.0,
        ) as client:

            async def worker(request_id: int):
                created_at = time.perf_counter()

                async with semaphore:
                    started_at = time.perf_counter()

                    try:
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": (
                                        "Explain what a large language "
                                        "model inference engine does."
                                    ),
                                }
                            ],
                            max_tokens=self.max_tokens,
                            temperature=0,
                        )

                        finished_at = time.perf_counter()

                        output_tokens = (
                            response.usage.completion_tokens
                            if response.usage
                            else None
                        )

                        return {
                            "request_id": request_id,
                            "success": True,
                            "queue_time": started_at - created_at,
                            "latency": finished_at - started_at,
                            "total_latency": finished_at - created_at,
                            "output_tokens": output_tokens,
                            "error": None,
                        }

                    except Exception as exc:
                        finished_at = time.perf_counter()

                        return {
                            "request_id": request_id,
                            "success": False,
                            "queue_time": started_at - created_at,
                            "latency": finished_at - started_at,
                            "total_latency": finished_at - created_at,
                            "output_tokens": None,
                            "error": str(exc),
                        }

            batch_start = time.perf_counter()

            tasks = [
                worker(i)
                for i in range(self.requests)
            ]

            results = await asyncio.gather(*tasks)

            batch_end = time.perf_counter()

        return results, batch_end - batch_start
