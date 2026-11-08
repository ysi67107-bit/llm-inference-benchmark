
import time

import httpx


class OllamaNativeClient:
    def __init__(
        self,
        base_url="http://localhost:11434",
        model_name="qwen2.5:3b"
    ):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=120.0
        )
        self.model_name = model_name

    async def measure(
        self,
        prompt,
        max_tokens=16
    ):
        start = time.perf_counter()

        try:
            response = await self.client.post(
                "/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0
                    }
                }
            )

            response.raise_for_status()
            data = response.json()

            client_latency = (
                time.perf_counter() - start
            )

            # Ollama 的 duration 单位是纳秒
            NS_TO_SECONDS = 1_000_000_000

            def seconds(field):
                value = data.get(field)
                return (
                    value / NS_TO_SECONDS
                    if value is not None
                    else None
                )

            return {
                "success": True,
                "client_latency": client_latency,
                "total_duration": seconds(
                    "total_duration"
                ),
                "load_duration": seconds(
                    "load_duration"
                ),
                "prompt_eval_duration": seconds(
                    "prompt_eval_duration"
                ),
                "eval_duration": seconds(
                    "eval_duration"
                ),
                "prompt_eval_count": data.get(
                    "prompt_eval_count"
                ),
                "prompt_eval_cached_count": data.get(
                    "prompt_eval_cached_count"
                ),
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason"),
                "error": ""
            }

        except Exception as e:
            return {
                "success": False,
                "client_latency": (
                    time.perf_counter() - start
                ),
                "error": str(e)
            }

    async def close(self):
        await self.client.aclose()
