
import time

import httpx


class OllamaNativeProfiler:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def profile(
        self,
        prompt: str,
        max_tokens: int = 128,
    ):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0,
            },
        }

        async with httpx.AsyncClient(
            timeout=180.0
        ) as client:

            start = time.perf_counter()

            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

            end = time.perf_counter()

        def ns_to_ms(value):
            return value / 1_000_000 if value is not None else None

        prompt_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")

        prefill_ms = ns_to_ms(
            data.get("prompt_eval_duration")
        )

        decode_ms = ns_to_ms(
            data.get("eval_duration")
        )

        return {
            "model": self.model,
            "client_latency_ms": (end - start) * 1000,
            "server_total_ms": ns_to_ms(
                data.get("total_duration")
            ),
            "load_ms": ns_to_ms(
                data.get("load_duration")
            ),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_tokens_per_s": (
                prompt_tokens / (prefill_ms / 1000)
                if prompt_tokens is not None
                and prefill_ms is not None
                and prefill_ms > 0
                else None
            ),
            "decode_tokens_per_s": (
                output_tokens / (decode_ms / 1000)
                if output_tokens is not None
                and decode_ms is not None
                and decode_ms > 0
                else None
            ),
            "done_reason": data.get("done_reason"),
        }
