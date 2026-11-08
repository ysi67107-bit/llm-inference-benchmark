
import json
import time

import httpx


class OllamaStreamingProfiler:
    def __init__(
        self,
        model="qwen2.5:3b",
        base_url="http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def profile(
        self,
        prompt,
        max_tokens=128,
    ):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0,
            },
        }

        first_token_time = None
        output_chunks = 0
        final_data = None

        async with httpx.AsyncClient(
            timeout=180.0
        ) as client:

            start = time.perf_counter()

            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    # 首次收到非空生成文本
                    if data.get("response"):
                        output_chunks += 1

                        if first_token_time is None:
                            first_token_time = (
                                time.perf_counter()
                            )

                    if data.get("done"):
                        final_data = data

            end = time.perf_counter()

        if final_data is None:
            raise RuntimeError(
                "Streaming response ended without final metadata"
            )

        ttft_ms = (
            (first_token_time - start) * 1000
            if first_token_time is not None
            else None
        )

        latency_ms = (end - start) * 1000

        output_tokens = final_data.get("eval_count")

        # 客户端观测的平均 Token 间隔
        # 注意：不等于逐 Token 的真实间隔
        if (
            ttft_ms is not None
            and output_tokens is not None
            and output_tokens > 1
        ):
            tpot_ms = (
                latency_ms - ttft_ms
            ) / (output_tokens - 1)
        else:
            tpot_ms = None

        return {
            "model": self.model,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "client_latency_ms": latency_ms,
            "output_tokens": output_tokens,
            "output_chunks": output_chunks,
            "server_total_ms": (
                final_data.get("total_duration", 0)
                / 1_000_000
            ),
            "prefill_ms": (
                final_data.get("prompt_eval_duration", 0)
                / 1_000_000
            ),
            "decode_ms": (
                final_data.get("eval_duration", 0)
                / 1_000_000
            ),
            "done_reason": final_data.get("done_reason"),
        }
