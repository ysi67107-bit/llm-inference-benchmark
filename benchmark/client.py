
import time
from openai import OpenAI


class BenchmarkClient:
    def __init__(self, base_url, model_name):
        self.model_name = model_name

        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=120.0,
            max_retries=0
        )

    def measure_latency(self, prompt):
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )

            latency = time.perf_counter() - start_time

            output_tokens = (
                response.usage.completion_tokens
                if response.usage else None
            )

            return {
                "latency": latency,
                "output_tokens": output_tokens,
                "success": True,
                "error": ""
            }

        except Exception as e:
            latency = time.perf_counter() - start_time

            return {
                "latency": latency,
                "output_tokens": None,
                "success": False,
                "error": str(e)
            }

