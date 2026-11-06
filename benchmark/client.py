
import time
from openai import OpenAI


class BenchmarkClient:
    def __init__(self, base_url, model_name):
        self.model_name = model_name

        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=120.0
        )

    def measure_latency(self, prompt):
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )

        end_time = time.perf_counter()

        latency = end_time - start_time

        return {
            "prompt": prompt,
            "latency": latency,
            "response": response.choices[0].message.content,
            "output_tokens": (
                response.usage.completion_tokens
                if response.usage else None
            )
        }
