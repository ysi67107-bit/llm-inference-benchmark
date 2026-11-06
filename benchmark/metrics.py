
import statistics


def calculate_metrics(results):
    # 只统计成功请求
    successful = [
        r for r in results if r["success"]
    ]

    total_requests = len(results)
    successful_requests = len(successful)
    failed_requests = total_requests - successful_requests

    if not successful:
        return {
            "total_requests": total_requests,
            "successful_requests": 0,
            "failed_requests": failed_requests,
            "mean_latency": None,
            "p50_latency": None,
            "p95_latency": None
        }

    latencies = [
        r["latency"] for r in successful
    ]

    # 平均延迟
    mean_latency = statistics.mean(latencies)

    # 中位数（P50）
    p50_latency = statistics.median(latencies)

    # 使用线性插值计算 P95
    sorted_latencies = sorted(latencies)

    def percentile(values, p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)

        weight = position - lower

        return (
            values[lower] * (1 - weight)
            + values[upper] * weight
        )

    p95_latency = percentile(sorted_latencies, 0.95)

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "mean_latency": mean_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency
    }
