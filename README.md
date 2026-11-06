Preliminary Benchmark Results

Environment: Apple M2 Pro (16GB unified memory), Ollama, Qwen2.5-3B.

Methodology: 2 warmup requests followed by 20 requests per concurrency level. Concurrency levels: 1, 2, 4, and 8. Non-streaming inference with a maximum of 100 output tokens.

Results:

| Concurrency | Mean Latency (s) | P95 Latency (s) | Throughput (req/s) |
|---|---:|---:|---:|
| 1 | 1.469 | 1.479 | 0.681 |
| 2 | 2.870 | 2.958 | 0.679 |
| 4 | 5.526 | 6.004 | 0.670 |
| 8 | 9.910 | 12.043 | 0.666 |

Key Findings:

- All 80 benchmark requests completed successfully.
- Increasing client concurrency from 1 to 8 did not improve request throughput.
- Mean latency increased approximately 6.75×, while throughput decreased by approximately 2.2%.
- The results suggest a throughput bottleneck under the tested configuration, but do not establish its underlying cause.

Next Steps: Investigate Ollama parallelism settings, collect resource utilization metrics, and compare results against vLLM on an NVIDIA GPU.
