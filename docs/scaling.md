# Scaling & Capacity Allocation

## Layers
1. **Pod autoscaling (HPA)**: target CPU 70% for gateway; for GPU workers use custom metrics (queue depth, GPU util).
2. **Event-driven (KEDA)**: scale on Redis queue length, PromQL latency, or requests-per-second.
3. **Node autoscaling**: Cluster Autoscaler or **Karpenter** for fast node provisioning.
4. **Predictive scaling (roadmap)**: use traffic time-series (Prophet/ARIMA) to pre-scale before daily peaks.

## Batching & Queues
- Enable micro-batching in workers (esp. Triton/vLLM) for higher GPU throughput.
- Keep bounded queues; apply timeouts and fallback to prevent overload (graceful degradation).

## Cost Controls
- Rightsize requests/limits; prefer FP16/BF16 or quantized models when acceptable.
- Prefer spot/preemptible nodes for non-critical workloads; maintain on-demand buffer.
