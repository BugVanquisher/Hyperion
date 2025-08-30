# Architecture

- **Gateway:** FastAPI (async), exposes `/v1/llm/chat`, `/healthz`, `/metrics`.
- **Model Workers:** start embedded -> evolve to dedicated services. Compatible with **vLLM**, **Triton**, **KServe**, **TorchServe**.
- **Cache:** Redis to cut latency/cost; keyed by input+params; TTL 5–60m depending on use.
- **Autoscaling:** HPA (CPU/latency), KEDA (queue length/custom PromQL), node autoscaling (Cluster Autoscaler/Karpenter).
- **Observability:** Prometheus metrics + Grafana dashboards; tracing via OpenTelemetry -> Jaeger.
- **Resilience:** Liveness/readiness probes, rolling updates, load shedding (queue caps/timeouts).
