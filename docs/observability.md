# Observability

- **Metrics (Prometheus):** req/sec, error rate, p50/p95/p99 latency, cache hit %, CPU/GPU/memory, queue depth.
- **Dashboards (Grafana):** service health, per-model panel (LLM, vision), infra panels (nodes, pods).
- **Alerts:** error rate > 5%, p99 > threshold, 5xx spikes, low cache hit rate, pod crash loops.
- **Tracing:** OpenTelemetry SDK in FastAPI; collector -> Jaeger (or vendor backend).
- **Logs:** Structured JSON via stdout; aggregate with Fluent Bit/Fluentd -> Elasticsearch/Cloud Logging.
