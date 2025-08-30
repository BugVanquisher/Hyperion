# Runtime Choices for Model Serving

- **Embedded Python worker (default):** simplest; great for early stage.
- **vLLM:** high-throughput LLM serving (KV cache, tensor parallel); easy to integrate behind FastAPI.
- **NVIDIA Triton:** multi-framework server (dynamic batching, concurrency tuning) for LLM/LVM workloads.
- **KServe:** Kubernetes-native control plane to declaratively deploy models (Triton/vLLM backends).

Recommendation: start embedded -> move hot paths to **vLLM** or **Triton**, orchestrate via **KServe** when fleet grows.
