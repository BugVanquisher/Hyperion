# MultiModel-Serve
**Scalable, Observable, and Reliable inference platform** for **LLMs** (extensible to **LVMs**) on **Kubernetes**.  
Built with **Python + FastAPI** (Go-ready), featuring **wise capacity allocation** (HPA/KEDA/Karpenter-ready), **caching**, and **deep observability** (Prometheus, Grafana, OpenTelemetry).

> This README merges the strategic rigor of your internal *Project Hyperion* with a clean, GitHub-friendly structure for fast adoption.

## ✨ Highlights
- **LLM-first, LVM-ready**: text endpoints now; vision endpoints drop-in later.
- **Kubernetes-native**: horizontal/cluster autoscaling, health probes, rolling deploys.
- **Wise capacity allocation**: HPA by CPU & custom metrics; KEDA for event-driven; node autoscaling with Karpenter/Cluster Autoscaler.
- **Caching built-in**: Redis for low-latency hot-paths and lower cost/req.
- **Observability first**: Prometheus metrics, Grafana dashboards, logs, tracing hooks.
- **Mac-friendly dev**: `docker compose up` and Kind/Minikube.

## 🔭 Architecture (High-level)
```mermaid
flowchart LR
    C[Client] --> I[Ingress/LB]
    I --> A[FastAPI Gateway]
    A -->|cache hit| R[(Redis)]
    A -->|cache miss| W1[Model Worker(s)]
    subgraph Kubernetes Cluster
      A --- R
      A --- W1
      classDef k8s fill:#eef,stroke:#99f,stroke-width:1px;
      class A,R,W1 k8s;
    end
    subgraph Observability
      P[Prometheus] --> G[Grafana]
      T[Tracing (OTel->Jaeger)]
    end
    A -->|/metrics| P
    A --> T
```
- **Gateway:** FastAPI (async, streaming optional).  
- **Workers:** start with embedded Python workers; can swap in **vLLM**/**Triton**/**KServe** later (see [docs/runtime](./docs/runtime.md)).  
- **Cache:** Redis for hot results and idempotent replies.  
- **Autoscaling:** HPA by CPU/latency; **KEDA** for queue length or custom PromQL; **Karpenter/Cluster Autoscaler** for nodes.  
- **Observability:** Prometheus scrape + Grafana dashboards; OpenTelemetry for traces.

## 🧭 Repo Layout
```
.
├─ src/app/                # FastAPI app + model stubs
├─ docs/                   # Deep dives: scaling, observability, runtime, gateway
├─ deploy/docker/          # Dockerfile + docker-compose for local dev
├─ deploy/k8s/             # K8s manifests (app, redis, hpa, ingress)
├─ deploy/helm/multimodel-serve/  # Helm chart skeleton
├─ .github/workflows/      # CI pipeline
├─ scripts/                # helper scripts
└─ README.md               # this file
```

## 🚀 Quickstart (Local, Mac-friendly)
Requirements: Docker Desktop (or Colima), Python 3.10+, Make.

```bash
# 1) Run API + Redis locally
docker compose -f deploy/docker/docker-compose.yml up --build

# 2) Call the API
curl -X POST http://localhost:8000/v1/llm/chat -H "Content-Type: application/json" \
  -d '{"prompt":"Explain transformers briefly.","max_tokens":64}'

# 3) Prometheus metrics (for scrape or debugging)
curl http://localhost:8000/metrics
```

## ☁️ Deploy to Kubernetes (dev)
```bash
# Create namespace
kubectl apply -f deploy/k8s/namespace.yaml

# App + Redis
kubectl apply -f deploy/k8s/app-deployment.yaml
kubectl apply -f deploy/k8s/app-service.yaml
kubectl apply -f deploy/k8s/redis-deployment.yaml
kubectl apply -f deploy/k8s/redis-service.yaml

# Autoscaling (tune targets in-file)
kubectl apply -f deploy/k8s/hpa-app.yaml

# Optional Ingress (requires an ingress controller like NGINX)
kubectl apply -f deploy/k8s/ingress.yaml
```

> For **Prometheus/Grafana**, prefer Helm: see [docs/observability.md](./docs/observability.md).

## 📈 Capacity & Scaling
- **Start** with CPU HPA on gateway, GPU HPA on workers (if used).  
- **Enable KEDA** for event-driven scaling (queue length, latency PromQL).  
- **Node autoscaling** with Cluster Autoscaler/**Karpenter**.  
- **Predictive autoscaling** (Hyperion roadmap): see [docs/scaling.md](./docs/scaling.md).

## 🧪 API Examples
```bash
# Chat (LLM)
curl -X POST http://localhost:8000/v1/llm/chat -H "Content-Type: application/json" \
  -d '{"prompt":"What is RAG?", "max_tokens":120, "temperature":0.7}'

# Health
curl http://localhost:8000/healthz
```

## 📚 Deeper Docs
- [Architecture](./docs/architecture.md) – components & flow  
- [Scaling](./docs/scaling.md) – HPA/KEDA/Karpenter + predictive roadmap  
- [Observability](./docs/observability.md) – metrics, logs, traces, SLOs  
- [Gateway](./docs/gateway.md) – NGINX/Istio/APISIX/Kong trade-offs  
- [Runtime](./docs/runtime.md) – Triton vs vLLM vs TorchServe vs KServe  
- [Roadmap](./docs/roadmap.md) – milestones & extensions

## 🛠️ Dev & Build
```bash
# Install (optional, for running outside Docker)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🤝 Contributing
Please see [CONTRIBUTING.md](./CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## 🪪 License
MIT © You
