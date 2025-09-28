import os
import time
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from opentelemetry import trace
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from .alerts import AlertmanagerWebhook, alert_processor
from .batching import get_batcher
from .cache import cache_key, get_cache
from .logging_config import (get_ml_logger, log_batch_metrics,
                             log_cache_operation, log_gpu_metrics,
                             log_inference_metrics, setup_structured_logging)
from .models.llm import (generate_text, get_device_info, health_check,
                         init_model)
from .tracing import (add_span_attributes, get_trace_id, instrument_app,
                      trace_operation)

# Set up structured logging
enable_json_logging = os.getenv("ENABLE_JSON_LOGS", "true").lower() == "true"
logger = setup_structured_logging(
    service_name="hyperion-app",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_json=enable_json_logging,
)

# Get ML-aware logger
ml_logger = get_ml_logger("hyperion.ml", service="hyperion-app")

app = FastAPI(
    title="Hyperion - ML Inference Platform",
    description="Scalable, Observable, and Reliable inference platform for LLMs",
    version="0.1.0",
)

# Initialize distributed tracing
instrument_app(app)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["path", "method", "status"]
)
LATENCY = Histogram(
    "http_request_latency_seconds", "Request latency in seconds", ["path", "method"]
)
INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds", "Time spent on model inference", ["model_name"]
)
CACHE_REQUESTS = Counter(
    "cache_requests_total", "Cache requests", ["status"]  # hit/miss
)
BATCH_REQUESTS = Counter(
    "batch_requests_total", "Requests processed through batching", ["batch_size"]
)
BATCH_WAIT_TIME = Histogram(
    "batch_wait_time_seconds", "Time requests spend waiting in batch queue"
)

# GPU Metrics
GPU_MEMORY_ALLOCATED = Gauge(
    "gpu_memory_allocated_bytes", "GPU memory currently allocated"
)
GPU_MEMORY_TOTAL = Gauge("gpu_memory_total_bytes", "Total GPU memory available")
MODEL_OPTIMIZATION_ENABLED = Gauge(
    "model_optimization_enabled",
    "Whether model optimization is enabled",
    ["optimization_type"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Input text prompt"
    )
    max_tokens: int = Field(
        default=64, ge=1, le=200, description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )


class ChatResponse(BaseModel):
    model: str
    response: str
    tokens_used: int
    cached: bool
    processing_time_ms: int


class HealthResponse(BaseModel):
    ok: bool
    version: str
    model_loaded: bool
    timestamp: str
    device_info: dict


# Global startup flag
model_loading_complete = False


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_loading_complete
    logger.info("Starting Hyperion service...")

    try:
        await init_model()
        model_loading_complete = True
        logger.info("Service startup complete!")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # Don't fail startup, but mark model as not loaded
        model_loading_complete = False


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """Collect Prometheus metrics for all requests."""
    start_time = time.time()

    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Record metrics
    try:
        LATENCY.labels(path=request.url.path, method=request.method).observe(duration)

        REQUESTS.labels(
            path=request.url.path, method=request.method, status=response.status_code
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record metrics: {str(e)}")

    return response


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Health check endpoint with detailed status."""
    model_healthy = await health_check() if model_loading_complete else False

    return HealthResponse(
        ok=model_loading_complete and model_healthy,
        version=app.version,
        model_loaded=model_loading_complete,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        device_info=get_device_info(),
    )


@app.get("/readiness")
async def readiness():
    """Kubernetes readiness probe - simpler check."""
    if not model_loading_complete:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"ready": True}


def update_gpu_metrics():
    """Update GPU-related Prometheus metrics."""
    import torch

    try:
        device_info = get_device_info()

        if "gpu_memory_allocated" in device_info:
            # Parse memory values (remove 'GB' and convert to bytes)
            allocated_gb = float(device_info["gpu_memory_allocated"].replace("GB", ""))
            total_gb = float(device_info["gpu_memory_total"].replace("GB", ""))

            GPU_MEMORY_ALLOCATED.set(allocated_gb * 1e9)
            GPU_MEMORY_TOTAL.set(total_gb * 1e9)

            # Log GPU metrics in structured format
            log_gpu_metrics(ml_logger, device_info)

        # Update optimization metrics
        if "optimizations" in device_info:
            opts = device_info["optimizations"]
            MODEL_OPTIMIZATION_ENABLED.labels(optimization_type="quantization").set(
                1 if opts.get("quantization_enabled", False) else 0
            )
            MODEL_OPTIMIZATION_ENABLED.labels(optimization_type="compilation").set(
                1 if opts.get("optimization_enabled", False) else 0
            )

    except Exception as e:
        logger.warning(f"Failed to update GPU metrics: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update GPU metrics before generating response
    update_gpu_metrics()

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Generate text using the loaded LLM.

    This endpoint supports caching to improve response times and reduce costs.
    """
    with trace_operation(
        "llm_chat_request",
        {
            "prompt_length": len(req.prompt),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "trace_id": get_trace_id(),
        },
    ) as span:

        # Check if model is ready
        if not model_loading_complete:
            span.set_status(trace.Status(trace.StatusCode.ERROR, "Model not loaded"))
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few moments.",
            )

        # Try cache first
        cache_hit = False
        with trace_operation(
            "cache_lookup", {"cache_key": cache_key("llm:chat", req.model_dump())[:50]}
        ):
            try:
                r = await get_cache()
                key = cache_key("llm:chat", req.model_dump())
                cached_response = await r.get(key)

                if cached_response:
                    CACHE_REQUESTS.labels(status="hit").inc()
                    cached_data = eval(
                        cached_response
                    )  # Note: In production, use json.loads
                    cached_data["cached"] = True
                    cache_hit = True

                    add_span_attributes(
                        span, {"cache_hit": True, "response_cached": True}
                    )

                    log_cache_operation(ml_logger, "lookup", key, hit=True)
                    return ChatResponse(**cached_data)
                else:
                    CACHE_REQUESTS.labels(status="miss").inc()
                    add_span_attributes(span, {"cache_hit": False})
                    log_cache_operation(ml_logger, "lookup", key, hit=False)

            except Exception as e:
                logger.warning(f"Cache error: {str(e)}. Proceeding without cache.")
                add_span_attributes(span, {"cache_error": str(e)})

        # Perform inference using batching
        start_time = time.time()
        with trace_operation(
            "model_inference",
            {
                "request_id": f"{int(time.time() * 1000)}_{hash(req.prompt) % 10000}",
                "prompt_preview": req.prompt[:100],
            },
        ):
            try:
                batcher = get_batcher()
                request_id = f"{int(time.time() * 1000)}_{hash(req.prompt) % 10000}"

                with BATCH_WAIT_TIME.time():
                    batch_result = await batcher.add_request(
                        prompt=req.prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        request_id=request_id,
                    )

                # Record batch metrics
                batch_stats = batcher.get_stats()
                if batch_stats["total_batches"] > 0:
                    avg_batch_size = int(batch_stats["avg_requests_per_batch"])
                    BATCH_REQUESTS.labels(batch_size=str(avg_batch_size)).inc()

                    add_span_attributes(
                        span,
                        {
                            "batch_size": avg_batch_size,
                            "total_batches": batch_stats["total_batches"],
                            "avg_batch_time_ms": batch_stats.get(
                                "avg_batch_time_ms", 0
                            ),
                        },
                    )

                text = batch_result.response
                tokens = batch_result.tokens_used
                model_name = batch_result.model_name

                add_span_attributes(
                    span,
                    {
                        "model_name": model_name,
                        "tokens_generated": tokens,
                        "response_preview": text[:100],
                    },
                )

                # Log inference metrics
                log_inference_metrics(
                    ml_logger,
                    model_name,
                    tokens,
                    int((time.time() - start_time) * 1000),
                )

            except Exception as e:
                logger.error(f"Inference failed: {str(e)}")
                add_span_attributes(span, {"inference_error": str(e)})
                raise HTTPException(
                    status_code=500, detail=f"Inference failed: {str(e)}"
                )

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_data = ChatResponse(
            model=model_name,
            response=text,
            tokens_used=tokens,
            cached=False,
            processing_time_ms=processing_time_ms,
        )

        # Cache the response
        with trace_operation("cache_store"):
            try:
                r = await get_cache()
                cache_ttl = 300  # 5 minutes
                await r.setex(key, cache_ttl, repr(response_data.model_dump()))
                log_cache_operation(ml_logger, "store", key)
                add_span_attributes(span, {"response_cached": True})
            except Exception as e:
                logger.warning(f"Failed to cache response: {str(e)}")
                add_span_attributes(span, {"cache_store_error": str(e)})

        # Final span attributes
        add_span_attributes(
            span, {"processing_time_ms": processing_time_ms, "success": True}
        )

        return response_data


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": os.getenv("MODEL_NAME", "microsoft/DialoGPT-small"),
                "status": "loaded" if model_loading_complete else "loading",
            }
        ]
    }


@app.get("/v1/batch/stats")
async def batch_stats():
    """Get current batching statistics."""
    batcher = get_batcher()
    stats = batcher.get_stats()
    stats.update(get_device_info())
    return stats


# Alert endpoints
@app.post("/alerts/{component}")
async def receive_alert(component: str, webhook: AlertmanagerWebhook):
    """Receive and process alerts from Alertmanager."""
    try:
        alert_event = alert_processor.process_webhook(webhook)

        ml_logger.info(
            f"Alert received for component: {component}",
            extra={
                "extra_fields": {
                    "alert": {
                        "component": component,
                        "status": webhook.status,
                        "alert_count": len(webhook.alerts),
                        "severity": webhook.commonLabels.get("severity", "unknown"),
                        "alertname": webhook.commonLabels.get("alertname", "unknown"),
                    }
                }
            },
        )

        return {
            "status": "received",
            "component": component,
            "alert_count": len(webhook.alerts),
            "processed_at": alert_event["timestamp"],
        }

    except Exception as e:
        logger.error(f"Failed to process alert for {component}: {str(e)}")
        return {"status": "error", "message": str(e)}, 500


@app.get("/alerts/summary")
async def alert_summary():
    """Get current alert summary."""
    return alert_processor.get_alert_summary()


@app.get("/alerts/active")
async def active_alerts():
    """Get currently active alerts."""
    active = []
    for fingerprint, alert in alert_processor.active_alerts.items():
        active.append(
            {
                "fingerprint": fingerprint,
                "alertname": alert.labels.get("alertname"),
                "component": alert.labels.get("component"),
                "severity": alert.labels.get("severity"),
                "summary": alert.annotations.get("summary"),
                "starts_at": alert.startsAt,
            }
        )
    return {"active_alerts": active, "count": len(active)}


@app.get("/alerts/history")
async def alert_history(limit: int = 50):
    """Get recent alert history."""
    return {
        "history": alert_processor.alert_history[-limit:],
        "total_count": len(alert_processor.alert_history),
    }


@app.post("/test/simulate-alert/{alert_type}")
async def simulate_alert(alert_type: str):
    """Simulate alert conditions for testing (development only)."""

    if os.getenv("ENVIRONMENT", "development") != "development":
        raise HTTPException(
            status_code=403, detail="Alert simulation only available in development"
        )

    if alert_type == "gpu-memory":
        # Artificially report high GPU memory usage
        GPU_MEMORY_ALLOCATED.set(0.9 * GPU_MEMORY_TOTAL._value.get())
        return {
            "status": "simulated",
            "alert_type": "gpu-memory",
            "message": "GPU memory usage set to 90%",
        }

    elif alert_type == "high-latency":
        # This would normally be handled by the inference system
        return {
            "status": "simulated",
            "alert_type": "high-latency",
            "message": "Use slow inference requests to trigger",
        }

    elif alert_type == "service-down":
        return {
            "status": "simulated",
            "alert_type": "service-down",
            "message": "Stop the service to trigger this alert",
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown alert type: {alert_type}")


# Add a simple root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service info."""
    return {
        "service": "Hyperion ML Inference Platform",
        "version": app.version,
        "status": "running",
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics",
            "chat": "/v1/llm/chat",
            "models": "/v1/models",
            "batch_stats": "/v1/batch/stats",
        },
    }
