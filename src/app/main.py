from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse, JSONResponse
from .models.llm import generate_text, init_model, health_check, get_device_info
from .cache import get_cache, cache_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hyperion - ML Inference Platform",
    description="Scalable, Observable, and Reliable inference platform for LLMs",
    version="0.1.0"
)

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
    "http_requests_total", 
    "Total HTTP requests", 
    ["path", "method", "status"]
)
LATENCY = Histogram(
    "http_request_latency_seconds", 
    "Request latency in seconds", 
    ["path", "method"]
)
INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Time spent on model inference",
    ["model_name"]
)
CACHE_REQUESTS = Counter(
    "cache_requests_total",
    "Cache requests",
    ["status"]  # hit/miss
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Input text prompt")
    max_tokens: int = Field(default=64, ge=1, le=200, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")

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
        LATENCY.labels(
            path=request.url.path, 
            method=request.method
        ).observe(duration)
        
        REQUESTS.labels(
            path=request.url.path, 
            method=request.method, 
            status=response.status_code
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
        device_info=get_device_info()
    )

@app.get("/readiness")
async def readiness():
    """Kubernetes readiness probe - simpler check."""
    if not model_loading_complete:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"ready": True}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest(), 
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/v1/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Generate text using the loaded LLM.
    
    This endpoint supports caching to improve response times and reduce costs.
    """
    # Check if model is ready
    if not model_loading_complete:
        raise HTTPException(
            status_code=503, 
            detail="Model is still loading. Please try again in a few moments."
        )
    
    # Try cache first
    cache_hit = False
    try:
        r = await get_cache()
        key = cache_key("llm:chat", req.model_dump())
        cached_response = await r.get(key)
        
        if cached_response:
            CACHE_REQUESTS.labels(status="hit").inc()
            cached_data = eval(cached_response)  # Note: In production, use json.loads
            cached_data["cached"] = True
            logger.info(f"Cache hit for prompt: '{req.prompt[:50]}...'")
            return ChatResponse(**cached_data)
        else:
            CACHE_REQUESTS.labels(status="miss").inc()
    except Exception as e:
        logger.warning(f"Cache error: {str(e)}. Proceeding without cache.")
    
    # Perform inference
    start_time = time.time()
    try:
        with INFERENCE_DURATION.labels(model_name="current_model").time():
            text, tokens, model_name = await generate_text(
                req.prompt, 
                req.max_tokens, 
                req.temperature
            )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Inference failed: {str(e)}"
        )
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    response_data = ChatResponse(
        model=model_name,
        response=text,
        tokens_used=tokens,
        cached=False,
        processing_time_ms=processing_time_ms
    )
    
    # Cache the response
    try:
        r = await get_cache()
        cache_ttl = 300  # 5 minutes
        await r.setex(key, cache_ttl, repr(response_data.model_dump()))
        logger.info(f"Cached response for prompt: '{req.prompt[:50]}...'")
    except Exception as e:
        logger.warning(f"Failed to cache response: {str(e)}")
    
    return response_data

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": os.getenv("MODEL_NAME", "microsoft/DialoGPT-small"),
                "status": "loaded" if model_loading_complete else "loading"
            }
        ]
    }

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
            "models": "/v1/models"
        }
    }