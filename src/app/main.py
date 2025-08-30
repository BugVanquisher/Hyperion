from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os, time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse, JSONResponse
from .models.llm import generate_text
from .cache import get_cache, cache_key

app = FastAPI(title="MultiModel-Serve", version="0.1.0")

# Prometheus metrics
REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["path", "method", "status"])
LATENCY = Histogram("http_request_latency_seconds", "Latency", ["path", "method"])

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = 128
    temperature: float = 0.7

class ChatResponse(BaseModel):
    model: str
    response: str
    tokens_used: int
    cached: bool
    processing_time_ms: int

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    try:
        LATENCY.labels(path=request.url.path, method=request.method).observe(duration)
        REQUESTS.labels(path=request.url.path, method=request.method, status=response.status_code).inc()
    except Exception:
        pass
    return response

@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": app.version}

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # cache first
    r = await get_cache()
    key = cache_key("llm:chat", req.model_dump())
    cached = await r.get(key)
    if cached:
        return JSONResponse(content={**eval(cached), "cached": True})

    # inference
    start = time.time()
    try:
        text, tokens, model_name = await generate_text(req.prompt, req.max_tokens, req.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    resp = {
        "model": model_name,
        "response": text,
        "tokens_used": tokens,
        "cached": False,
        "processing_time_ms": int((time.time() - start) * 1000),
    }
    # store cache briefly
    await r.setex(key, 300, repr(resp))
    return resp
