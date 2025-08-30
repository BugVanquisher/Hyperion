import os
from redis import asyncio as aioredis
import hashlib, json

_redis = None

async def get_cache():
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        _redis = aioredis.from_url(url, decode_responses=True)
    return _redis

def cache_key(prefix: str, payload: dict) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{prefix}:{h}"
