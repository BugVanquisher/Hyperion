import asyncio
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.main import app

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Hyperion ML Inference Platform"
    assert "endpoints" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert "version" in data
    assert "model_loaded" in data


def test_metrics_endpoint():
    """Test that metrics endpoint returns Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Check for some expected Prometheus metrics
    content = response.text
    assert "http_requests_total" in content


def test_models_endpoint():
    """Test the models listing endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_chat_endpoint_structure():
    """Test chat endpoint accepts valid requests."""
    # Note: This test might be slow on first run due to model loading
    test_payload = {
        "prompt": "Hello, how are you?",
        "max_tokens": 10,
        "temperature": 0.5,
    }

    response = client.post("/v1/llm/chat", json=test_payload)

    # Should either succeed or fail with 503 if model is still loading
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "model" in data
        assert "response" in data
        assert "tokens_used" in data
        assert "cached" in data
        assert "processing_time_ms" in data


def test_chat_endpoint_validation():
    """Test chat endpoint validates input properly."""
    # Test empty prompt
    response = client.post("/v1/llm/chat", json={"prompt": "", "max_tokens": 10})
    assert response.status_code == 422

    # Test prompt too long
    response = client.post(
        "/v1/llm/chat",
        json={"prompt": "x" * 3000, "max_tokens": 10},  # Over 2000 char limit
    )
    assert response.status_code == 422

    # Test invalid max_tokens
    response = client.post(
        "/v1/llm/chat", json={"prompt": "Hello", "max_tokens": 0}  # Below minimum
    )
    assert response.status_code == 422

    # Test invalid temperature
    response = client.post(
        "/v1/llm/chat",
        json={
            "prompt": "Hello",
            "max_tokens": 10,
            "temperature": -1.0,  # Below minimum
        },
    )
    assert response.status_code == 422


# Test for cache functionality
def test_cache_functionality():
    """Test that caching works correctly."""
    test_payload = {
        "prompt": "Test caching with this specific prompt",
        "max_tokens": 5,
        "temperature": 0.0,  # Deterministic
    }

    # First request
    response1 = client.post("/v1/llm/chat", json=test_payload)
    if response1.status_code == 200:
        # Second identical request should be faster (cached)
        response2 = client.post("/v1/llm/chat", json=test_payload)

        assert response2.status_code == 200
        data1 = response1.json()
        data2 = response2.json()

        # Second request should be marked as cached
        assert data2.get("cached") is True
        # Response should be identical
        assert data1["response"] == data2["response"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
