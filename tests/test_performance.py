"""
Performance and load tests for Hyperion ML inference platform.
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.main import app

# Create test client
client = TestClient(app)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_health_endpoint_response_time(self):
        """Test health endpoint responds quickly."""
        start_time = time.time()
        response = client.get("/healthz")
        duration = time.time() - start_time

        assert response.status_code == 200
        assert duration < 0.1  # Should respond within 100ms

    def test_metrics_endpoint_response_time(self):
        """Test metrics endpoint responds quickly."""
        start_time = time.time()
        response = client.get("/metrics")
        duration = time.time() - start_time

        assert response.status_code == 200
        assert duration < 0.5  # Should respond within 500ms

    def test_root_endpoint_response_time(self):
        """Test root endpoint responds quickly."""
        start_time = time.time()
        response = client.get("/")
        duration = time.time() - start_time

        assert response.status_code == 200
        assert duration < 0.1  # Should respond within 100ms


@pytest.mark.performance
class TestConcurrentRequests:
    """Test performance under concurrent load."""

    def make_health_request(self):
        """Helper function to make a health check request."""
        response = client.get("/healthz")
        return response.status_code == 200

    def test_concurrent_health_checks(self):
        """Test multiple concurrent health check requests."""
        num_requests = 10

        with ThreadPoolExecutor(max_workers=5) as executor:
            start_time = time.time()
            futures = [
                executor.submit(self.make_health_request) for _ in range(num_requests)
            ]

            results = [future.result() for future in futures]
            duration = time.time() - start_time

        # All requests should succeed
        assert all(results)
        # Should complete within reasonable time
        assert duration < 2.0
        # Calculate average response time per request
        avg_time = duration / num_requests
        assert avg_time < 0.2  # Average response time should be under 200ms

    def make_chat_request(self):
        """Helper function to make a chat request."""
        try:
            response = client.post(
                "/v1/llm/chat",
                json={"prompt": "Hello", "max_tokens": 5, "temperature": 0.5},
            )
            return response.status_code in [200, 503]  # 503 if model not loaded
        except Exception:
            return False

    def test_concurrent_chat_requests(self):
        """Test multiple concurrent chat requests."""
        num_requests = 5

        with ThreadPoolExecutor(max_workers=3) as executor:
            start_time = time.time()
            futures = [
                executor.submit(self.make_chat_request) for _ in range(num_requests)
            ]

            results = [future.result() for future in futures]
            duration = time.time() - start_time

        # All requests should get valid responses
        assert all(results)
        # Should complete within reasonable time (allowing for model loading)
        assert duration < 30.0


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage characteristics."""

    def test_memory_after_multiple_requests(self):
        """Test memory doesn't leak after multiple requests."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Make multiple requests
        for _ in range(50):
            response = client.get("/healthz")
            assert response.status_code == 200

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024


@pytest.mark.load
class TestLoadTests:
    """Load tests for the application."""

    @pytest.mark.slow
    def test_sustained_load_health_endpoint(self):
        """Test sustained load on health endpoint."""
        duration_seconds = 10
        start_time = time.time()
        request_count = 0
        errors = 0

        while time.time() - start_time < duration_seconds:
            try:
                response = client.get("/healthz")
                if response.status_code != 200:
                    errors += 1
                request_count += 1
            except Exception:
                errors += 1
                request_count += 1

        # Calculate requests per second
        rps = request_count / duration_seconds

        # Should handle at least 10 RPS with minimal errors
        assert rps >= 10
        assert errors / request_count < 0.05  # Less than 5% error rate

    @pytest.mark.slow
    def test_burst_load(self):
        """Test handling burst of requests."""
        num_requests = 50
        errors = 0

        def make_request():
            try:
                response = client.get("/healthz")
                return response.status_code == 200
            except Exception:
                return False

        start_time = time.time()

        # Send all requests in parallel
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]

        duration = time.time() - start_time
        success_count = sum(results)

        # At least 90% should succeed
        assert success_count / num_requests >= 0.9
        # Should complete within reasonable time
        assert duration < 5.0


@pytest.mark.asyncio
@pytest.mark.performance
class TestAsyncPerformance:
    """Test async performance characteristics."""

    async def make_async_request(self, session, url):
        """Make an async HTTP request."""
        import aiohttp

        async with session.get(url) as response:
            return response.status

    async def test_async_concurrent_requests(self):
        """Test async concurrent request handling."""
        import aiohttp

        base_url = "http://localhost:8000"  # Assuming test server
        num_requests = 20

        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            # Create concurrent requests
            tasks = [
                self.make_async_request(session, f"{base_url}/healthz")
                for _ in range(num_requests)
            ]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time

                # Count successful requests (status 200)
                success_count = sum(1 for r in results if r == 200)

                # Most requests should succeed if server is running
                # Skip assertion if server is not available
                if success_count > 0:
                    assert success_count / num_requests >= 0.8
                    assert duration < 2.0

            except Exception:
                # Skip test if server is not available
                pytest.skip("Test server not available for async testing")


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for key operations."""

    def test_json_serialization_performance(self):
        """Test JSON serialization performance."""
        import json

        large_data = {
            "prompt": "x" * 1000,
            "max_tokens": 100,
            "temperature": 0.7,
            "metadata": {"key": "value"} * 100,
        }

        # Time JSON serialization
        start_time = time.time()
        for _ in range(1000):
            json.dumps(large_data)
        duration = time.time() - start_time

        # Should complete 1000 serializations quickly
        assert duration < 1.0

    def test_cache_key_generation_performance(self):
        """Test cache key generation performance."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from app.cache import cache_key

        test_payload = {
            "prompt": "Test prompt",
            "max_tokens": 100,
            "temperature": 0.7,
            "additional_data": ["item"] * 100,
        }

        # Time cache key generation
        start_time = time.time()
        for _ in range(1000):
            cache_key("test", test_payload)
        duration = time.time() - start_time

        # Should generate 1000 cache keys quickly
        assert duration < 0.5