"""
Tests for caching functionality.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.cache import cache_key, get_cache


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_basic(self):
        """Test basic cache key generation."""
        payload = {"prompt": "hello", "max_tokens": 10}
        key = cache_key("test", payload)

        assert key.startswith("test:")
        assert len(key) > 5  # Should have hash component

    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        payload = {"prompt": "hello", "max_tokens": 10}
        key1 = cache_key("test", payload)
        key2 = cache_key("test", payload)

        assert key1 == key2

    def test_cache_key_different_payloads(self):
        """Test that different payloads generate different keys."""
        payload1 = {"prompt": "hello", "max_tokens": 10}
        payload2 = {"prompt": "world", "max_tokens": 10}

        key1 = cache_key("test", payload1)
        key2 = cache_key("test", payload2)

        assert key1 != key2

    def test_cache_key_different_prefixes(self):
        """Test that different prefixes generate different keys."""
        payload = {"prompt": "hello", "max_tokens": 10}
        key1 = cache_key("prefix1", payload)
        key2 = cache_key("prefix2", payload)

        assert key1 != key2
        assert key1.startswith("prefix1:")
        assert key2.startswith("prefix2:")

    def test_cache_key_order_independence(self):
        """Test that key order in payload doesn't affect cache key."""
        payload1 = {"prompt": "hello", "max_tokens": 10, "temperature": 0.7}
        payload2 = {"max_tokens": 10, "temperature": 0.7, "prompt": "hello"}

        key1 = cache_key("test", payload1)
        key2 = cache_key("test", payload2)

        assert key1 == key2

    def test_cache_key_nested_objects(self):
        """Test cache key generation with nested objects."""
        payload = {
            "prompt": "hello",
            "options": {"max_tokens": 10, "temperature": 0.7},
            "metadata": {"user": "test", "session": "123"},
        }

        key = cache_key("test", payload)
        assert key.startswith("test:")
        assert len(key) > 10

    def test_cache_key_with_none_values(self):
        """Test cache key generation with None values."""
        payload = {"prompt": "hello", "max_tokens": None, "temperature": 0.7}

        key = cache_key("test", payload)
        assert key.startswith("test:")

    def test_cache_key_with_lists(self):
        """Test cache key generation with list values."""
        payload = {"prompt": "hello", "tokens": [1, 2, 3], "options": ["a", "b"]}

        key = cache_key("test", payload)
        assert key.startswith("test:")


@pytest.mark.asyncio
class TestCacheConnection:
    """Test Redis cache connection functionality."""

    @patch("app.cache.aioredis.from_url")
    async def test_get_cache_creates_connection(self, mock_from_url):
        """Test that get_cache creates Redis connection."""
        # Reset the global cache
        import app.cache

        app.cache._redis = None

        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis

        result = await get_cache()

        assert result == mock_redis
        mock_from_url.assert_called_once_with(
            "redis://redis:6379/0", decode_responses=True
        )

    @patch("app.cache.aioredis.from_url")
    async def test_get_cache_reuses_connection(self, mock_from_url):
        """Test that get_cache reuses existing connection."""
        # Reset and set up initial connection
        import app.cache

        mock_redis = AsyncMock()
        app.cache._redis = mock_redis

        result = await get_cache()

        assert result == mock_redis
        # Should not create new connection
        mock_from_url.assert_not_called()

    @patch.dict(os.environ, {"REDIS_URL": "redis://custom:1234/1"})
    @patch("app.cache.aioredis.from_url")
    async def test_get_cache_custom_url(self, mock_from_url):
        """Test get_cache with custom Redis URL."""
        # Reset the global cache
        import app.cache

        app.cache._redis = None

        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis

        await get_cache()

        mock_from_url.assert_called_once_with(
            "redis://custom:1234/1", decode_responses=True
        )


@pytest.mark.asyncio
@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for cache functionality (requires Redis)."""

    @pytest.fixture
    async def cache(self):
        """Fixture to provide cache connection."""
        try:
            cache = await get_cache()
            # Test if Redis is available
            await cache.ping()
            yield cache
        except Exception:
            pytest.skip("Redis not available for integration tests")

    async def test_cache_set_and_get(self, cache):
        """Test basic cache set and get operations."""
        key = "test:key:123"
        value = "test_value"

        await cache.set(key, value, ex=60)  # Set with 60s expiration
        result = await cache.get(key)

        assert result == value

    async def test_cache_expiration(self, cache):
        """Test cache key expiration."""
        key = "test:expire:123"
        value = "test_value"

        await cache.set(key, value, ex=1)  # Set with 1s expiration

        # Should exist immediately
        result = await cache.get(key)
        assert result == value

        # Wait for expiration (add small buffer)
        await asyncio.sleep(1.1)

        result = await cache.get(key)
        assert result is None

    async def test_cache_delete(self, cache):
        """Test cache key deletion."""
        key = "test:delete:123"
        value = "test_value"

        await cache.set(key, value)
        await cache.delete(key)

        result = await cache.get(key)
        assert result is None

    async def test_cache_exists(self, cache):
        """Test checking if cache key exists."""
        key = "test:exists:123"
        value = "test_value"

        # Key should not exist initially
        exists = await cache.exists(key)
        assert exists == 0

        # Set key and check existence
        await cache.set(key, value)
        exists = await cache.exists(key)
        assert exists == 1

    async def test_cache_json_storage(self, cache):
        """Test storing and retrieving JSON data."""
        import json

        key = "test:json:123"
        data = {"prompt": "hello", "max_tokens": 10, "temperature": 0.7}

        # Store as JSON string
        await cache.set(key, json.dumps(data))

        # Retrieve and parse
        result = await cache.get(key)
        parsed_data = json.loads(result)

        assert parsed_data == data

    async def test_cache_concurrent_access(self, cache):
        """Test concurrent cache access."""
        keys = [f"test:concurrent:{i}" for i in range(10)]
        values = [f"value_{i}" for i in range(10)]

        # Set multiple keys concurrently
        set_tasks = [cache.set(key, value) for key, value in zip(keys, values)]
        await asyncio.gather(*set_tasks)

        # Get multiple keys concurrently
        get_tasks = [cache.get(key) for key in keys]
        results = await asyncio.gather(*get_tasks)

        assert results == values


@pytest.mark.unit
class TestCacheUtilities:
    """Unit tests for cache utility functions."""

    def test_cache_key_with_large_payload(self):
        """Test cache key generation with large payload."""
        large_payload = {
            "prompt": "x" * 10000,  # Large string
            "tokens": list(range(1000)),  # Large list
            "metadata": {f"key_{i}": f"value_{i}" for i in range(100)},  # Large dict
        }

        key = cache_key("large", large_payload)

        # Should still generate a fixed-length hash
        assert key.startswith("large:")
        hash_part = key.split(":", 1)[1]
        assert len(hash_part) == 64  # SHA256 hex digest length

    def test_cache_key_special_characters(self):
        """Test cache key generation with special characters."""
        payload = {
            "prompt": "Hello, 世界! 🌍",
            "unicode": "émojis: 😀🎉",
            "special": "!@#$%^&*(){}[]|\\:;\"'<>?",
        }

        key = cache_key("special", payload)
        assert key.startswith("special:")

    def test_cache_key_empty_payload(self):
        """Test cache key generation with empty payload."""
        payload = {}
        key = cache_key("empty", payload)

        assert key.startswith("empty:")
        assert len(key) > 6  # Should have hash even for empty payload

    def test_cache_key_numeric_values(self):
        """Test cache key generation with various numeric types."""
        payload = {
            "int": 42,
            "float": 3.14159,
            "negative": -123,
            "zero": 0,
            "scientific": 1e-10,
        }

        key = cache_key("numeric", payload)
        assert key.startswith("numeric:")

    def test_cache_key_boolean_values(self):
        """Test cache key generation with boolean values."""
        payload = {"enabled": True, "disabled": False, "optional": None}

        key = cache_key("boolean", payload)
        assert key.startswith("boolean:")