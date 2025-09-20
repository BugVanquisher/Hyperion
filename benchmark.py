#!/usr/bin/env python3
"""
Hyperion Performance Benchmarking Suite
========================================

Tests throughput, latency, and scalability of the Hyperion inference platform.
"""

import asyncio
import aiohttp
import time
import statistics
import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 10
    requests_per_user: int = 20
    test_duration: int = 60  # seconds
    prompts: List[str] = None
    max_tokens: int = 50
    temperature: float = 0.7


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    latencies: List[float]
    p50_latency: float
    p95_latency: float
    p99_latency: float
    cache_hit_rate: float
    errors: List[str]


class HyperionBenchmark:
    """Benchmark suite for Hyperion ML platform."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def health_check(self) -> bool:
        """Check if Hyperion service is healthy."""
        try:
            async with self.session.get(f"{self.config.base_url}/healthz") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
                return False
        except Exception:
            return False

    async def single_inference_request(self, prompt: str) -> Dict[str, Any]:
        """Make a single inference request and measure latency."""
        start_time = time.time()

        payload = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        try:
            async with self.session.post(
                f"{self.config.base_url}/v1/llm/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                latency = end_time - start_time

                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "latency": latency,
                        "response": data,
                        "cached": data.get("cached", False),
                        "tokens": data.get("tokens_used", 0),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "cached": False,
                        "tokens": 0,
                        "error": f"HTTP {response.status}"
                    }

        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "latency": end_time - start_time,
                "response": None,
                "cached": False,
                "tokens": 0,
                "error": str(e)
            }

    async def throughput_test(self) -> BenchmarkResult:
        """Test maximum throughput with concurrent requests."""
        print(f"🚀 Running throughput test: {self.config.concurrent_users} users, "
              f"{self.config.requests_per_user} requests each")

        prompts = self.config.prompts or [
            "What is machine learning?",
            "Explain artificial intelligence.",
            "How do neural networks work?",
            "What is deep learning?",
            "Tell me about transformers."
        ]

        tasks = []
        start_time = time.time()

        # Create concurrent tasks
        for user_id in range(self.config.concurrent_users):
            for req_id in range(self.config.requests_per_user):
                prompt = prompts[req_id % len(prompts)]
                task = asyncio.create_task(self.single_inference_request(prompt))
                tasks.append(task)

        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Process results
        successful_requests = 0
        failed_requests = 0
        latencies = []
        cache_hits = 0
        errors = []

        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
                errors.append(str(result))
            elif result["success"]:
                successful_requests += 1
                latencies.append(result["latency"])
                if result["cached"]:
                    cache_hits += 1
            else:
                failed_requests += 1
                errors.append(result["error"])

        total_time = end_time - start_time
        total_requests = len(tasks)

        cache_hit_rate = cache_hits / successful_requests if successful_requests > 0 else 0

        return BenchmarkResult(
            test_name="Throughput Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=successful_requests / total_time,
            latencies=latencies,
            p50_latency=statistics.median(latencies) if latencies else 0,
            p95_latency=np.percentile(latencies, 95) if latencies else 0,
            p99_latency=np.percentile(latencies, 99) if latencies else 0,
            cache_hit_rate=cache_hit_rate,
            errors=errors[:10]  # Limit error list
        )

    async def latency_test(self) -> BenchmarkResult:
        """Test latency with sequential requests (no concurrency)."""
        print("⏱️  Running latency test: sequential requests")

        prompts = self.config.prompts or ["What is the meaning of life?"]
        latencies = []
        successful_requests = 0
        failed_requests = 0
        cache_hits = 0
        errors = []

        start_time = time.time()

        for i in range(50):  # 50 sequential requests
            prompt = prompts[i % len(prompts)]
            result = await self.single_inference_request(prompt)

            if result["success"]:
                successful_requests += 1
                latencies.append(result["latency"])
                if result["cached"]:
                    cache_hits += 1
            else:
                failed_requests += 1
                errors.append(result["error"])

        end_time = time.time()
        total_time = end_time - start_time
        cache_hit_rate = cache_hits / successful_requests if successful_requests > 0 else 0

        return BenchmarkResult(
            test_name="Latency Test",
            total_requests=50,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=successful_requests / total_time,
            latencies=latencies,
            p50_latency=statistics.median(latencies) if latencies else 0,
            p95_latency=np.percentile(latencies, 95) if latencies else 0,
            p99_latency=np.percentile(latencies, 99) if latencies else 0,
            cache_hit_rate=cache_hit_rate,
            errors=errors
        )

    async def stress_test(self) -> BenchmarkResult:
        """Stress test with increasing load."""
        print("💪 Running stress test: gradually increasing load")

        all_latencies = []
        successful_requests = 0
        failed_requests = 0
        errors = []
        cache_hits = 0

        start_time = time.time()

        # Gradually increase concurrent users
        for users in [1, 5, 10, 20, 50]:
            print(f"  Testing with {users} concurrent users...")

            tasks = []
            for _ in range(users * 5):  # 5 requests per user
                task = asyncio.create_task(
                    self.single_inference_request("Stress test prompt")
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    failed_requests += 1
                    errors.append(str(result))
                elif result["success"]:
                    successful_requests += 1
                    all_latencies.append(result["latency"])
                    if result["cached"]:
                        cache_hits += 1
                else:
                    failed_requests += 1
                    errors.append(result["error"])

        end_time = time.time()
        total_time = end_time - start_time
        total_requests = successful_requests + failed_requests
        cache_hit_rate = cache_hits / successful_requests if successful_requests > 0 else 0

        return BenchmarkResult(
            test_name="Stress Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=successful_requests / total_time,
            latencies=all_latencies,
            p50_latency=statistics.median(all_latencies) if all_latencies else 0,
            p95_latency=np.percentile(all_latencies, 95) if all_latencies else 0,
            p99_latency=np.percentile(all_latencies, 99) if all_latencies else 0,
            cache_hit_rate=cache_hit_rate,
            errors=errors[:20]
        )

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            # Get batch stats
            async with self.session.get(f"{self.config.base_url}/v1/batch/stats") as response:
                if response.status == 200:
                    batch_stats = await response.json()
                else:
                    batch_stats = {}

            # Get health info
            async with self.session.get(f"{self.config.base_url}/healthz") as response:
                if response.status == 200:
                    health_info = await response.json()
                else:
                    health_info = {}

            return {
                "batch_stats": batch_stats,
                "health_info": health_info
            }
        except Exception as e:
            return {"error": str(e)}

    def print_results(self, result: BenchmarkResult):
        """Print benchmark results in a formatted way."""
        print(f"\n📊 {result.test_name} Results")
        print("=" * 50)
        print(f"Total Requests:      {result.total_requests}")
        print(f"Successful:          {result.successful_requests}")
        print(f"Failed:              {result.failed_requests}")
        print(f"Success Rate:        {result.successful_requests/result.total_requests*100:.1f}%")
        print(f"Total Time:          {result.total_time:.2f}s")
        print(f"Throughput:          {result.requests_per_second:.2f} req/s")
        print(f"Cache Hit Rate:      {result.cache_hit_rate*100:.1f}%")
        print(f"\nLatency Percentiles:")
        print(f"  P50 (median):      {result.p50_latency*1000:.0f}ms")
        print(f"  P95:               {result.p95_latency*1000:.0f}ms")
        print(f"  P99:               {result.p99_latency*1000:.0f}ms")

        if result.errors:
            print(f"\nSample Errors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")

    def save_results(self, results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        data = []
        for result in results:
            data.append({
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "total_time": result.total_time,
                "requests_per_second": result.requests_per_second,
                "p50_latency_ms": result.p50_latency * 1000,
                "p95_latency_ms": result.p95_latency * 1000,
                "p99_latency_ms": result.p99_latency * 1000,
                "cache_hit_rate": result.cache_hit_rate,
                "error_count": len(result.errors)
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Results saved to {filename}")

    async def run_all_tests(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        print("🧪 Starting Hyperion Benchmark Suite")
        print("=" * 50)

        # Health check first
        if not await self.health_check():
            print("❌ Hyperion service is not healthy!")
            sys.exit(1)

        print("✅ Hyperion service is healthy")

        # Get system info
        system_stats = await self.get_system_stats()
        print(f"📋 System Info: {system_stats.get('health_info', {}).get('device_info', {})}")

        results = []

        # Run tests
        results.append(await self.latency_test())
        results.append(await self.throughput_test())
        results.append(await self.stress_test())

        # Print all results
        for result in results:
            self.print_results(result)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Hyperion Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Hyperion service URL")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users for throughput test")
    parser.add_argument("--requests", type=int, default=20, help="Requests per user")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")

    args = parser.parse_args()

    config = BenchmarkConfig(
        base_url=args.url,
        concurrent_users=args.users,
        requests_per_user=args.requests
    )

    async with HyperionBenchmark(config) as benchmark:
        results = await benchmark.run_all_tests()
        benchmark.save_results(results, args.output)

        print("\n🎉 Benchmark completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)