"""
Request batching system for improved throughput.
Collects multiple inference requests and processes them together.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request within a batch."""
    prompt: str
    max_tokens: int
    temperature: float
    request_id: str
    future: asyncio.Future


@dataclass
class BatchResult:
    """Result of a batched inference."""
    response: str
    tokens_used: int
    model_name: str
    processing_time_ms: int


class RequestBatcher:
    """
    Batches inference requests for improved throughput.

    Collects requests over a time window and processes them together,
    which is more efficient for GPU inference.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,  # 100ms
        enable_batching: bool = True
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.enable_batching = enable_batching

        self.pending_requests: List[BatchRequest] = []
        self.batch_lock = asyncio.Lock()
        self.processing = False

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_time = 0.0

        logger.info(f"RequestBatcher initialized: max_batch_size={max_batch_size}, "
                   f"max_wait_time={max_wait_time}s, enabled={enable_batching}")

    async def add_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str
    ) -> BatchResult:
        """
        Add a request to the batch queue and wait for result.

        If batching is disabled, processes immediately.
        """
        if not self.enable_batching:
            # Process immediately without batching
            from .models.llm import generate_text
            start_time = time.time()
            response, tokens, model_name = await generate_text(prompt, max_tokens, temperature)
            processing_time = int((time.time() - start_time) * 1000)

            return BatchResult(
                response=response,
                tokens_used=tokens,
                model_name=model_name,
                processing_time_ms=processing_time
            )

        # Create future for this request
        future = asyncio.Future()

        request = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            future=future
        )

        async with self.batch_lock:
            self.pending_requests.append(request)
            self.total_requests += 1

            # Start batch processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batch_after_delay())

        # Wait for result
        return await future

    async def _process_batch_after_delay(self):
        """Wait for max_wait_time or until batch is full, then process."""
        self.processing = True

        try:
            # Wait for either timeout or batch to fill up
            start_time = time.time()

            while True:
                async with self.batch_lock:
                    if len(self.pending_requests) >= self.max_batch_size:
                        break

                elapsed = time.time() - start_time
                if elapsed >= self.max_wait_time:
                    break

                # Sleep briefly before checking again
                await asyncio.sleep(0.01)

            # Process the batch
            await self._process_current_batch()

        finally:
            self.processing = False

    async def _process_current_batch(self):
        """Process all currently pending requests as a batch."""
        async with self.batch_lock:
            if not self.pending_requests:
                return

            batch = self.pending_requests.copy()
            self.pending_requests.clear()

        logger.info(f"Processing batch of {len(batch)} requests")
        batch_start_time = time.time()

        try:
            # Group requests by similar parameters for optimal batching
            grouped_requests = self._group_similar_requests(batch)

            # Process each group
            for group in grouped_requests:
                await self._process_request_group(group)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set error for all requests in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

        finally:
            batch_time = time.time() - batch_start_time
            self.total_batches += 1
            self.total_batch_time += batch_time

            logger.info(f"Batch completed in {batch_time:.3f}s, "
                       f"avg_batch_time={self.total_batch_time/self.total_batches:.3f}s")

    def _group_similar_requests(self, requests: List[BatchRequest]) -> List[List[BatchRequest]]:
        """Group requests with similar parameters for efficient batching."""
        # For now, group by temperature and max_tokens
        groups = defaultdict(list)

        for request in requests:
            key = (request.temperature, request.max_tokens)
            groups[key].append(request)

        return list(groups.values())

    async def _process_request_group(self, requests: List[BatchRequest]):
        """Process a group of similar requests together."""
        if len(requests) == 1:
            # Single request - process normally
            request = requests[0]
            try:
                from .models.llm import generate_text
                start_time = time.time()
                response, tokens, model_name = await generate_text(
                    request.prompt, request.max_tokens, request.temperature
                )
                processing_time = int((time.time() - start_time) * 1000)

                result = BatchResult(
                    response=response,
                    tokens_used=tokens,
                    model_name=model_name,
                    processing_time_ms=processing_time
                )
                request.future.set_result(result)

            except Exception as e:
                request.future.set_exception(e)

        else:
            # Multiple requests - use batch generation
            await self._process_batch_generation(requests)

    async def _process_batch_generation(self, requests: List[BatchRequest]):
        """Process multiple requests using batch generation."""
        # For now, process them sequentially but with shared model state
        # In a full implementation, this would use actual batch inference

        from .models.llm import generate_text
        start_time = time.time()

        tasks = []
        for request in requests:
            task = asyncio.create_task(
                self._generate_single_in_batch(request, start_time)
            )
            tasks.append(task)

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_single_in_batch(self, request: BatchRequest, batch_start_time: float):
        """Generate response for a single request within a batch."""
        try:
            from .models.llm import generate_text
            response, tokens, model_name = await generate_text(
                request.prompt, request.max_tokens, request.temperature
            )

            # Use batch start time for consistent timing
            processing_time = int((time.time() - batch_start_time) * 1000)

            result = BatchResult(
                response=response,
                tokens_used=tokens,
                model_name=model_name,
                processing_time_ms=processing_time
            )
            request.future.set_result(result)

        except Exception as e:
            request.future.set_exception(e)

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        avg_batch_time = (
            self.total_batch_time / self.total_batches
            if self.total_batches > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_requests_per_batch": (
                self.total_requests / self.total_batches
                if self.total_batches > 0 else 0
            ),
            "avg_batch_time_ms": avg_batch_time * 1000,
            "pending_requests": len(self.pending_requests),
            "batching_enabled": self.enable_batching
        }


# Global batcher instance
_batcher = None


def get_batcher() -> RequestBatcher:
    """Get the global request batcher instance."""
    global _batcher
    if _batcher is None:
        import os

        max_batch_size = int(os.getenv("BATCH_SIZE", "4"))
        max_wait_time = float(os.getenv("BATCH_WAIT_TIME", "0.1"))
        enable_batching = os.getenv("ENABLE_BATCHING", "true").lower() == "true"

        _batcher = RequestBatcher(
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            enable_batching=enable_batching
        )

    return _batcher