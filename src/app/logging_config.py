"""
Structured logging configuration for Hyperion.

Provides JSON structured logging with ML-specific fields for better
log aggregation and analysis in ELK stack.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class MLJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for ML inference logs.

    Adds ML-specific fields and structured data for ELK processing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with ML-specific fields."""

        # Base log structure
        log_data = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
            "service": "hyperion",
            "component": "ml-inference",
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add custom fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add ML-specific context if available
        ml_context = self._extract_ml_context(record)
        if ml_context:
            log_data["ml"] = ml_context

        return json.dumps(log_data, default=str)

    def _extract_ml_context(
        self, record: logging.LogRecord
    ) -> Optional[Dict[str, Any]]:
        """Extract ML-specific context from log record."""
        ml_context = {}
        message = record.getMessage().lower()

        # GPU context
        if "gpu" in message or "cuda" in message:
            ml_context["device_type"] = "gpu"
            if "memory" in message:
                ml_context["metric_type"] = "memory"
        elif "cpu" in message:
            ml_context["device_type"] = "cpu"

        # Batch context
        if "batch" in message:
            ml_context["operation"] = "batching"
            if "processing" in message:
                ml_context["phase"] = "processing"
            elif "completed" in message:
                ml_context["phase"] = "completed"

        # Model context
        if "model" in message:
            ml_context["operation"] = "inference"
            if "loaded" in message:
                ml_context["phase"] = "initialization"
            elif "inference" in message:
                ml_context["phase"] = "prediction"

        # Cache context
        if "cache" in message:
            ml_context["operation"] = "caching"
            if "hit" in message:
                ml_context["cache_result"] = "hit"
            elif "miss" in message:
                ml_context["cache_result"] = "miss"

        return ml_context if ml_context else None


class MLContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds ML-specific context to log records.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add ML context."""
        # Get extra fields from kwargs
        extra = kwargs.get("extra", {})

        # Add ML context from adapter
        if self.extra:
            extra.update(self.extra)

        kwargs["extra"] = extra
        return msg, kwargs


def setup_structured_logging(
    service_name: str = "hyperion-app",
    log_level: str = "INFO",
    enable_json: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for the application.

    Args:
        service_name: Name of the service for logging context
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting

    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)

    if enable_json:
        # Use JSON formatter for structured logging
        formatter = MLJSONFormatter()
    else:
        # Use standard formatter for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set logging level for key libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


def get_ml_logger(name: str, **context) -> MLContextAdapter:
    """
    Get an ML-aware logger with context.

    Args:
        name: Logger name
        **context: Additional context to add to all log messages

    Returns:
        ML context-aware logger adapter
    """
    logger = logging.getLogger(name)
    return MLContextAdapter(logger, context)


# Utility functions for common ML logging patterns
def log_gpu_metrics(logger: logging.Logger, device_info: Dict[str, Any]):
    """Log GPU metrics in a structured way."""
    logger.info(
        "GPU metrics recorded",
        extra={
            "extra_fields": {
                "ml": {
                    "device_type": "gpu",
                    "operation": "monitoring",
                    "gpu_name": device_info.get("gpu_name"),
                    "memory_allocated": device_info.get("gpu_memory_allocated"),
                    "memory_reserved": device_info.get("gpu_memory_reserved"),
                    "memory_total": device_info.get("gpu_memory_total"),
                }
            }
        },
    )


def log_batch_metrics(
    logger: logging.Logger, batch_size: int, duration: float, avg_time: float
):
    """Log batch processing metrics in a structured way."""
    logger.info(
        f"Batch processing completed: {batch_size} requests in {duration:.3f}s",
        extra={
            "extra_fields": {
                "ml": {
                    "operation": "batching",
                    "phase": "completed",
                    "batch_size": batch_size,
                    "duration_seconds": duration,
                    "avg_time_seconds": avg_time,
                }
            }
        },
    )


def log_inference_metrics(
    logger: logging.Logger, model_name: str, tokens: int, time_ms: int
):
    """Log model inference metrics in a structured way."""
    logger.info(
        f"Model inference completed: {tokens} tokens in {time_ms}ms",
        extra={
            "extra_fields": {
                "ml": {
                    "operation": "inference",
                    "phase": "completed",
                    "model_name": model_name,
                    "tokens_generated": tokens,
                    "processing_time_ms": time_ms,
                }
            }
        },
    )


def log_cache_operation(
    logger: logging.Logger, operation: str, key: str, hit: bool = None
):
    """Log cache operations in a structured way."""
    extra_data = {
        "ml": {
            "operation": "caching",
            "cache_operation": operation,
            "cache_key_preview": key[:50] + "..." if len(key) > 50 else key,
        }
    }

    if hit is not None:
        extra_data["ml"]["cache_result"] = "hit" if hit else "miss"

    logger.info(
        f"Cache {operation}: {'hit' if hit else 'miss' if hit is not None else 'operation'}",
        extra={"extra_fields": extra_data},
    )
