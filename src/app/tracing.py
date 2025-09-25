"""
OpenTelemetry tracing configuration for Hyperion.

This module sets up distributed tracing with Jaeger backend for end-to-end
request tracking across model inference, batching, and cache operations.
"""

import os
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# Global tracer instance
tracer = None
_initialized = False

def init_tracing():
    """Initialize OpenTelemetry tracing with Jaeger exporter."""
    global tracer, _initialized

    if _initialized:
        logger.info("Tracing already initialized")
        return tracer

    try:
        # Service configuration
        service_name = os.getenv("OTEL_SERVICE_NAME", "hyperion-app")
        jaeger_endpoint = os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT", "http://localhost:14268/api/traces")

        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        })

        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()

        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            collector_endpoint=jaeger_endpoint,
        )

        # Add batch span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Get tracer instance
        tracer = trace.get_tracer(__name__)

        logger.info(f"Tracing initialized for service: {service_name}")
        logger.info(f"Jaeger endpoint: {jaeger_endpoint}")

        _initialized = True
        return tracer

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {str(e)}")
        # Return no-op tracer to prevent application failure
        tracer = trace.NoOpTracer()
        return tracer

def instrument_app(app):
    """Instrument FastAPI application with OpenTelemetry."""
    try:
        # Initialize tracing
        init_tracing()

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        # Instrument Redis
        RedisInstrumentor().instrument()

        # Instrument requests (for external HTTP calls)
        RequestsInstrumentor().instrument()

        logger.info("FastAPI application instrumented with OpenTelemetry")

    except Exception as e:
        logger.error(f"Failed to instrument application: {str(e)}")

@contextmanager
def trace_operation(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for creating custom spans.

    Args:
        name: Name of the operation being traced
        attributes: Additional attributes to add to the span
    """
    if not tracer:
        init_tracing()

    with tracer.start_as_current_span(name) as span:
        try:
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            yield span

        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

def add_span_attributes(span, attributes: Dict[str, Any]):
    """Add attributes to the current span."""
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)

def get_current_span():
    """Get the current active span."""
    return trace.get_current_span()

def get_trace_id():
    """Get the current trace ID as a string."""
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        return format(current_span.get_span_context().trace_id, '032x')
    return None

def add_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add an event to the current span."""
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})