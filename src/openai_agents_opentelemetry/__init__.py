"""OpenTelemetry tracing extension for the OpenAI Agents SDK.

This package provides an OpenTelemetry-based TracingProcessor that bridges
the Agents SDK's tracing system to OpenTelemetry, enabling export to any
OTLP-compatible backend (Jaeger, Datadog, Honeycomb, etc.).

Usage:
    from agents import add_trace_processor
    from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

    # Create and register the processor
    otel_processor = OpenTelemetryTracingProcessor()
    add_trace_processor(otel_processor)

    # Or replace the default processor entirely
    from agents import set_trace_processors
    set_trace_processors([otel_processor])

Requirements:
    pip install openai-agents-opentelemetry
"""

from ._version import __version__
from .opentelemetry_processor import OpenTelemetryTracingProcessor

__all__ = ["OpenTelemetryTracingProcessor", "__version__"]
