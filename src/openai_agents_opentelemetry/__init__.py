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

Configuration:
    from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

    # Configure content capture and filtering
    config = ProcessorConfig(
        capture_prompts=True,
        capture_completions=True,
        max_attribute_length=1024,
    )
    processor = OpenTelemetryTracingProcessor(config=config)

Resource Helper:
    from openai_agents_opentelemetry import create_resource
    from opentelemetry.sdk.trace import TracerProvider

    resource = create_resource(
        service_name="my-agent-service",
        service_version="1.0.0",
    )
    provider = TracerProvider(resource=resource)

Metrics with Recommended Bucket Boundaries:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from openai_agents_opentelemetry import (
        OpenTelemetryTracingProcessor,
        create_metrics_views,
    )

    # Create meter provider with recommended views for GenAI metrics
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    views = create_metrics_views()
    meter_provider = MeterProvider(metric_readers=[reader], views=views)
    metrics.set_meter_provider(meter_provider)

    # Create processor with metrics enabled
    processor = OpenTelemetryTracingProcessor(enable_metrics=True)

Requirements:
    pip install openai-agents-opentelemetry
"""

from ._version import __version__
from .opentelemetry_processor import (
    DURATION_BUCKETS,
    TOKEN_BUCKETS,
    ContentFilter,
    OpenTelemetryTracingProcessor,
    ProcessorConfig,
    create_metrics_views,
    create_resource,
)

__all__ = [
    "OpenTelemetryTracingProcessor",
    "ProcessorConfig",
    "ContentFilter",
    "create_resource",
    "create_metrics_views",
    "TOKEN_BUCKETS",
    "DURATION_BUCKETS",
    "__version__",
]
