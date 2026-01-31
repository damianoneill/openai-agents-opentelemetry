"""Integration tests using real OpenTelemetry SDK with in-memory exporters.

These tests verify the processor works correctly with the actual OTel SDK,
not just mocks. They use InMemorySpanExporter and InMemoryMetricReader to
capture and assert on real telemetry data.

Note: OTel SDK global providers can only be set once per process, so we use
module-scoped fixtures and clear exporters between tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry import baggage, context, metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openai_agents_opentelemetry import (
    DURATION_BUCKETS,
    TOKEN_BUCKETS,
    OpenTelemetryTracingProcessor,
    ProcessorConfig,
    create_metrics_views,
    create_resource,
)

# =============================================================================
# Mock Objects - Complete implementations matching SDK interfaces
# =============================================================================


class MockTrace:
    """Mock SDK Trace object."""

    def __init__(
        self,
        trace_id: str = "test-trace-123",
        name: str = "Test Trace",
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.trace_id = trace_id
        self.name = name
        self.group_id = group_id
        self.metadata = metadata


class MockSpan:
    """Mock SDK Span object."""

    def __init__(
        self,
        span_id: str = "span-1",
        trace_id: str = "trace-1",
        parent_id: str | None = None,
        span_data: Any = None,
        error: dict[str, Any] | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ):
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.span_data = span_data
        self.error = error
        self.started_at = started_at
        self.ended_at = ended_at


class MockAgentSpanData:
    """Mock AgentSpanData with all required attributes."""

    type = "agent"
    export_friendly_name = "agent"

    def __init__(
        self,
        name: str = "TestAgent",
        tools: list[str] | None = None,
        handoffs: list[str] | None = None,
        output_type: str | None = None,
    ):
        self.name = name
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.output_type = output_type


class MockGenerationSpanData:
    """Mock GenerationSpanData with all required attributes."""

    type = "generation"
    export_friendly_name = "generation"

    def __init__(
        self,
        model: str = "gpt-4",
        model_config: dict[str, Any] | None = None,
        input: list[dict[str, Any]] | None = None,
        output: list[Any] | None = None,
        usage: dict[str, Any] | None = None,
    ):
        self.model = model
        self.model_config = model_config or {}
        self.input = input
        self.output = output
        self.usage = usage


class MockFunctionSpanData:
    """Mock FunctionSpanData with all required attributes."""

    type = "function"
    export_friendly_name = "function"

    def __init__(
        self,
        name: str = "test_function",
        input: str | dict[str, Any] | list[Any] | None = None,
        output: str | dict[str, Any] | list[Any] | None = None,
        mcp_data: dict[str, Any] | None = None,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.mcp_data = mcp_data


class MockHandoffSpanData:
    """Mock HandoffSpanData with all required attributes."""

    type = "handoff"
    export_friendly_name = "handoff"

    def __init__(self, from_agent: Any = None, to_agent: Any = None):
        self.from_agent = from_agent
        self.to_agent = to_agent


class MockGuardrailSpanData:
    """Mock GuardrailSpanData with all required attributes."""

    type = "guardrail"
    export_friendly_name = "guardrail"

    def __init__(self, name: str = "test_guardrail", triggered: bool = False):
        self.name = name
        self.triggered = triggered


# =============================================================================
# Module-scoped Fixtures - OTel SDK only allows one provider per process
# =============================================================================


@pytest.fixture(scope="module")
def otel_span_exporter():
    """Module-scoped in-memory span exporter."""
    exporter = InMemorySpanExporter()

    # Create and configure tracer provider
    tracer_provider = TracerProvider(
        resource=create_resource(
            service_name="test-service",
            service_version="1.0.0",
        )
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Set as global provider (can only be done once per process)
    trace.set_tracer_provider(tracer_provider)

    yield exporter

    # Cleanup at end of module
    exporter.clear()
    tracer_provider.shutdown()


@pytest.fixture(scope="module")
def otel_metric_reader():
    """Module-scoped in-memory metric reader."""
    reader = InMemoryMetricReader()

    # Create and configure meter provider
    meter_provider = MeterProvider(metric_readers=[reader])

    # Set as global provider (can only be done once per process)
    metrics.set_meter_provider(meter_provider)

    yield reader

    # Cleanup at end of module
    meter_provider.shutdown()


@pytest.fixture
def span_exporter(otel_span_exporter):
    """Per-test fixture that clears spans before each test."""
    otel_span_exporter.clear()
    return otel_span_exporter


@pytest.fixture
def metric_reader(otel_metric_reader):
    """Per-test fixture for metrics."""
    return otel_metric_reader


# =============================================================================
# Span Integration Tests
# =============================================================================


class TestIntegrationSpans:
    """Integration tests for span creation with real OTel SDK."""

    def test_trace_creates_real_span(self, span_exporter):
        """Test that starting a trace creates a real OTel span."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-1", name="Integration Test")

        processor.on_trace_start(mock_trace)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "workflow: Integration Test"

    def test_agent_span_attributes(self, span_exporter):
        """Test that agent spans have correct attributes."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-2")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-2",
            span_data=MockAgentSpanData(name="MyAgent", tools=["tool1", "tool2"]),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        # Find the agent span (not the trace span)
        agent_spans = [s for s in spans if s.name == "agent: MyAgent"]
        assert len(agent_spans) == 1

        agent_span = agent_spans[0]
        assert agent_span.attributes.get("agent.name") == "MyAgent"
        assert "tool1" in agent_span.attributes.get("agent.tools", "")
        assert "tool2" in agent_span.attributes.get("agent.tools", "")

    def test_generation_span_with_otel_conventions(self, span_exporter):
        """Test that generation spans follow OTel GenAI conventions."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-3")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-3",
            span_data=MockGenerationSpanData(
                model="gpt-4-turbo",
                model_config={"temperature": 0.7},
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        gen_spans = [s for s in spans if s.name == "chat gpt-4-turbo"]
        assert len(gen_spans) == 1

        gen_span = gen_spans[0]
        assert gen_span.attributes.get("gen_ai.operation.name") == "chat"
        assert gen_span.attributes.get("gen_ai.provider.name") == "openai"
        assert gen_span.attributes.get("gen_ai.request.model") == "gpt-4-turbo"
        assert gen_span.attributes.get("gen_ai.request.temperature") == 0.7
        assert gen_span.attributes.get("gen_ai.usage.input_tokens") == 100
        assert gen_span.attributes.get("gen_ai.usage.output_tokens") == 50

    def test_function_span_with_tool_attributes(self, span_exporter):
        """Test that function spans have correct tool attributes."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-4")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-4",
            span_data=MockFunctionSpanData(
                name="get_weather",
                input='{"city": "London"}',
                output='{"temp": 20}',
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        func_spans = [s for s in spans if s.name == "execute_tool get_weather"]
        assert len(func_spans) == 1

        func_span = func_spans[0]
        assert func_span.attributes.get("gen_ai.tool.name") == "get_weather"
        assert func_span.attributes.get("gen_ai.tool.call.arguments") == '{"city": "London"}'
        assert func_span.attributes.get("gen_ai.tool.call.result") == '{"temp": 20}'

    def test_parent_child_relationships(self, span_exporter):
        """Test that parent-child span relationships are preserved."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-5")
        parent_span = MockSpan(
            span_id="parent-span",
            trace_id="trace-5",
            span_data=MockAgentSpanData(name="ParentAgent"),
        )
        child_span = MockSpan(
            span_id="child-span",
            trace_id="trace-5",
            parent_id="parent-span",
            span_data=MockGenerationSpanData(model="gpt-4"),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(parent_span)
        processor.on_span_start(child_span)
        processor.on_span_end(child_span)
        processor.on_span_end(parent_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 3  # trace + agent + generation

        # Find spans by name
        agent_span = next(s for s in spans if "ParentAgent" in s.name)
        gen_span = next(s for s in spans if "gpt-4" in s.name)

        # Verify parent-child relationship
        assert gen_span.parent is not None
        assert gen_span.parent.span_id == agent_span.context.span_id

    def test_error_handling_sets_status(self, span_exporter):
        """Test that errors are properly recorded on spans."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-6")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-6",
            span_data=MockGenerationSpanData(model="gpt-4"),
            error={"message": "API rate limit exceeded", "type": "RateLimitError"},
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        gen_span = next(s for s in spans if "gpt-4" in s.name)

        # Check error status
        from opentelemetry.trace import StatusCode

        assert gen_span.status.status_code == StatusCode.ERROR
        assert "API rate limit exceeded" in gen_span.attributes.get("error.message", "")


# =============================================================================
# Content Capture Integration Tests
# =============================================================================


class TestIntegrationContentCapture:
    """Integration tests for content capture with span events."""

    def test_prompt_capture_creates_event(self, span_exporter):
        """Test that prompts are captured as span events."""
        config = ProcessorConfig(capture_prompts=True)
        processor = OpenTelemetryTracingProcessor(config=config)

        mock_trace = MockTrace(trace_id="trace-7")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-7",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello, world!"}],
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        gen_span = next(s for s in spans if "gpt-4" in s.name)

        # Check for prompt event
        prompt_events = [e for e in gen_span.events if e.name == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1
        assert "Hello, world!" in str(prompt_events[0].attributes.get("gen_ai.prompt", ""))

    def test_content_filter_redacts_data(self, span_exporter):
        """Test that content filters are applied to captured content."""

        def redact_filter(content: str, context: str) -> str:
            return content.replace("secret123", "[REDACTED]")

        config = ProcessorConfig(capture_prompts=True, content_filter=redact_filter)
        processor = OpenTelemetryTracingProcessor(config=config)

        mock_trace = MockTrace(trace_id="trace-8")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-8",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "My password is secret123"}],
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        gen_span = next(s for s in spans if "gpt-4" in s.name)

        prompt_events = [e for e in gen_span.events if e.name == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1
        prompt_content = str(prompt_events[0].attributes.get("gen_ai.prompt", ""))
        assert "secret123" not in prompt_content
        assert "[REDACTED]" in prompt_content

    def test_disabled_capture_no_events(self, span_exporter):
        """Test that disabled capture settings prevent event creation."""
        config = ProcessorConfig(capture_prompts=False, capture_completions=False)
        processor = OpenTelemetryTracingProcessor(config=config)

        mock_output = MagicMock()
        mock_output.content = "Hi there!"

        mock_trace = MockTrace(trace_id="trace-9")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-9",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
                output=[mock_output],
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        gen_span = next(s for s in spans if "gpt-4" in s.name)

        # Should have no content events
        content_events = [
            e
            for e in gen_span.events
            if e.name in ("gen_ai.content.prompt", "gen_ai.content.completion")
        ]
        assert len(content_events) == 0


# =============================================================================
# Metrics Integration Tests
# =============================================================================


class TestIntegrationMetrics:
    """Integration tests for metrics with real OTel SDK."""

    def test_token_usage_metric_recorded(self, span_exporter, metric_reader):
        """Test that token usage metrics are recorded."""
        processor = OpenTelemetryTracingProcessor(enable_metrics=True)

        mock_trace = MockTrace(trace_id="trace-10")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-10",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                usage={"input_tokens": 150, "output_tokens": 75},
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        # Force metric collection
        metrics_data = metric_reader.get_metrics_data()

        # Find token usage metric
        token_metrics = []
        if metrics_data and metrics_data.resource_metrics:
            for resource_metric in metrics_data.resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        if metric.name == "gen_ai.client.token.usage":
                            token_metrics.append(metric)

        assert len(token_metrics) > 0

    def test_tool_invocation_counter(self, span_exporter, metric_reader):
        """Test that tool invocation counter is incremented."""
        processor = OpenTelemetryTracingProcessor(enable_metrics=True)

        mock_trace = MockTrace(trace_id="trace-11")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-11",
            span_data=MockFunctionSpanData(name="search_web"),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        metrics_data = metric_reader.get_metrics_data()

        # Find tool invocation metric
        tool_metrics = []
        if metrics_data and metrics_data.resource_metrics:
            for resource_metric in metrics_data.resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        if metric.name == "agent.tool.invocations":
                            tool_metrics.append(metric)

        assert len(tool_metrics) > 0

    def test_guardrail_trigger_counter(self, span_exporter, metric_reader):
        """Test that guardrail trigger counter is incremented when triggered."""
        processor = OpenTelemetryTracingProcessor(enable_metrics=True)

        mock_trace = MockTrace(trace_id="trace-12")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-12",
            span_data=MockGuardrailSpanData(name="content_filter", triggered=True),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        metrics_data = metric_reader.get_metrics_data()

        # Find guardrail metric
        guardrail_metrics = []
        if metrics_data and metrics_data.resource_metrics:
            for resource_metric in metrics_data.resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        if metric.name == "agent.guardrail.triggers":
                            guardrail_metrics.append(metric)

        assert len(guardrail_metrics) > 0


# =============================================================================
# Resource Integration Tests
# =============================================================================


class TestIntegrationResource:
    """Integration tests for resource attributes."""

    def test_resource_attributes_on_spans(self, span_exporter):
        """Test that resource attributes are present on exported spans."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-13", name="Resource Test")

        processor.on_trace_start(mock_trace)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1

        # Check resource attributes
        resource = spans[0].resource
        assert resource.attributes.get("service.name") == "test-service"
        assert resource.attributes.get("service.version") == "1.0.0"


# =============================================================================
# Concurrency Integration Tests
# =============================================================================


class TestIntegrationConcurrency:
    """Integration tests for concurrent span handling."""

    def test_concurrent_spans_isolated(self, span_exporter):
        """Test that concurrent spans don't interfere with each other."""
        processor = OpenTelemetryTracingProcessor()

        # Create two concurrent traces with unique IDs
        trace1 = MockTrace(trace_id="trace-14a", name="Trace 1")
        trace2 = MockTrace(trace_id="trace-14b", name="Trace 2")

        span1 = MockSpan(
            span_id="span-1a",
            trace_id="trace-14a",
            span_data=MockAgentSpanData(name="Agent1"),
        )
        span2 = MockSpan(
            span_id="span-2b",
            trace_id="trace-14b",
            span_data=MockAgentSpanData(name="Agent2"),
        )

        # Interleave operations
        processor.on_trace_start(trace1)
        processor.on_trace_start(trace2)
        processor.on_span_start(span1)
        processor.on_span_start(span2)
        processor.on_span_end(span2)
        processor.on_span_end(span1)
        processor.on_trace_end(trace1)
        processor.on_trace_end(trace2)

        spans = span_exporter.get_finished_spans()

        # Should have 4 spans: 2 traces + 2 agents
        assert len(spans) == 4

        # Verify both agents exist
        agent_names = [
            s.attributes.get("agent.name") for s in spans if "agent.name" in s.attributes
        ]
        assert "Agent1" in agent_names
        assert "Agent2" in agent_names


# =============================================================================
# Shutdown Integration Tests
# =============================================================================


class TestIntegrationBaggage:
    """Integration tests for baggage context propagation."""

    def test_baggage_propagates_to_spans(self, span_exporter):
        """Test that baggage values are added as span attributes."""
        config = ProcessorConfig(baggage_keys=["user.id", "session.id"])
        processor = OpenTelemetryTracingProcessor(config=config)

        # Set baggage in the OpenTelemetry context
        ctx = baggage.set_baggage("user.id", "user-123")
        ctx = baggage.set_baggage("session.id", "session-456", context=ctx)

        # Attach the context and run agent operations
        token = context.attach(ctx)
        try:
            mock_trace = MockTrace(trace_id="trace-16", name="Baggage Test")
            mock_span = MockSpan(
                span_id="span-1",
                trace_id="trace-16",
                span_data=MockAgentSpanData(name="BaggageAgent"),
            )

            processor.on_trace_start(mock_trace)
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)
            processor.on_trace_end(mock_trace)
        finally:
            context.detach(token)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "BaggageAgent" in s.name]
        assert len(agent_spans) == 1

        agent_span = agent_spans[0]
        assert agent_span.attributes.get("user.id") == "user-123"
        assert agent_span.attributes.get("session.id") == "session-456"

    def test_baggage_without_config_not_added(self, span_exporter):
        """Test that baggage is not added when not configured."""
        # No baggage_keys configured
        config = ProcessorConfig(baggage_keys=[])
        processor = OpenTelemetryTracingProcessor(config=config)

        # Set baggage in the OpenTelemetry context
        ctx = baggage.set_baggage("user.id", "user-123")
        token = context.attach(ctx)
        try:
            mock_trace = MockTrace(trace_id="trace-17", name="No Baggage Test")
            mock_span = MockSpan(
                span_id="span-1",
                trace_id="trace-17",
                span_data=MockAgentSpanData(name="NoBaggageAgent"),
            )

            processor.on_trace_start(mock_trace)
            processor.on_span_start(mock_span)
            processor.on_span_end(mock_span)
            processor.on_trace_end(mock_trace)
        finally:
            context.detach(token)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "NoBaggageAgent" in s.name]
        assert len(agent_spans) == 1

        agent_span = agent_spans[0]
        # user.id should NOT be in attributes since it wasn't configured
        assert "user.id" not in agent_span.attributes


class TestIntegrationShutdown:
    """Integration tests for processor shutdown."""

    def test_shutdown_flushes_spans(self, span_exporter):
        """Test that shutdown properly flushes pending spans."""
        processor = OpenTelemetryTracingProcessor()

        mock_trace = MockTrace(trace_id="trace-15")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-15",
            span_data=MockAgentSpanData(name="TestAgent"),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)

        # Shutdown without explicitly ending spans
        processor.shutdown()

        # Spans should still be exported (shutdown ends unclosed spans)
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1


class TestIntegrationMetricsViews:
    """Integration tests for create_metrics_views with real OTel SDK."""

    def test_create_metrics_views_with_meter_provider(self, span_exporter):
        """Test that create_metrics_views works with a real MeterProvider."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        # Create views using our helper
        views = create_metrics_views()

        # Create a MeterProvider with the views
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader], views=views)

        # Get a meter and create the histograms
        meter = provider.get_meter("test-meter")

        token_histogram = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Token usage",
        )
        duration_histogram = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Operation duration",
        )

        # Record some values
        token_histogram.record(1000, {"gen_ai.token.type": "input"})
        token_histogram.record(500, {"gen_ai.token.type": "output"})
        duration_histogram.record(1.5, {"gen_ai.operation.name": "chat"})

        # Get metrics and verify they were recorded
        metrics_data = reader.get_metrics_data()
        assert metrics_data is not None

        # Find the token histogram metric
        token_metric = None
        duration_metric = None
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "gen_ai.client.token.usage":
                        token_metric = metric
                    elif metric.name == "gen_ai.client.operation.duration":
                        duration_metric = metric

        assert token_metric is not None, "Token metric should be recorded"
        assert duration_metric is not None, "Duration metric should be recorded"

        # Verify the histogram has the correct bucket boundaries
        # The data points should use our custom buckets
        token_data_points = list(token_metric.data.data_points)
        assert len(token_data_points) == 2  # input and output

        # Verify bucket boundaries match TOKEN_BUCKETS
        for dp in token_data_points:
            assert dp.explicit_bounds == TOKEN_BUCKETS

        duration_data_points = list(duration_metric.data.data_points)
        assert len(duration_data_points) == 1

        # Verify bucket boundaries match DURATION_BUCKETS
        for dp in duration_data_points:
            assert dp.explicit_bounds == DURATION_BUCKETS

        # Cleanup
        provider.shutdown()

    def test_views_have_correct_instrument_names(self):
        """Test that views target the correct instrument names."""
        views = create_metrics_views()

        instrument_names = [v._instrument_name for v in views]
        assert "gen_ai.client.token.usage" in instrument_names
        assert "gen_ai.client.operation.duration" in instrument_names

    def test_views_have_explicit_bucket_aggregation(self):
        """Test that views use ExplicitBucketHistogramAggregation."""
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation

        views = create_metrics_views()

        for view in views:
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)

    def test_token_view_has_correct_boundaries(self):
        """Test that token usage view has OTel GenAI recommended bucket boundaries."""
        views = create_metrics_views()

        token_view = next(v for v in views if v._instrument_name == "gen_ai.client.token.usage")
        assert token_view._aggregation._boundaries == TOKEN_BUCKETS

    def test_duration_view_has_correct_boundaries(self):
        """Test that duration view has OTel GenAI recommended bucket boundaries."""
        views = create_metrics_views()

        duration_view = next(
            v for v in views if v._instrument_name == "gen_ai.client.operation.duration"
        )
        assert duration_view._aggregation._boundaries == DURATION_BUCKETS


class TestIntegrationDictToolInputs:
    """Integration tests for dict/list tool inputs (Issue 1 fix)."""

    def test_function_span_with_dict_input(self, span_exporter):
        """Test that function spans handle dict inputs correctly."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-dict-1")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-dict-1",
            span_data=MockFunctionSpanData(
                name="get_weather",
                input={"city": "London", "units": "celsius"},  # dict, not string!
                output={"temperature": 20, "condition": "sunny"},
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        func_spans = [s for s in spans if s.name == "execute_tool get_weather"]
        assert len(func_spans) == 1

        func_span = func_spans[0]
        # Arguments should be serialized to JSON string
        args = func_span.attributes.get("gen_ai.tool.call.arguments")
        assert isinstance(args, str)
        assert "London" in args
        assert "celsius" in args

        # Output should also be serialized to JSON string
        result = func_span.attributes.get("gen_ai.tool.call.result")
        assert isinstance(result, str)
        assert "20" in result
        assert "sunny" in result

    def test_function_span_with_list_input(self, span_exporter):
        """Test that function spans handle list inputs correctly."""
        processor = OpenTelemetryTracingProcessor()
        mock_trace = MockTrace(trace_id="trace-list-1")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-list-1",
            span_data=MockFunctionSpanData(
                name="search",
                input=["query1", "query2", "query3"],  # list, not string!
            ),
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        spans = span_exporter.get_finished_spans()
        func_spans = [s for s in spans if s.name == "execute_tool search"]
        assert len(func_spans) == 1

        func_span = func_spans[0]
        args = func_span.attributes.get("gen_ai.tool.call.arguments")
        assert isinstance(args, str)
        assert "query1" in args


class TestIntegrationDurationMetrics:
    """Integration tests for duration metrics (Issue 3 fix)."""

    def test_duration_metric_recorded_with_timestamps(self, span_exporter, metric_reader):
        """Test that duration metrics are recorded when timestamps are provided."""
        processor = OpenTelemetryTracingProcessor(enable_metrics=True)

        mock_trace = MockTrace(trace_id="trace-duration-1")
        mock_span = MockSpan(
            span_id="span-1",
            trace_id="trace-duration-1",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            started_at="2024-01-01T10:00:00.000Z",
            ended_at="2024-01-01T10:00:01.500Z",  # 1.5 seconds later
        )

        processor.on_trace_start(mock_trace)
        processor.on_span_start(mock_span)
        processor.on_span_end(mock_span)
        processor.on_trace_end(mock_trace)

        # Force metric collection
        metrics_data = metric_reader.get_metrics_data()

        # Find duration metric
        duration_metrics = []
        if metrics_data and metrics_data.resource_metrics:
            for resource_metric in metrics_data.resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        if metric.name == "gen_ai.client.operation.duration":
                            duration_metrics.append(metric)

        assert len(duration_metrics) > 0, "Duration metric should be recorded"
