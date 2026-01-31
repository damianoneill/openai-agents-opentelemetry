"""Tests for OpenTelemetryTracingProcessor."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class MockOTelSpan:
    """Mock OpenTelemetry span for testing."""

    def __init__(self, name: str = "test"):
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.status: Any = None
        self.ended = False
        self.events: list[dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.status = status

    def end(self) -> None:
        self.ended = True

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "attributes": attributes or {}})


class MockTracer:
    """Mock OpenTelemetry tracer for testing."""

    def __init__(self) -> None:
        self.spans: list[MockOTelSpan] = []

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> MockOTelSpan:
        span = MockOTelSpan(name)
        if attributes:
            span.attributes.update(attributes)
        self.spans.append(span)
        return span


class MockTrace:
    """Mock SDK Trace for testing."""

    def __init__(
        self,
        trace_id: str = "trace_abc",
        name: str = "Test",
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.trace_id = trace_id
        self.name = name
        self.group_id = group_id
        self.metadata = metadata


class MockAgentSpanData:
    """Mock agent span data."""

    type = "agent"

    def __init__(
        self,
        name: str = "TestAgent",
        handoffs: list[str] | None = None,
        tools: list[str] | None = None,
        output_type: str | None = None,
    ):
        self.name = name
        self.handoffs = handoffs or []
        self.tools = tools or []
        self.output_type = output_type


class MockGenerationSpanData:
    """Mock generation span data."""

    type = "generation"

    def __init__(
        self,
        model: str | None = None,
        model_config: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        input: list[Any] | None = None,
        output: list[Any] | None = None,
    ):
        self.model = model
        self.model_config = model_config
        self.usage = usage
        self.input = input
        self.output = output


class MockFunctionSpanData:
    """Mock function/tool span data."""

    type = "function"

    def __init__(
        self,
        name: str = "test_tool",
        input: str | None = None,
        output: str | None = None,
        mcp_data: dict[str, Any] | None = None,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.mcp_data = mcp_data


class MockHandoffSpanData:
    """Mock handoff span data."""

    type = "handoff"

    def __init__(
        self,
        from_agent: str | None = None,
        to_agent: str | None = None,
    ):
        self.from_agent = from_agent
        self.to_agent = to_agent


class MockGuardrailSpanData:
    """Mock guardrail span data."""

    type = "guardrail"

    def __init__(
        self,
        name: str = "test_guardrail",
        triggered: bool = False,
    ):
        self.name = name
        self.triggered = triggered


class MockCustomSpanData:
    """Mock custom span data."""

    type = "custom"

    def __init__(
        self,
        name: str = "custom_operation",
        data: dict[str, Any] | None = None,
    ):
        self.name = name
        self.data = data


class MockResponseSpanData:
    """Mock response span data."""

    type = "response"

    def __init__(self, response: Any = None):
        self.response = response


class MockTranscriptionSpanData:
    """Mock transcription span data."""

    type = "transcription"

    def __init__(
        self,
        model: str | None = None,
        output: str | None = None,
    ):
        self.model = model
        self.output = output


class MockSpeechSpanData:
    """Mock speech span data."""

    type = "speech"

    def __init__(self, model: str | None = None):
        self.model = model


class MockSpeechGroupSpanData:
    """Mock speech group span data."""

    type = "speech_group"


class MockMCPToolsSpanData:
    """Mock MCP tools span data."""

    type = "mcp_tools"

    def __init__(
        self,
        server: str | None = None,
        result: list[str] | None = None,
    ):
        self.server = server
        self.result = result


class MockUnknownSpanData:
    """Mock unknown span type for testing fallback."""

    type = "unknown_type"

    def export(self) -> dict[str, Any]:
        return {"type": "unknown_type", "custom_field": "custom_value"}


class MockSDKSpan:
    """Mock SDK Span for testing."""

    def __init__(
        self,
        span_id: str = "span_123",
        trace_id: str = "trace_abc",
        parent_id: str | None = None,
        span_data: Any = None,
        error: dict[str, Any] | None = None,
    ):
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.span_data = span_data or MockAgentSpanData()
        self.error = error


class MockResponse:
    """Mock OpenAI Response object."""

    def __init__(self, id: str = "resp_123", model: str | None = None):
        self.id = id
        self.model = model


@pytest.fixture
def mock_otel() -> Any:
    """Fixture providing mocked OpenTelemetry components."""
    mock_trace = MagicMock()
    mock_tracer = MockTracer()
    mock_trace.get_tracer.return_value = mock_tracer
    mock_context = MagicMock()
    mock_context.attach.return_value = "token"
    mock_span_kind = MagicMock()
    mock_span_kind.INTERNAL = "INTERNAL"
    mock_span_kind.CLIENT = "CLIENT"
    mock_status = MagicMock()
    mock_status_code = MagicMock()
    mock_status_code.OK = "OK"
    mock_status_code.ERROR = "ERROR"
    mock_baggage = MagicMock()
    mock_baggage.get_baggage.return_value = None

    with patch(
        "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry",
        return_value=(
            mock_trace,
            mock_span_kind,
            mock_status,
            mock_status_code,
            mock_context,
            mock_baggage,
        ),
    ):
        yield {
            "trace": mock_trace,
            "tracer": mock_tracer,
            "context": mock_context,
            "baggage": mock_baggage,
        }


class TestInstrumentationScopeVersioning:
    """Tests for instrumentation scope versioning (Phase 1)."""

    def test_tracer_includes_version_and_schema_url(self, mock_otel: Any) -> None:
        """Test that tracer is created with version and schema URL."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor
        from openai_agents_opentelemetry._version import __version__
        from openai_agents_opentelemetry.opentelemetry_processor import SCHEMA_URL

        OpenTelemetryTracingProcessor()

        mock_otel["trace"].get_tracer.assert_called_once_with(
            "openai.agents",
            instrumenting_library_version=__version__,
            schema_url=SCHEMA_URL,
        )

    def test_schema_url_is_valid_format(self) -> None:
        """Test that SCHEMA_URL follows the expected format."""
        from openai_agents_opentelemetry.opentelemetry_processor import SCHEMA_URL

        assert SCHEMA_URL.startswith("https://opentelemetry.io/schemas/")
        # Verify it contains a version number pattern
        import re

        assert re.search(r"\d+\.\d+\.\d+$", SCHEMA_URL), "Schema URL should end with semver"

    def test_version_is_valid_semver(self) -> None:
        """Test that __version__ is a valid semver string."""
        import re

        from openai_agents_opentelemetry._version import __version__

        # Basic semver pattern
        assert re.match(r"^\d+\.\d+\.\d+", __version__), "__version__ should be semver format"


class TestOpenTelemetryTracingProcessor:
    """Tests for the OpenTelemetryTracingProcessor class."""

    def test_on_trace_start_creates_span(self, mock_otel: Any) -> None:
        """Test that on_trace_start creates an OTel span."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123", name="Test Workflow")
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        assert len(mock_otel["tracer"].spans) == 1
        assert "workflow: Test Workflow" in mock_otel["tracer"].spans[0].name

    def test_on_trace_start_with_group_id(self, mock_otel: Any) -> None:
        """Test that group_id is included in span attributes."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123", name="Test", group_id="group_456")
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        span = mock_otel["tracer"].spans[0]
        assert span.attributes["agent.group_id"] == "group_456"

    def test_on_trace_start_with_metadata(self, mock_otel: Any) -> None:
        """Test that metadata is included in span attributes."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(
            trace_id="trace_123",
            name="Test",
            metadata={"user_id": "user_789", "session": "sess_abc"},
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        span = mock_otel["tracer"].spans[0]
        assert span.attributes["agent.metadata.user_id"] == "user_789"
        assert span.attributes["agent.metadata.session"] == "sess_abc"

    def test_on_trace_end_closes_span(self, mock_otel: Any) -> None:
        """Test that on_trace_end closes the OTel span."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_trace_end(trace)  # type: ignore[arg-type]
        assert mock_otel["tracer"].spans[0].ended

    def test_on_trace_end_without_start_logs_warning(self, mock_otel: Any) -> None:
        """Test that ending a trace without starting logs a warning."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="nonexistent_trace")
        # Should not raise, just log warning.
        processor.on_trace_end(trace)  # type: ignore[arg-type]

    def test_custom_tracer_name(self, mock_otel: Any) -> None:
        """Test that custom tracer name is used with version and schema URL."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor
        from openai_agents_opentelemetry._version import __version__
        from openai_agents_opentelemetry.opentelemetry_processor import SCHEMA_URL

        processor = OpenTelemetryTracingProcessor(tracer_name="my.custom.tracer")
        assert processor._tracer_name == "my.custom.tracer"
        mock_otel["trace"].get_tracer.assert_called_with(
            "my.custom.tracer",
            instrumenting_library_version=__version__,
            schema_url=SCHEMA_URL,
        )


class TestAgentSpan:
    """Tests for agent span mapping."""

    def test_agent_span_basic(self, mock_otel: Any) -> None:
        """Test basic agent span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockAgentSpanData(name="MyAgent"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "agent: MyAgent" in otel_span.name
        assert otel_span.attributes["agent.name"] == "MyAgent"

    def test_agent_span_with_tools_and_handoffs(self, mock_otel: Any) -> None:
        """Test agent span with tools and handoffs."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockAgentSpanData(
                name="MyAgent",
                tools=["tool1", "tool2"],
                handoffs=["agent2", "agent3"],
                output_type="str",
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["agent.tools"] == '["tool1", "tool2"]'
        assert otel_span.attributes["agent.handoffs"] == '["agent2", "agent3"]'
        assert otel_span.attributes["agent.output_type"] == "str"


class TestGenerationSpan:
    """Tests for generation span mapping with OTel GenAI semantic conventions."""

    def test_generation_span_attributes(self, mock_otel: Any) -> None:
        """Test generation span has required and recommended OTel GenAI attributes."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(model="gpt-4", usage={"input_tokens": 100}),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]

        # Span name follows OTel convention: "{operation} {model}"
        assert otel_span.name == "chat gpt-4"

        # Required attributes per OTel GenAI semantic conventions
        assert otel_span.attributes["gen_ai.operation.name"] == "chat"
        assert otel_span.attributes["gen_ai.provider.name"] == "openai"

        # Conditionally required / recommended attributes
        assert otel_span.attributes["gen_ai.request.model"] == "gpt-4"
        assert otel_span.attributes["gen_ai.usage.input_tokens"] == 100

    def test_generation_span_filters_empty_model_config(self, mock_otel: Any) -> None:
        """Test that None and empty string values in model_config are filtered out."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                model_config={
                    "temperature": 0.7,
                    "top_p": None,
                    "frequency_penalty": "",
                    "max_tokens": 100,
                    "presence_penalty": None,
                },
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]

        # Valid values should be present.
        assert otel_span.attributes["gen_ai.request.temperature"] == 0.7
        assert otel_span.attributes["gen_ai.request.max_tokens"] == 100

        # None and empty string values should be filtered out.
        assert "gen_ai.request.top_p" not in otel_span.attributes
        assert "gen_ai.request.frequency_penalty" not in otel_span.attributes
        assert "gen_ai.request.presence_penalty" not in otel_span.attributes

    def test_generation_span_with_usage_metrics(self, mock_otel: Any) -> None:
        """Test generation span captures all usage metrics."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["gen_ai.usage.input_tokens"] == 100
        assert otel_span.attributes["gen_ai.usage.output_tokens"] == 50
        assert otel_span.attributes["gen_ai.usage.total_tokens"] == 150

    def test_generation_span_with_input_output(self, mock_otel: Any) -> None:
        """Test generation span captures input/output preview."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi there!"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "gen_ai.input.preview" in otel_span.attributes
        assert "gen_ai.output.preview" in otel_span.attributes


class TestFunctionSpan:
    """Tests for function/tool span mapping."""

    def test_function_span_basic(self, mock_otel: Any) -> None:
        """Test basic function span creation with OTel GenAI semantic conventions."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockFunctionSpanData(name="search_web", input='{"query": "test"}'),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        # Span name follows OTel convention: "execute_tool {tool_name}"
        assert "execute_tool search_web" in otel_span.name
        # Required attribute
        assert otel_span.attributes["gen_ai.operation.name"] == "execute_tool"
        # Recommended attributes
        assert otel_span.attributes["gen_ai.tool.name"] == "search_web"
        assert otel_span.attributes["gen_ai.tool.type"] == "function"
        assert otel_span.attributes["gen_ai.tool.call.arguments"] == '{"query": "test"}'

    def test_function_span_with_output(self, mock_otel: Any) -> None:
        """Test function span captures output on end with OTel GenAI semantic conventions."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockFunctionSpanData(
                name="search_web",
                input='{"query": "test"}',
                output="Search results here",
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        # Output uses OTel GenAI semantic convention attribute
        assert otel_span.attributes["gen_ai.tool.call.result"] == "Search results here"

    def test_function_span_with_mcp_data(self, mock_otel: Any) -> None:
        """Test function span with MCP data."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockFunctionSpanData(
                name="mcp_tool",
                mcp_data={"server": "my_server", "tool": "my_tool"},
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "mcp.tool.data" in otel_span.attributes


class TestHandoffSpan:
    """Tests for handoff span mapping."""

    def test_handoff_span_basic(self, mock_otel: Any) -> None:
        """Test basic handoff span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockHandoffSpanData(from_agent="Agent1", to_agent="Agent2"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "handoff: Agent1 -> Agent2" in otel_span.name
        assert otel_span.attributes["agent.handoff.from"] == "Agent1"
        assert otel_span.attributes["agent.handoff.to"] == "Agent2"

    def test_handoff_span_unknown_agents(self, mock_otel: Any) -> None:
        """Test handoff span with unknown agents defaults."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockHandoffSpanData(from_agent=None, to_agent=None),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "handoff: unknown -> unknown" in otel_span.name


class TestGuardrailSpan:
    """Tests for guardrail span mapping."""

    def test_guardrail_span_not_triggered(self, mock_otel: Any) -> None:
        """Test guardrail span when not triggered."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGuardrailSpanData(name="content_filter", triggered=False),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "guardrail: content_filter" in otel_span.name
        assert otel_span.attributes["agent.guardrail.name"] == "content_filter"
        assert otel_span.attributes["agent.guardrail.triggered"] is False

    def test_guardrail_span_triggered(self, mock_otel: Any) -> None:
        """Test guardrail span when triggered."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGuardrailSpanData(name="pii_filter", triggered=True),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["agent.guardrail.triggered"] is True


class TestCustomSpan:
    """Tests for custom span mapping."""

    def test_custom_span_basic(self, mock_otel: Any) -> None:
        """Test basic custom span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockCustomSpanData(name="my_operation"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "custom: my_operation" in otel_span.name
        assert otel_span.attributes["custom.name"] == "my_operation"

    def test_custom_span_with_data(self, mock_otel: Any) -> None:
        """Test custom span with custom data."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockCustomSpanData(
                name="my_operation",
                data={"key1": "value1", "key2": 42},
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["custom.data.key1"] == "value1"
        assert otel_span.attributes["custom.data.key2"] == 42


class TestResponseSpan:
    """Tests for response span mapping."""

    def test_response_span_with_response(self, mock_otel: Any) -> None:
        """Test response span with response object captures id and model."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockResponseSpanData(
                response=MockResponse(id="resp_abc123", model="gpt-4-0613")
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "gen_ai.response" in otel_span.name
        assert otel_span.attributes["gen_ai.response.id"] == "resp_abc123"
        assert otel_span.attributes["gen_ai.response.model"] == "gpt-4-0613"

    def test_response_span_without_response(self, mock_otel: Any) -> None:
        """Test response span without response object."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockResponseSpanData(response=None),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "gen_ai.response" in otel_span.name
        assert "gen_ai.response.id" not in otel_span.attributes


class TestTranscriptionSpan:
    """Tests for transcription span mapping."""

    def test_transcription_span_basic(self, mock_otel: Any) -> None:
        """Test basic transcription span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockTranscriptionSpanData(model="whisper-1"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "audio.transcription: whisper-1" in otel_span.name
        assert otel_span.attributes["audio.model"] == "whisper-1"

    def test_transcription_span_with_output(self, mock_otel: Any) -> None:
        """Test transcription span captures output."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockTranscriptionSpanData(model="whisper-1", output="Hello world"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["audio.transcription.output"] == "Hello world"


class TestSpeechSpan:
    """Tests for speech span mapping."""

    def test_speech_span_basic(self, mock_otel: Any) -> None:
        """Test basic speech span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockSpeechSpanData(model="tts-1"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "audio.speech: tts-1" in otel_span.name
        assert otel_span.attributes["audio.model"] == "tts-1"

    def test_speech_group_span(self, mock_otel: Any) -> None:
        """Test speech group span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockSpeechGroupSpanData(),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "audio.speech_group" in otel_span.name


class TestMCPToolsSpan:
    """Tests for MCP tools span mapping."""

    def test_mcp_tools_span_basic(self, mock_otel: Any) -> None:
        """Test basic MCP tools span creation."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockMCPToolsSpanData(server="my_mcp_server"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "mcp.list_tools: my_mcp_server" in otel_span.name
        assert otel_span.attributes["mcp.server"] == "my_mcp_server"

    def test_mcp_tools_span_with_result(self, mock_otel: Any) -> None:
        """Test MCP tools span captures result."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockMCPToolsSpanData(
                server="my_mcp_server",
                result=["tool1", "tool2", "tool3"],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["mcp.tools.count"] == 3
        assert "tool1" in otel_span.attributes["mcp.tools.list"]


class TestUnknownSpanType:
    """Tests for unknown span type fallback."""

    def test_unknown_span_type_fallback(self, mock_otel: Any) -> None:
        """Test that unknown span types are handled gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockUnknownSpanData(),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert "agent.unknown_type" in otel_span.name
        assert otel_span.attributes["span.custom_field"] == "custom_value"


class TestErrorHandling:
    """Tests for error handling in spans."""

    def test_span_with_error(self, mock_otel: Any) -> None:
        """Test that span errors are properly recorded."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockAgentSpanData(name="FailingAgent"),
            error={"message": "Something went wrong", "data": {"code": 500}},
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["error.message"] == "Something went wrong"
        assert "500" in otel_span.attributes["error.data"]

    def test_span_with_error_no_data(self, mock_otel: Any) -> None:
        """Test that span errors without data are handled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockAgentSpanData(name="FailingAgent"),
            error={"message": "Error occurred"},
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]
        otel_span = mock_otel["tracer"].spans[1]
        assert otel_span.attributes["error.message"] == "Error occurred"
        assert "error.data" not in otel_span.attributes

    def test_span_end_without_start(self, mock_otel: Any) -> None:
        """Test that ending a span without starting logs a warning."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(span_id="nonexistent_span", trace_id="trace_123")
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        # Should not raise, just log warning.
        processor.on_span_end(span)  # type: ignore[arg-type]

    def test_span_with_non_serializable_error_data(self, mock_otel: Any) -> None:
        """Test that non-serializable error.data doesn't prevent span from ending.

        Previously, json.dumps() on non-serializable data would raise an exception,
        causing span.end() to never be called and leaking the span.
        """
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")

        # Create an object that can't be JSON serialized.
        class NonSerializable:
            def __str__(self) -> str:
                return "NonSerializable()"

        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockAgentSpanData(name="TestAgent"),
            error={
                "message": "Something failed",
                "data": {"obj": NonSerializable()},  # This would fail json.dumps()
            },
        )

        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        # The span should still be ended despite non-serializable error data.
        otel_span = mock_otel["tracer"].spans[1]  # spans[0] is the trace
        assert otel_span.ended, "Span should be ended even with non-serializable error data"
        assert "error.message" in otel_span.attributes
        assert otel_span.attributes["error.message"] == "Something failed"


class TestOverlappingSpans:
    """Tests for overlapping span scenarios."""

    def test_overlapping_spans_maintain_correct_parents(self, mock_otel: Any) -> None:
        """Test that overlapping spans don't break parent-child relationships.

        Scenario: Span A starts, Span B starts, Span A ends, Span B ends.
        Both A and B should correctly reference the trace as parent.
        Previously, this would break due to out-of-order context detaches.
        """
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")

        # Two sibling spans (both children of the trace, not of each other).
        span_a = MockSDKSpan(
            span_id="span_a",
            trace_id="trace_123",
            parent_id=None,  # Parent is the trace root
            span_data=MockFunctionSpanData(name="tool_a"),
        )
        span_b = MockSDKSpan(
            span_id="span_b",
            trace_id="trace_123",
            parent_id=None,  # Parent is the trace root
            span_data=MockFunctionSpanData(name="tool_b"),
        )

        # Start trace and both spans.
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span_a)  # type: ignore[arg-type]
        processor.on_span_start(span_b)  # type: ignore[arg-type]

        # End spans in opposite order (A ends before B).
        processor.on_span_end(span_a)  # type: ignore[arg-type]
        processor.on_span_end(span_b)  # type: ignore[arg-type]
        processor.on_trace_end(trace)  # type: ignore[arg-type]

        # All spans should be ended successfully.
        assert len(mock_otel["tracer"].spans) == 3  # trace + span_a + span_b
        for otel_span in mock_otel["tracer"].spans:
            assert otel_span.ended, f"Span {otel_span.name} was not ended"


class TestParentChildRelationships:
    """Tests for parent-child span relationships."""

    def test_span_with_parent(self, mock_otel: Any) -> None:
        """Test that child spans reference parent correctly."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        parent_span = MockSDKSpan(
            span_id="parent_span",
            trace_id="trace_123",
            span_data=MockAgentSpanData(name="ParentAgent"),
        )
        child_span = MockSDKSpan(
            span_id="child_span",
            trace_id="trace_123",
            parent_id="parent_span",
            span_data=MockFunctionSpanData(name="child_tool"),
        )

        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(parent_span)  # type: ignore[arg-type]
        processor.on_span_start(child_span)  # type: ignore[arg-type]

        # Both spans should be created.
        assert len(mock_otel["tracer"].spans) == 3  # trace + parent + child
        child_otel_span = mock_otel["tracer"].spans[2]
        assert child_otel_span.attributes["agent.parent_span_id"] == "parent_span"


class TestShutdownAndFlush:
    """Tests for shutdown and force_flush methods."""

    def test_shutdown_clears_spans(self, mock_otel: Any) -> None:
        """Test that shutdown clears active spans."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(trace_id="trace_123")

        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]

        # Verify spans are tracked.
        assert len(processor._active_spans) == 1
        assert len(processor._trace_root_spans) == 1

        processor.shutdown()

        # Verify spans are cleared.
        assert len(processor._active_spans) == 0
        assert len(processor._trace_root_spans) == 0

    def test_shutdown_ends_unclosed_spans(self, mock_otel: Any) -> None:
        """Test that shutdown ends any unclosed spans."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(trace_id="trace_123")

        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]

        processor.shutdown()

        # All spans should be ended.
        for otel_span in mock_otel["tracer"].spans:
            assert otel_span.ended

    def test_force_flush(self, mock_otel: Any) -> None:
        """Test that force_flush delegates to tracer provider."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_provider = MagicMock()
        mock_otel["trace"].get_tracer_provider.return_value = mock_provider

        processor = OpenTelemetryTracingProcessor()
        processor.force_flush()

        mock_provider.force_flush.assert_called_once()

    def test_force_flush_no_provider_method(self, mock_otel: Any) -> None:
        """Test that force_flush handles provider without force_flush method."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_provider = MagicMock(spec=[])  # No force_flush method.
        mock_otel["trace"].get_tracer_provider.return_value = mock_provider

        processor = OpenTelemetryTracingProcessor()
        # Should not raise.
        processor.force_flush()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_safe_attribute_value_string(self) -> None:
        """Test _safe_attribute_value with string."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        assert _safe_attribute_value("test") == "test"

    def test_safe_attribute_value_int(self) -> None:
        """Test _safe_attribute_value with int."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        assert _safe_attribute_value(42) == 42

    def test_safe_attribute_value_float(self) -> None:
        """Test _safe_attribute_value with float."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        assert _safe_attribute_value(3.14) == 3.14

    def test_safe_attribute_value_bool(self) -> None:
        """Test _safe_attribute_value with bool."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        assert _safe_attribute_value(True) is True
        assert _safe_attribute_value(False) is False

    def test_safe_attribute_value_none(self) -> None:
        """Test _safe_attribute_value with None."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        assert _safe_attribute_value(None) == ""

    def test_safe_attribute_value_dict(self) -> None:
        """Test _safe_attribute_value with dict (JSON serialization)."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        result = _safe_attribute_value({"a": 1, "b": "test"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": "test"}

    def test_safe_attribute_value_list(self) -> None:
        """Test _safe_attribute_value with list (JSON serialization)."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        result = _safe_attribute_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_safe_attribute_value_non_serializable(self) -> None:
        """Test _safe_attribute_value with non-serializable object."""
        from openai_agents_opentelemetry.opentelemetry_processor import _safe_attribute_value

        class CustomObject:
            def __str__(self) -> str:
                return "CustomObject()"

        result = _safe_attribute_value(CustomObject())
        assert result == "CustomObject()"

    def test_truncate_string_short(self) -> None:
        """Test _truncate_string with short string."""
        from openai_agents_opentelemetry.opentelemetry_processor import _truncate_string

        assert _truncate_string("short", 100) == "short"

    def test_truncate_string_exact(self) -> None:
        """Test _truncate_string with exact length string."""
        from openai_agents_opentelemetry.opentelemetry_processor import _truncate_string

        assert _truncate_string("12345", 5) == "12345"

    def test_truncate_string_long(self) -> None:
        """Test _truncate_string with long string."""
        from openai_agents_opentelemetry.opentelemetry_processor import _truncate_string

        result = _truncate_string("x" * 100, 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_string_default_max(self) -> None:
        """Test _truncate_string with default max length."""
        from openai_agents_opentelemetry.opentelemetry_processor import _truncate_string

        short_string = "x" * 100
        assert _truncate_string(short_string) == short_string

        long_string = "x" * 5000
        result = _truncate_string(long_string)
        assert len(result) == 4096
        assert result.endswith("...")


class TestImportError:
    """Tests for OpenTelemetry import error handling."""

    def test_import_error_message(self) -> None:
        """Test that ImportError has helpful message."""
        with (
            patch.dict("sys.modules", {"opentelemetry": None}),
            patch(
                "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry",
                side_effect=ImportError(
                    "OpenTelemetry packages are required for OpenTelemetryTracingProcessor. "
                    "Install them with: pip install opentelemetry-api opentelemetry-sdk "
                    "or pip install openai-agents[opentelemetry]"
                ),
            ),
        ):
            from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

            with pytest.raises(ImportError) as exc_info:
                OpenTelemetryTracingProcessor()

            assert "opentelemetry-api" in str(exc_info.value)
            assert "openai-agents[opentelemetry]" in str(exc_info.value)

    def test_try_import_opentelemetry_raises_import_error(self) -> None:
        """Test that _try_import_opentelemetry raises ImportError when not installed."""
        import sys

        from openai_agents_opentelemetry.opentelemetry_processor import (
            _try_import_opentelemetry,
        )

        # Save references to real modules
        real_otel = sys.modules.get("opentelemetry")
        real_otel_trace = sys.modules.get("opentelemetry.trace")
        real_otel_context = sys.modules.get("opentelemetry.context")

        try:
            # Remove opentelemetry from sys.modules to simulate it not being installed
            sys.modules["opentelemetry"] = None  # type: ignore[assignment]
            sys.modules["opentelemetry.trace"] = None  # type: ignore[assignment]
            sys.modules["opentelemetry.context"] = None  # type: ignore[assignment]

            with pytest.raises(ImportError) as exc_info:
                _try_import_opentelemetry()

            error_message = str(exc_info.value)
            assert "OpenTelemetry packages are required" in error_message
            assert "opentelemetry-api" in error_message
            assert "opentelemetry-sdk" in error_message
        finally:
            # Restore real modules
            if real_otel is not None:
                sys.modules["opentelemetry"] = real_otel
            if real_otel_trace is not None:
                sys.modules["opentelemetry.trace"] = real_otel_trace
            if real_otel_context is not None:
                sys.modules["opentelemetry.context"] = real_otel_context


class TestExceptionHandling:
    """Tests for exception handling in edge cases."""

    def test_on_trace_start_handles_exception(self, mock_otel: Any) -> None:
        """Test that on_trace_start handles exceptions gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Create a trace that will cause an exception
        trace = MagicMock()
        trace.trace_id = "test_trace"
        # Make name property raise an exception
        type(trace).name = property(lambda self: (_ for _ in ()).throw(RuntimeError("test error")))

        # Should not raise, just log the error
        processor.on_trace_start(trace)

    def test_on_trace_end_handles_exception(self, mock_otel: Any) -> None:
        """Test that on_trace_end handles exceptions gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace first
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        # Make the span's set_status raise an exception
        with processor._lock:
            span = processor._trace_root_spans.get("test_trace")
            if span:
                span.set_status = MagicMock(side_effect=RuntimeError("status error"))

        # Should not raise, just log the error
        processor.on_trace_end(trace)

    def test_on_trace_end_span_end_exception(self, mock_otel: Any) -> None:
        """Test that on_trace_end handles span.end() exceptions."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace first
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        # Make the span's end() raise an exception
        with processor._lock:
            span = processor._trace_root_spans.get("test_trace")
            if span:
                span.end = MagicMock(side_effect=RuntimeError("end error"))

        # Should not raise, just log the error
        processor.on_trace_end(trace)

    def test_on_span_start_handles_exception(self, mock_otel: Any) -> None:
        """Test that on_span_start handles exceptions gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Create a span that will cause an exception
        span = MagicMock()
        span.span_id = "test_span"
        span.trace_id = "test_trace"
        span.parent_id = None
        # Make span_data property raise an exception
        type(span).span_data = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("test error"))
        )

        # Should not raise, just log the error
        processor.on_span_start(span)

    def test_on_span_end_handles_exception(self, mock_otel: Any) -> None:
        """Test that on_span_end handles exceptions gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace and span first
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="test_span",
            trace_id="test_trace",
            parent_id=None,
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        # Make the otel span's set_status raise an exception
        with processor._lock:
            otel_span = processor._active_spans.get("test_span")
            if otel_span:
                otel_span.set_status = MagicMock(side_effect=RuntimeError("status error"))

        # Should not raise, just log the error
        processor.on_span_end(span)

    def test_on_span_end_span_end_exception(self, mock_otel: Any) -> None:
        """Test that on_span_end handles span.end() exceptions."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace and span first
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="test_span",
            trace_id="test_trace",
            parent_id=None,
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        # Make the otel span's end() raise an exception
        with processor._lock:
            otel_span = processor._active_spans.get("test_span")
            if otel_span:
                otel_span.end = MagicMock(side_effect=RuntimeError("end error"))

        # Should not raise, just log the error
        processor.on_span_end(span)

    def test_shutdown_handles_span_exceptions(self, mock_otel: Any) -> None:
        """Test that shutdown handles exceptions when closing spans."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace and span but don't end them
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="test_span",
            trace_id="test_trace",
            parent_id=None,
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        # Make the spans raise exceptions on end()
        with processor._lock:
            for span_obj in processor._active_spans.values():
                span_obj.end = MagicMock(side_effect=RuntimeError("end error"))
            for span_obj in processor._trace_root_spans.values():
                span_obj.end = MagicMock(side_effect=RuntimeError("end error"))

        # Should not raise, just handle exceptions silently
        processor.shutdown()

        # Verify spans were cleared
        assert len(processor._active_spans) == 0
        assert len(processor._trace_root_spans) == 0

    def test_force_flush_handles_exception(self, mock_otel: Any) -> None:
        """Test that force_flush handles exceptions gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Make get_tracer_provider raise an exception
        processor._trace.get_tracer_provider = MagicMock(side_effect=RuntimeError("provider error"))

        # Should not raise, just log the warning
        processor.force_flush()

    def test_unknown_span_type_export_exception(self, mock_otel: Any) -> None:
        """Test unknown span type handles export() exception."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        # Create span data with unknown type that raises on export
        span_data = MagicMock()
        span_data.type = "unknown_type"
        span_data.export = MagicMock(side_effect=RuntimeError("export error"))

        span = MockSDKSpan(
            span_id="test_span",
            trace_id="test_trace",
            parent_id=None,
            span_data=span_data,
        )

        # Should not raise
        processor.on_span_start(span)
        processor.on_span_end(span)

    def test_generation_span_input_serialization_exception(self, mock_otel: Any) -> None:
        """Test generation span handles input serialization exception."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()

        # Start a trace
        trace = MockTrace(trace_id="test_trace", name="Test")
        processor.on_trace_start(trace)

        # Create generation span data with non-serializable input
        class NonSerializable:
            pass

        span_data = MockGenerationSpanData(
            model="gpt-4",
            input=[NonSerializable()],
            output=[NonSerializable()],
        )

        span = MockSDKSpan(
            span_id="test_span",
            trace_id="test_trace",
            parent_id=None,
            span_data=span_data,
        )

        processor.on_span_start(span)
        # Should not raise on end even with non-serializable data
        processor.on_span_end(span)


class TestProcessorConfig:
    """Tests for ProcessorConfig (Phase 3)."""

    def test_default_config(self) -> None:
        """Test ProcessorConfig has sensible defaults."""
        from openai_agents_opentelemetry import ProcessorConfig

        config = ProcessorConfig()
        assert config.capture_prompts is True
        assert config.capture_completions is True
        assert config.capture_tool_inputs is True
        assert config.capture_tool_outputs is True
        assert config.max_attribute_length == 4096
        assert config.max_event_length == 8192
        assert config.content_filter is None

    def test_custom_config(self) -> None:
        """Test ProcessorConfig with custom values."""
        from openai_agents_opentelemetry import ProcessorConfig

        config = ProcessorConfig(
            capture_prompts=False,
            capture_completions=False,
            capture_tool_inputs=False,
            capture_tool_outputs=False,
            max_attribute_length=1024,
            max_event_length=2048,
        )
        assert config.capture_prompts is False
        assert config.capture_completions is False
        assert config.capture_tool_inputs is False
        assert config.capture_tool_outputs is False
        assert config.max_attribute_length == 1024
        assert config.max_event_length == 2048

    def test_processor_accepts_config(self, mock_otel: Any) -> None:
        """Test that processor accepts ProcessorConfig."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_prompts=False)
        processor = OpenTelemetryTracingProcessor(config=config)
        assert processor._config.capture_prompts is False


class TestSpanEvents:
    """Tests for span events (Phase 3)."""

    def test_generation_span_prompt_event(self, mock_otel: Any) -> None:
        """Test generation span adds prompt event when configured."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_prompts=True)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        prompt_events = [e for e in otel_span.events if e["name"] == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1
        assert "gen_ai.prompt" in prompt_events[0]["attributes"]

    def test_generation_span_completion_event(self, mock_otel: Any) -> None:
        """Test generation span adds completion event when configured."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_completions=True)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                output=[{"role": "assistant", "content": "Hi there!"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        completion_events = [
            e for e in otel_span.events if e["name"] == "gen_ai.content.completion"
        ]
        assert len(completion_events) == 1
        assert "gen_ai.completion" in completion_events[0]["attributes"]

    def test_generation_span_no_events_when_disabled(self, mock_otel: Any) -> None:
        """Test generation span does not add events when capture is disabled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_prompts=False, capture_completions=False)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        gen_ai_events = [e for e in otel_span.events if e["name"].startswith("gen_ai.content")]
        assert len(gen_ai_events) == 0

    def test_function_span_input_event(self, mock_otel: Any) -> None:
        """Test function span adds input event when configured."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_tool_inputs=True)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockFunctionSpanData(name="search", input='{"query": "test"}'),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        input_events = [e for e in otel_span.events if e["name"] == "gen_ai.tool.input"]
        assert len(input_events) == 1
        assert "gen_ai.tool.call.arguments" in input_events[0]["attributes"]

    def test_function_span_output_event(self, mock_otel: Any) -> None:
        """Test function span adds output event when configured."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_tool_outputs=True)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span_data = MockFunctionSpanData(name="search", input='{"query": "test"}')
        span_data.output = "search results here"
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=span_data,
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        output_events = [e for e in otel_span.events if e["name"] == "gen_ai.tool.output"]
        assert len(output_events) == 1
        assert "gen_ai.tool.call.result" in output_events[0]["attributes"]

    def test_function_span_no_events_when_disabled(self, mock_otel: Any) -> None:
        """Test function span does not add events when capture is disabled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(capture_tool_inputs=False, capture_tool_outputs=False)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span_data = MockFunctionSpanData(name="search", input='{"query": "test"}')
        span_data.output = "results"
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=span_data,
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        tool_events = [e for e in otel_span.events if e["name"].startswith("gen_ai.tool")]
        assert len(tool_events) == 0

    def test_guardrail_span_event(self, mock_otel: Any) -> None:
        """Test guardrail span adds evaluated event."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGuardrailSpanData(name="content_filter", triggered=True),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        guardrail_events = [e for e in otel_span.events if e["name"] == "guardrail.evaluated"]
        assert len(guardrail_events) == 1
        assert guardrail_events[0]["attributes"]["guardrail.name"] == "content_filter"
        assert guardrail_events[0]["attributes"]["guardrail.triggered"] is True

    def test_handoff_span_event(self, mock_otel: Any) -> None:
        """Test handoff span adds executed event."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockHandoffSpanData(from_agent="agent_a", to_agent="agent_b"),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        handoff_events = [e for e in otel_span.events if e["name"] == "handoff.executed"]
        assert len(handoff_events) == 1
        assert handoff_events[0]["attributes"]["handoff.from"] == "agent_a"
        assert handoff_events[0]["attributes"]["handoff.to"] == "agent_b"


class TestContentFilter:
    """Tests for content filtering (Phase 3)."""

    def test_content_filter_applied_to_prompt(self, mock_otel: Any) -> None:
        """Test content filter is applied to prompt content."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        def redact_filter(content: str, context: str) -> str:
            return content.replace("secret", "[REDACTED]")

        config = ProcessorConfig(capture_prompts=True, content_filter=redact_filter)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "The secret code is 123"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        prompt_events = [e for e in otel_span.events if e["name"] == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1
        assert "[REDACTED]" in prompt_events[0]["attributes"]["gen_ai.prompt"]
        assert "secret" not in prompt_events[0]["attributes"]["gen_ai.prompt"]

    def test_content_filter_applied_to_completion(self, mock_otel: Any) -> None:
        """Test content filter is applied to completion content."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        def redact_filter(content: str, context: str) -> str:
            return content.replace("password", "[HIDDEN]")

        config = ProcessorConfig(capture_completions=True, content_filter=redact_filter)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                output=[{"role": "assistant", "content": "Your password is abc123"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        completion_events = [
            e for e in otel_span.events if e["name"] == "gen_ai.content.completion"
        ]
        assert len(completion_events) == 1
        assert "[HIDDEN]" in completion_events[0]["attributes"]["gen_ai.completion"]

    def test_content_filter_receives_context(self, mock_otel: Any) -> None:
        """Test content filter receives correct context parameter."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        contexts_received: list[str] = []

        def tracking_filter(content: str, context: str) -> str:
            contexts_received.append(context)
            return content

        config = ProcessorConfig(
            capture_prompts=True,
            capture_completions=True,
            content_filter=tracking_filter,
        )
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "test"}],
                output=[{"role": "assistant", "content": "response"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        assert "prompt" in contexts_received
        assert "completion" in contexts_received

    def test_content_filter_exception_handled(self, mock_otel: Any) -> None:
        """Test content filter exceptions are handled gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        def failing_filter(content: str, context: str) -> str:
            raise ValueError("Filter failed")

        config = ProcessorConfig(capture_prompts=True, content_filter=failing_filter)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockGenerationSpanData(
                model="gpt-4",
                input=[{"role": "user", "content": "test content"}],
            ),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        # Should not raise even though filter fails
        processor.on_span_end(span)  # type: ignore[arg-type]

        # Original content should be used when filter fails
        otel_span = mock_otel["tracer"].spans[1]
        prompt_events = [e for e in otel_span.events if e["name"] == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1
        assert "test content" in prompt_events[0]["attributes"]["gen_ai.prompt"]

    def test_content_filter_applied_to_tool_input(self, mock_otel: Any) -> None:
        """Test content filter is applied to tool input."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        def redact_filter(content: str, context: str) -> str:
            if context == "tool_input":
                return content.replace("api_key", "[API_KEY]")
            return content

        config = ProcessorConfig(capture_tool_inputs=True, content_filter=redact_filter)
        processor = OpenTelemetryTracingProcessor(config=config)
        trace = MockTrace(trace_id="trace_123")
        span = MockSDKSpan(
            trace_id="trace_123",
            span_data=MockFunctionSpanData(name="api_call", input='{"api_key": "secret123"}'),
        )
        processor.on_trace_start(trace)  # type: ignore[arg-type]
        processor.on_span_start(span)  # type: ignore[arg-type]
        processor.on_span_end(span)  # type: ignore[arg-type]

        otel_span = mock_otel["tracer"].spans[1]
        input_events = [e for e in otel_span.events if e["name"] == "gen_ai.tool.input"]
        assert len(input_events) == 1
        assert "[API_KEY]" in input_events[0]["attributes"]["gen_ai.tool.call.arguments"]


class TestCreateResource:
    """Tests for create_resource helper (Phase 3)."""

    def test_create_resource_basic(self) -> None:
        """Test create_resource with basic parameters."""
        mock_resource = MagicMock()
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = mock_resource

        mock_sdk_resources = MagicMock()
        mock_sdk_resources.Resource = mock_resource_cls

        with patch.dict("sys.modules", {"opentelemetry.sdk.resources": mock_sdk_resources}):
            from openai_agents_opentelemetry import create_resource

            result = create_resource(service_name="test-service")

            mock_resource_cls.create.assert_called_once()
            call_args = mock_resource_cls.create.call_args[0][0]
            assert call_args["service.name"] == "test-service"
            assert call_args["telemetry.sdk.name"] == "openai-agents-opentelemetry"
            assert call_args["telemetry.sdk.language"] == "python"
            assert call_args["agent.sdk.name"] == "openai-agents"
            assert result == mock_resource

    def test_create_resource_with_version(self) -> None:
        """Test create_resource with service version."""
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = MagicMock()

        mock_sdk_resources = MagicMock()
        mock_sdk_resources.Resource = mock_resource_cls

        with patch.dict("sys.modules", {"opentelemetry.sdk.resources": mock_sdk_resources}):
            from openai_agents_opentelemetry import create_resource

            create_resource(service_name="test-service", service_version="1.2.3")

            call_args = mock_resource_cls.create.call_args[0][0]
            assert call_args["service.version"] == "1.2.3"

    def test_create_resource_with_additional_attributes(self) -> None:
        """Test create_resource with additional custom attributes."""
        mock_resource_cls = MagicMock()
        mock_resource_cls.create.return_value = MagicMock()

        mock_sdk_resources = MagicMock()
        mock_sdk_resources.Resource = mock_resource_cls

        with patch.dict("sys.modules", {"opentelemetry.sdk.resources": mock_sdk_resources}):
            from openai_agents_opentelemetry import create_resource

            create_resource(
                service_name="test-service",
                additional_attributes={
                    "deployment.environment": "production",
                    "custom.attribute": "value",
                },
            )

            call_args = mock_resource_cls.create.call_args[0][0]
            assert call_args["deployment.environment"] == "production"
            assert call_args["custom.attribute"] == "value"

    def test_create_resource_import_error(self) -> None:
        """Test create_resource raises ImportError when SDK not installed."""
        with patch.dict("sys.modules", {"opentelemetry.sdk.resources": None}):
            from openai_agents_opentelemetry.opentelemetry_processor import create_resource

            with pytest.raises(ImportError, match="OpenTelemetry SDK is required"):
                create_resource(service_name="test-service")


class TestContentFilterType:
    """Tests for ContentFilter type alias."""

    def test_content_filter_type_is_callable(self) -> None:
        """Test ContentFilter type can be used as a type hint."""
        from openai_agents_opentelemetry import ContentFilter

        def my_filter(content: str, context: str) -> str:
            return content.upper()

        # This should type-check correctly
        filter_func: ContentFilter = my_filter
        assert filter_func("test", "prompt") == "TEST"


class TestMetricsSupport:
    """Tests for metrics support (Phase 4)."""

    def test_metrics_disabled_by_default(self, mock_otel: Any) -> None:
        """Test that metrics are disabled by default."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor()
        assert processor._enable_metrics is False
        assert processor._token_histogram is None
        assert processor._duration_histogram is None
        assert processor._tool_counter is None

    def test_metrics_enabled_flag(self, mock_otel: Any) -> None:
        """Test that enable_metrics flag is stored."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = MagicMock()

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            assert processor._enable_metrics is True

    def test_metrics_init_creates_instruments(self, mock_otel: Any) -> None:
        """Test that metrics instruments are created when enabled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_metrics.return_value.get_meter.return_value = mock_meter

            _processor = OpenTelemetryTracingProcessor(enable_metrics=True)

            # Verify histograms created
            histogram_calls = mock_meter.create_histogram.call_args_list
            assert len(histogram_calls) == 2
            assert histogram_calls[0][0][0] == "gen_ai.client.token.usage"
            assert histogram_calls[1][0][0] == "gen_ai.client.operation.duration"

            # Verify counters created
            counter_calls = mock_meter.create_counter.call_args_list
            assert len(counter_calls) == 4
            counter_names = [call[0][0] for call in counter_calls]
            assert "agent.tool.invocations" in counter_names
            assert "agent.handoffs" in counter_names
            assert "agent.guardrail.triggers" in counter_names
            assert "agent.errors" in counter_names

    def test_record_token_usage_when_disabled(self, mock_otel: Any) -> None:
        """Test that token usage is not recorded when metrics disabled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        processor = OpenTelemetryTracingProcessor(enable_metrics=False)
        # Should not raise even though histogram is None
        processor._record_token_usage({"input_tokens": 100, "output_tokens": 50}, "gpt-4")

    def test_record_token_usage_when_enabled(self, mock_otel: Any) -> None:
        """Test that token usage is recorded when metrics enabled."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_histogram = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = mock_histogram
            mock_meter.create_counter.return_value = MagicMock()

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_token_usage({"input_tokens": 100, "output_tokens": 50}, "gpt-4")

            # Verify histogram.record was called twice (input and output)
            assert mock_histogram.record.call_count == 2
            calls = mock_histogram.record.call_args_list

            # Check input tokens call
            assert calls[0][0][0] == 100
            assert calls[0][1]["attributes"]["gen_ai.token.type"] == "input"
            assert calls[0][1]["attributes"]["gen_ai.request.model"] == "gpt-4"

            # Check output tokens call
            assert calls[1][0][0] == 50
            assert calls[1][1]["attributes"]["gen_ai.token.type"] == "output"

    def test_record_tool_invocation(self, mock_otel: Any) -> None:
        """Test that tool invocations are recorded."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_tool_invocation("search_web")

            mock_counter.add.assert_called()
            call_args = mock_counter.add.call_args
            assert call_args[0][0] == 1
            assert call_args[1]["attributes"]["gen_ai.tool.name"] == "search_web"

    def test_record_handoff(self, mock_otel: Any) -> None:
        """Test that handoffs are recorded."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_handoff("agent_a", "agent_b")

            mock_counter.add.assert_called()
            call_args = mock_counter.add.call_args
            assert call_args[0][0] == 1
            assert call_args[1]["attributes"]["agent.handoff.from"] == "agent_a"
            assert call_args[1]["attributes"]["agent.handoff.to"] == "agent_b"

    def test_record_guardrail_trigger_when_triggered(self, mock_otel: Any) -> None:
        """Test that guardrail triggers are recorded when triggered=True."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_guardrail_trigger("content_filter", triggered=True)

            mock_counter.add.assert_called()
            call_args = mock_counter.add.call_args
            assert call_args[0][0] == 1
            assert call_args[1]["attributes"]["agent.guardrail.name"] == "content_filter"

    def test_record_guardrail_trigger_when_not_triggered(self, mock_otel: Any) -> None:
        """Test that guardrail triggers are NOT recorded when triggered=False."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            # Reset mock to clear init calls
            mock_counter.reset_mock()

            processor._record_guardrail_trigger("content_filter", triggered=False)

            # Should not be called when not triggered
            mock_counter.add.assert_not_called()

    def test_record_error(self, mock_otel: Any) -> None:
        """Test that errors are recorded."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_error("ValueError", "generation")

            mock_counter.add.assert_called()
            call_args = mock_counter.add.call_args
            assert call_args[0][0] == 1
            assert call_args[1]["attributes"]["error.type"] == "ValueError"
            assert call_args[1]["attributes"]["agent.span.type"] == "generation"

    def test_record_operation_duration(self, mock_otel: Any) -> None:
        """Test that operation duration is recorded."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_histogram = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = mock_histogram
            mock_meter.create_counter.return_value = MagicMock()

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)
            processor._record_operation_duration(1.5, "gpt-4")

            mock_histogram.record.assert_called()
            call_args = mock_histogram.record.call_args
            assert call_args[0][0] == 1.5
            assert call_args[1]["attributes"]["gen_ai.request.model"] == "gpt-4"

    def test_metrics_recorded_on_span_end(self, mock_otel: Any) -> None:
        """Test that metrics are recorded when span ends."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_histogram = MagicMock()
        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = mock_histogram
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)

            trace = MockTrace(trace_id="trace_123")
            span = MockSDKSpan(
                trace_id="trace_123",
                span_data=MockGenerationSpanData(
                    model="gpt-4", usage={"input_tokens": 100, "output_tokens": 50}
                ),
            )

            processor.on_trace_start(trace)  # type: ignore[arg-type]
            processor.on_span_start(span)  # type: ignore[arg-type]

            # Reset to only count calls from on_span_end
            mock_histogram.reset_mock()

            processor.on_span_end(span)  # type: ignore[arg-type]

            # Should have recorded token usage
            assert mock_histogram.record.call_count >= 2

    def test_tool_metric_recorded_on_function_span_end(self, mock_otel: Any) -> None:
        """Test that tool invocation metric is recorded on function span end."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        mock_counter = MagicMock()

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_meter = MagicMock()
            mock_metrics.return_value.get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = mock_counter

            processor = OpenTelemetryTracingProcessor(enable_metrics=True)

            trace = MockTrace(trace_id="trace_123")
            span = MockSDKSpan(
                trace_id="trace_123",
                span_data=MockFunctionSpanData(name="search_web", input="{}"),
            )

            processor.on_trace_start(trace)  # type: ignore[arg-type]
            processor.on_span_start(span)  # type: ignore[arg-type]

            # Reset to only count calls from on_span_end
            mock_counter.reset_mock()

            processor.on_span_end(span)  # type: ignore[arg-type]

            # Should have recorded tool invocation
            mock_counter.add.assert_called()


class TestMetricsBuckets:
    """Tests for metric bucket constants."""

    def test_token_buckets_exported(self) -> None:
        """Test that TOKEN_BUCKETS is exported and has expected values."""
        from openai_agents_opentelemetry import TOKEN_BUCKETS

        assert isinstance(TOKEN_BUCKETS, tuple)
        assert len(TOKEN_BUCKETS) > 0
        assert TOKEN_BUCKETS[0] == 1
        # Verify exponential growth
        assert all(TOKEN_BUCKETS[i] < TOKEN_BUCKETS[i + 1] for i in range(len(TOKEN_BUCKETS) - 1))

    def test_duration_buckets_exported(self) -> None:
        """Test that DURATION_BUCKETS is exported and has expected values."""
        from openai_agents_opentelemetry import DURATION_BUCKETS

        assert isinstance(DURATION_BUCKETS, tuple)
        assert len(DURATION_BUCKETS) > 0
        assert DURATION_BUCKETS[0] == 0.01
        # Verify exponential growth
        assert all(
            DURATION_BUCKETS[i] < DURATION_BUCKETS[i + 1] for i in range(len(DURATION_BUCKETS) - 1)
        )


class TestBaggageSupport:
    """Tests for baggage support (Phase 5)."""

    def test_baggage_keys_default_empty(self, mock_otel: Any) -> None:
        """Test that baggage_keys defaults to empty list."""
        from openai_agents_opentelemetry import ProcessorConfig

        config = ProcessorConfig()
        assert config.baggage_keys == []

    def test_baggage_keys_configured(self, mock_otel: Any) -> None:
        """Test that baggage_keys can be configured."""
        from openai_agents_opentelemetry import ProcessorConfig

        config = ProcessorConfig(baggage_keys=["user.id", "session.id", "tenant.id"])
        assert config.baggage_keys == ["user.id", "session.id", "tenant.id"]

    def test_baggage_attributes_added_to_span(self, mock_otel: Any) -> None:
        """Test that baggage values are added as span attributes."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        # Configure baggage to return values
        mock_otel["baggage"].get_baggage.side_effect = lambda key: {
            "user.id": "user-123",
            "session.id": "session-456",
        }.get(key)

        config = ProcessorConfig(baggage_keys=["user.id", "session.id"])
        processor = OpenTelemetryTracingProcessor(config=config)

        # Create a trace and span
        trace = MockTrace(trace_id="trace-1")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="span-1",
            trace_id="trace-1",
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        # Check that baggage values were added to the span
        otel_span = mock_otel["tracer"].spans[-1]
        assert otel_span.attributes.get("user.id") == "user-123"
        assert otel_span.attributes.get("session.id") == "session-456"

    def test_baggage_missing_key_not_added(self, mock_otel: Any) -> None:
        """Test that missing baggage keys are not added as attributes."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        # Configure baggage to return None for missing keys
        mock_otel["baggage"].get_baggage.return_value = None

        config = ProcessorConfig(baggage_keys=["user.id", "missing.key"])
        processor = OpenTelemetryTracingProcessor(config=config)

        trace = MockTrace(trace_id="trace-1")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="span-1",
            trace_id="trace-1",
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        otel_span = mock_otel["tracer"].spans[-1]
        assert "user.id" not in otel_span.attributes
        assert "missing.key" not in otel_span.attributes

    def test_baggage_no_keys_configured_skips_lookup(self, mock_otel: Any) -> None:
        """Test that baggage lookup is skipped when no keys configured."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        config = ProcessorConfig(baggage_keys=[])  # Empty list
        processor = OpenTelemetryTracingProcessor(config=config)

        trace = MockTrace(trace_id="trace-1")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="span-1",
            trace_id="trace-1",
            span_data=MockAgentSpanData(name="TestAgent"),
        )
        processor.on_span_start(span)

        # get_baggage should not be called when no keys are configured
        mock_otel["baggage"].get_baggage.assert_not_called()

    def test_baggage_exception_handled_gracefully(self, mock_otel: Any) -> None:
        """Test that exceptions during baggage reading are handled gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

        # Configure baggage to raise an exception
        mock_otel["baggage"].get_baggage.side_effect = RuntimeError("Baggage error")

        config = ProcessorConfig(baggage_keys=["user.id"])
        processor = OpenTelemetryTracingProcessor(config=config)

        trace = MockTrace(trace_id="trace-1")
        processor.on_trace_start(trace)

        span = MockSDKSpan(
            span_id="span-1",
            trace_id="trace-1",
            span_data=MockAgentSpanData(name="TestAgent"),
        )

        # Should not raise, exception is caught and logged
        processor.on_span_start(span)

        # Span should still be created (2 spans: trace root + agent span)
        assert len(mock_otel["tracer"].spans) == 2


class TestMetricsImportError:
    """Tests for metrics import error handling."""

    def test_metrics_import_error(self) -> None:
        """Test that metrics import error is handled gracefully."""
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        with patch(
            "openai_agents_opentelemetry.opentelemetry_processor._try_import_opentelemetry_metrics"
        ) as mock_metrics:
            mock_metrics.side_effect = ImportError("Metrics not available")

            with pytest.raises(ImportError, match="Metrics not available"):
                OpenTelemetryTracingProcessor(enable_metrics=True)
