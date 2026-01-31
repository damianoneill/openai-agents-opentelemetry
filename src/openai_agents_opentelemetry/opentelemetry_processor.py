"""OpenTelemetry tracing processor for the OpenAI Agents SDK.

This module provides an OpenTelemetry-based TracingProcessor that bridges
the Agents SDK's tracing system to OpenTelemetry, enabling export to any
OTLP-compatible backend (Jaeger, Datadog, Honeycomb, Grafana Tempo, etc.).

The processor maps SDK trace and span types to OpenTelemetry spans following
the OpenTelemetry Semantic Conventions for Generative AI systems where applicable.

See: https://opentelemetry.io/docs/specs/semconv/gen-ai/

Example:
    ```python
    from agents import add_trace_processor
    from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

    # Create and register the processor
    otel_processor = OpenTelemetryTracingProcessor()
    add_trace_processor(otel_processor)

    # Now all agent traces will be exported to your configured OTel backend
    result = await Runner.run(agent, "Hello!")
    ```

Requirements:
    pip install opentelemetry-api opentelemetry-sdk
    # Or: pip install openai-agents-opentelemetry
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.tracing import TracingProcessor

from ._version import __version__

if TYPE_CHECKING:
    from agents.tracing.spans import Span as AgentSpan
    from agents.tracing.traces import Trace as AgentTrace

logger = logging.getLogger(__name__)

# Default tracer name for agent spans
DEFAULT_TRACER_NAME = "openai.agents"

# Schema URL indicates which semantic conventions version this instrumentation follows.
# This should be updated when adopting newer semantic convention versions.
# See: https://opentelemetry.io/docs/specs/otel/schemas/
# Current versions: https://github.com/open-telemetry/semantic-conventions/tree/main/schemas
#
# Note: GenAI semantic conventions are in Development status. Update this URL
# when conventions stabilize or when adopting newer experimental versions.
SCHEMA_URL = "https://opentelemetry.io/schemas/1.28.0"

# Attribute key prefixes following OTel semantic conventions
_ATTR_PREFIX_GEN_AI = "gen_ai"
_ATTR_PREFIX_AGENT = "agent"

# Recommended histogram buckets per OTel GenAI semantic conventions
# Token buckets: exponential growth for token counts
TOKEN_BUCKETS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
# Duration buckets: exponential growth for seconds (10ms to ~82s)
DURATION_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)

# Type alias for content filter callback
# Receives (content, context) where context is e.g., "prompt", "completion", "tool_input"
# Returns filtered content string
ContentFilter = Callable[[str, str], str]


@dataclass
class ProcessorConfig:
    """Configuration for the OpenTelemetry tracing processor.

    Controls what content is captured and how it is processed for privacy
    and compliance requirements.

    Attributes:
        capture_prompts: Whether to capture prompt/input content as span events.
        capture_completions: Whether to capture completion/output content as span events.
        capture_tool_inputs: Whether to capture tool input arguments.
        capture_tool_outputs: Whether to capture tool output results.
        max_attribute_length: Maximum length for span attributes (default 4096).
        max_event_length: Maximum length for span event attributes (default 8192).
        content_filter: Optional callback to filter/redact content before capture.
            Receives (content, context) and returns filtered content.
        baggage_keys: List of baggage keys to read from OpenTelemetry context and
            add as span attributes. Enables context propagation across services.

    Example:
        ```python
        import re

        def redact_pii(content: str, context: str) -> str:
            # Redact SSNs
            content = re.sub(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN REDACTED]", content)
            # Redact email addresses
            content = re.sub(r"\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b", "[EMAIL REDACTED]", content)
            return content

        config = ProcessorConfig(
            capture_prompts=True,
            capture_completions=True,
            max_attribute_length=1024,
            content_filter=redact_pii,
            baggage_keys=["user.id", "session.id", "tenant.id"],
        )
        processor = OpenTelemetryTracingProcessor(config=config)
        ```
    """

    # Content capture toggles
    capture_prompts: bool = True
    capture_completions: bool = True
    capture_tool_inputs: bool = True
    capture_tool_outputs: bool = True

    # Size limits
    max_attribute_length: int = 4096
    max_event_length: int = 8192

    # Custom content filter callback for redaction/transformation
    content_filter: ContentFilter | None = field(default=None)

    # Baggage keys to read from OpenTelemetry context and add as span attributes
    # Enables propagation of context like user.id, session.id across agent spans
    baggage_keys: list[str] = field(default_factory=list)


def _try_import_opentelemetry() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Try to import OpenTelemetry dependencies.

    Returns:
        Tuple of (trace module, SpanKind, Status, StatusCode, Context, baggage)

    Raises:
        ImportError: If opentelemetry packages are not installed.
    """
    try:
        from opentelemetry import baggage, trace
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry packages are required for OpenTelemetryTracingProcessor. "
            "Install them with: pip install opentelemetry-api opentelemetry-sdk "
            "or pip install openai-agents[opentelemetry]"
        ) from e
    return trace, SpanKind, Status, StatusCode, otel_context, baggage


def _try_import_opentelemetry_metrics() -> Any:
    """Try to import OpenTelemetry metrics dependencies.

    Returns:
        The metrics module.

    Raises:
        ImportError: If opentelemetry metrics packages are not installed.
    """
    try:
        from opentelemetry import metrics
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry metrics packages are required for metrics support. "
            "Install them with: pip install opentelemetry-api opentelemetry-sdk"
        ) from e
    return metrics


class OpenTelemetryTracingProcessor(TracingProcessor):
    """A TracingProcessor that exports traces to OpenTelemetry.

    This processor receives trace and span events from the Agents SDK and
    creates corresponding OpenTelemetry spans. It maintains the parent-child
    relationships between spans and maps SDK-specific data to OTel attributes.

    The processor is thread-safe and can be used in concurrent environments.

    Attributes:
        tracer_name: The name of the OTel tracer (default: "openai.agents").

    Example:
        ```python
        from agents import add_trace_processor, set_trace_processors
        from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

        # Option 1: Add alongside default OpenAI backend processor
        processor = OpenTelemetryTracingProcessor()
        add_trace_processor(processor)

        # Option 2: Replace default processor (traces only go to OTel)
        processor = OpenTelemetryTracingProcessor()
        set_trace_processors([processor])
        ```
    """

    def __init__(
        self,
        tracer_name: str = DEFAULT_TRACER_NAME,
        config: ProcessorConfig | None = None,
        enable_metrics: bool = False,
    ) -> None:
        """Initialize the OpenTelemetry tracing processor.

        Args:
            tracer_name: Name of the tracer to use. Defaults to "openai.agents".
            config: Optional configuration for content capture and filtering.
                If not provided, uses default ProcessorConfig settings.
            enable_metrics: Whether to enable metrics collection. Defaults to False.
                When enabled, collects token usage, operation duration, and
                agent-specific metrics (tool invocations, handoffs, guardrail triggers).
        """
        self._config = config or ProcessorConfig()
        self._enable_metrics = enable_metrics
        trace, span_kind_class, status_class, status_code_class, otel_context, baggage = (
            _try_import_opentelemetry()
        )

        self._trace = trace
        self._SpanKind = span_kind_class
        self._Status = status_class
        self._StatusCode = status_code_class
        self._otel_context = otel_context
        self._baggage = baggage

        self._tracer_name = tracer_name
        self._tracer = trace.get_tracer(
            tracer_name,
            instrumenting_library_version=__version__,
            schema_url=SCHEMA_URL,
        )

        # Lock for thread-safe access to span tracking dictionaries
        self._lock = threading.Lock()

        # Map SDK trace/span IDs to OTel spans for parent-child relationships.
        # We store only the OTel Span (not context tokens) because we do NOT attach
        # spans to the global context. This avoids out-of-order detach issues when
        # SDK spans overlap (e.g., parallel tool calls).
        self._active_spans: dict[str, Any] = {}
        self._trace_root_spans: dict[str, Any] = {}

        # Initialize metrics instruments if enabled
        self._token_histogram: Any = None
        self._duration_histogram: Any = None
        self._tool_counter: Any = None
        self._handoff_counter: Any = None
        self._guardrail_counter: Any = None
        self._error_counter: Any = None

        if enable_metrics:
            self._init_metrics(tracer_name)

    def _init_metrics(self, meter_name: str) -> None:
        """Initialize OpenTelemetry metrics instruments.

        Args:
            meter_name: Name for the meter (typically same as tracer name).
        """
        metrics = _try_import_opentelemetry_metrics()
        meter = metrics.get_meter(
            meter_name,
            version=__version__,
            schema_url=SCHEMA_URL,
        )

        # Standard OTel GenAI metrics
        self._token_histogram = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        )
        self._duration_histogram = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI operations",
        )

        # OpenAI Agents SDK-specific metrics
        self._tool_counter = meter.create_counter(
            "agent.tool.invocations",
            unit="{invocation}",
            description="Number of tool invocations",
        )
        self._handoff_counter = meter.create_counter(
            "agent.handoffs",
            unit="{handoff}",
            description="Number of agent handoffs",
        )
        self._guardrail_counter = meter.create_counter(
            "agent.guardrail.triggers",
            unit="{trigger}",
            description="Number of guardrail triggers",
        )
        self._error_counter = meter.create_counter(
            "agent.errors",
            unit="{error}",
            description="Number of errors by type",
        )

    def on_trace_start(self, trace: AgentTrace) -> None:
        """Handle SDK trace start by creating an OTel root span.

        Args:
            trace: The SDK trace that started.

        Note:
            We do NOT attach the root span to the global OpenTelemetry context.
            Instead, child spans explicitly reference their parent via the span ID
            lookup in _trace_root_spans. This avoids context ordering issues when
            multiple traces or spans overlap.
        """
        try:
            trace_id = trace.trace_id
            workflow_name = trace.name

            # Create root span for this workflow/trace
            span = self._tracer.start_span(
                name=f"workflow: {workflow_name}",
                kind=self._SpanKind.INTERNAL,
                attributes={
                    f"{_ATTR_PREFIX_AGENT}.workflow.name": workflow_name,
                    f"{_ATTR_PREFIX_AGENT}.trace_id": trace_id,
                },
            )

            # Add optional trace metadata
            group_id = getattr(trace, "group_id", None)
            if group_id:
                span.set_attribute(f"{_ATTR_PREFIX_AGENT}.group_id", group_id)

            metadata = getattr(trace, "metadata", None)
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    attr_key = f"{_ATTR_PREFIX_AGENT}.metadata.{key}"
                    span.set_attribute(attr_key, _safe_attribute_value(value))

            with self._lock:
                self._trace_root_spans[trace_id] = span

            logger.debug(f"Started OTel span for trace: {trace_id} ({workflow_name})")

        except Exception as e:
            logger.error(f"Failed to create OTel span for trace start: {e}")

    def on_trace_end(self, trace: AgentTrace) -> None:
        """Handle SDK trace end by closing the OTel root span.

        Args:
            trace: The SDK trace that ended.
        """
        trace_id = trace.trace_id
        otel_span = None

        try:
            with self._lock:
                otel_span = self._trace_root_spans.pop(trace_id, None)

            if otel_span:
                otel_span.set_status(self._Status(self._StatusCode.OK))
                logger.debug(f"Ended OTel span for trace: {trace_id}")
            else:
                logger.warning(f"No OTel span found for trace end: {trace_id}")

        except Exception as e:
            logger.error(f"Failed to process OTel span for trace end: {e}")

        finally:
            # Always end the span, even if processing failed
            if otel_span:
                try:
                    otel_span.end()
                except Exception as e:
                    logger.error(f"Failed to end OTel trace span: {e}")

    def on_span_start(self, span: AgentSpan[Any]) -> None:
        """Handle SDK span start by creating a child OTel span.

        Args:
            span: The SDK span that started.

        Note:
            We do NOT attach spans to the global OpenTelemetry context because SDK spans
            can overlap (e.g., parallel tool calls). Attaching would require LIFO detach
            order, which the SDK does not guarantee. Instead, we explicitly pass parent
            context when creating child spans, avoiding global context manipulation.
        """
        try:
            span_id = span.span_id
            trace_id = span.trace_id
            parent_id = span.parent_id
            span_data = span.span_data

            # Determine the parent context by looking up the parent span explicitly.
            # We do NOT use the global context to avoid issues with overlapping spans.
            parent_context = None
            with self._lock:
                if parent_id and parent_id in self._active_spans:
                    parent_span = self._active_spans[parent_id]
                    parent_context = self._trace.set_span_in_context(parent_span)
                elif trace_id in self._trace_root_spans:
                    parent_span = self._trace_root_spans[trace_id]
                    parent_context = self._trace.set_span_in_context(parent_span)

            # Map span data to OTel span name and attributes
            otel_span_name, attributes, span_kind = self._map_span_data(span_data)

            # Add common attributes
            attributes[f"{_ATTR_PREFIX_AGENT}.span_id"] = span_id
            attributes[f"{_ATTR_PREFIX_AGENT}.trace_id"] = trace_id
            if parent_id:
                attributes[f"{_ATTR_PREFIX_AGENT}.parent_span_id"] = parent_id

            # Create the OTel span with explicit parent context (not global context)
            otel_span = self._tracer.start_span(
                name=otel_span_name,
                context=parent_context,
                kind=span_kind,
                attributes=attributes,
            )

            # Read baggage from current OpenTelemetry context and add as span attributes
            # This enables context propagation (e.g., user.id, session.id) across agent spans
            self._add_baggage_attributes(otel_span)

            with self._lock:
                self._active_spans[span_id] = otel_span

            logger.debug(f"Started OTel span: {otel_span_name} ({span_id})")

        except Exception as e:
            logger.error(f"Failed to create OTel span for span start: {e}")

    def on_span_end(self, span: AgentSpan[Any]) -> None:
        """Handle SDK span end by closing the OTel span.

        Args:
            span: The SDK span that ended.
        """
        span_id = span.span_id
        otel_span = None
        span_data = span.span_data

        try:
            with self._lock:
                otel_span = self._active_spans.pop(span_id, None)

            if not otel_span:
                logger.warning(f"No OTel span found for span end: {span_id}")
                return

            # Update span with final data (e.g., usage metrics, outputs)
            self._update_span_with_final_data(otel_span, span_data)

            # Add span events for content capture (controlled by config)
            self._add_span_events(otel_span, span_data)

            # Record metrics if enabled
            self._record_span_metrics(span_data, span.error)

            # Handle errors - use _safe_attribute_value to avoid serialization failures
            error = span.error
            if error:
                error_msg = error.get("message", "Unknown error")
                otel_span.set_status(self._Status(self._StatusCode.ERROR, error_msg))
                otel_span.set_attribute("error.message", error.get("message", ""))
                error_data = error.get("data")
                if error_data:
                    # Use _safe_attribute_value to handle non-serializable data
                    otel_span.set_attribute("error.data", _safe_attribute_value(error_data))
                # Record error metric
                self._record_error(type(error).__name__, span_data.type)
            else:
                otel_span.set_status(self._Status(self._StatusCode.OK))

            logger.debug(f"Ended OTel span: {span_id}")

        except Exception as e:
            logger.error(f"Failed to process OTel span end: {e}")

        finally:
            # Always end the span, even if processing failed
            if otel_span:
                try:
                    otel_span.end()
                except Exception as e:
                    logger.error(f"Failed to end OTel span: {e}")

    def shutdown(self) -> None:
        """Clean up resources and close any unclosed spans."""
        with self._lock:
            # End any spans that weren't properly closed
            for _, span in list(self._active_spans.items()):
                try:
                    span.set_status(
                        self._Status(self._StatusCode.ERROR, "Span not properly closed at shutdown")
                    )
                    span.end()
                except Exception:
                    pass

            for _, span in list(self._trace_root_spans.items()):
                try:
                    span.set_status(
                        self._Status(
                            self._StatusCode.ERROR, "Trace not properly closed at shutdown"
                        )
                    )
                    span.end()
                except Exception:
                    pass

            self._active_spans.clear()
            self._trace_root_spans.clear()

        logger.debug("OpenTelemetry tracing processor shutdown complete")

    def force_flush(self) -> None:
        """Force flush any pending spans.

        Note: OTel spans are exported by the TracerProvider's span processor,
        so we delegate to that if available.
        """
        try:
            provider = self._trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush()
        except Exception as e:
            logger.warning(f"Failed to force flush tracer provider: {e}")

    def _map_span_data(self, span_data: Any) -> tuple[str, dict[str, Any], Any]:
        """Map SDK span data to OTel span name, attributes, and kind.

        Args:
            span_data: The SDK SpanData object.

        Returns:
            Tuple of (span_name, attributes_dict, SpanKind).
        """
        span_type = span_data.type
        attributes: dict[str, Any] = {f"{_ATTR_PREFIX_AGENT}.span.type": span_type}
        kind = self._SpanKind.INTERNAL

        if span_type == "agent":
            return self._map_agent_span(span_data, attributes)

        elif span_type == "generation":
            return self._map_generation_span(span_data, attributes)

        elif span_type == "function":
            return self._map_function_span(span_data, attributes)

        elif span_type == "handoff":
            return self._map_handoff_span(span_data, attributes)

        elif span_type == "guardrail":
            return self._map_guardrail_span(span_data, attributes)

        elif span_type == "custom":
            return self._map_custom_span(span_data, attributes)

        elif span_type == "response":
            return self._map_response_span(span_data, attributes)

        elif span_type == "transcription":
            return self._map_transcription_span(span_data, attributes)

        elif span_type == "speech":
            return self._map_speech_span(span_data, attributes)

        elif span_type == "speech_group":
            return "audio.speech_group", attributes, kind

        elif span_type == "mcp_tools":
            return self._map_mcp_tools_span(span_data, attributes)

        else:
            # Generic fallback for unknown span types
            name = f"agent.{span_type}"
            try:
                exported = span_data.export()
                for key, value in exported.items():
                    if key != "type":
                        attributes[f"span.{key}"] = _safe_attribute_value(value)
            except Exception:
                pass
            return name, attributes, kind

    def _map_agent_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map agent span data to OTel format."""
        name = f"agent: {span_data.name}"
        attributes[f"{_ATTR_PREFIX_AGENT}.name"] = span_data.name

        if span_data.handoffs:
            attributes[f"{_ATTR_PREFIX_AGENT}.handoffs"] = json.dumps(span_data.handoffs)
        if span_data.tools:
            attributes[f"{_ATTR_PREFIX_AGENT}.tools"] = json.dumps(span_data.tools)
        if span_data.output_type:
            attributes[f"{_ATTR_PREFIX_AGENT}.output_type"] = span_data.output_type

        return name, attributes, self._SpanKind.INTERNAL

    def _map_generation_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map generation (LLM call) span data to OTel format.

        Uses OpenTelemetry Semantic Conventions for GenAI where applicable.
        See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
        """
        kind = self._SpanKind.CLIENT  # LLM call is an outbound request

        # Required attributes per OTel GenAI semantic conventions
        attributes[f"{_ATTR_PREFIX_GEN_AI}.operation.name"] = "chat"
        attributes[f"{_ATTR_PREFIX_GEN_AI}.provider.name"] = "openai"

        # Span name follows convention: "{operation} {model}"
        name = "chat"
        if span_data.model:
            attributes[f"{_ATTR_PREFIX_GEN_AI}.request.model"] = span_data.model
            name = f"chat {span_data.model}"

        # Recommended attributes from model_config
        if span_data.model_config:
            config = span_data.model_config

            # Map specific known config keys to their semantic convention attributes
            # Helper to check if value is valid (not None and not empty string)
            def _is_valid(val: Any) -> bool:
                return val is not None and val != ""

            if "temperature" in config and _is_valid(config["temperature"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.temperature"] = config["temperature"]
            if "max_tokens" in config and _is_valid(config["max_tokens"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.max_tokens"] = config["max_tokens"]
            if "top_p" in config and _is_valid(config["top_p"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.top_p"] = config["top_p"]
            if "frequency_penalty" in config and _is_valid(config["frequency_penalty"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.frequency_penalty"] = config[
                    "frequency_penalty"
                ]
            if "presence_penalty" in config and _is_valid(config["presence_penalty"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.presence_penalty"] = config[
                    "presence_penalty"
                ]
            if "stop" in config and _is_valid(config["stop"]):
                attributes[f"{_ATTR_PREFIX_GEN_AI}.request.stop_sequences"] = _safe_attribute_value(
                    config["stop"]
                )

        return name, attributes, kind

    def _map_function_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map function/tool span data to OTel format.

        Uses OpenTelemetry Semantic Conventions for GenAI tool execution.
        See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
        """
        # Per OTel convention: span name SHOULD be "execute_tool {gen_ai.tool.name}"
        name = f"execute_tool {span_data.name}"

        # Required attribute
        attributes[f"{_ATTR_PREFIX_GEN_AI}.operation.name"] = "execute_tool"

        # Recommended attributes
        attributes[f"{_ATTR_PREFIX_GEN_AI}.tool.name"] = span_data.name
        attributes[f"{_ATTR_PREFIX_GEN_AI}.tool.type"] = "function"

        # Opt-in: tool call arguments (may contain sensitive data)
        if span_data.input:
            attributes[f"{_ATTR_PREFIX_GEN_AI}.tool.call.arguments"] = _truncate_string(
                span_data.input, 4096
            )

        # MCP-specific metadata
        if span_data.mcp_data:
            attributes["mcp.tool.data"] = json.dumps(span_data.mcp_data)

        return name, attributes, self._SpanKind.INTERNAL

    def _map_handoff_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map handoff span data to OTel format."""
        from_agent = span_data.from_agent or "unknown"
        to_agent = span_data.to_agent or "unknown"
        name = f"handoff: {from_agent} -> {to_agent}"

        attributes[f"{_ATTR_PREFIX_AGENT}.handoff.from"] = from_agent
        attributes[f"{_ATTR_PREFIX_AGENT}.handoff.to"] = to_agent

        return name, attributes, self._SpanKind.INTERNAL

    def _map_guardrail_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map guardrail span data to OTel format."""
        name = f"guardrail: {span_data.name}"
        attributes[f"{_ATTR_PREFIX_AGENT}.guardrail.name"] = span_data.name
        attributes[f"{_ATTR_PREFIX_AGENT}.guardrail.triggered"] = span_data.triggered

        return name, attributes, self._SpanKind.INTERNAL

    def _map_custom_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map custom span data to OTel format."""
        name = f"custom: {span_data.name}"
        attributes["custom.name"] = span_data.name

        if span_data.data:
            for key, value in span_data.data.items():
                attributes[f"custom.data.{key}"] = _safe_attribute_value(value)

        return name, attributes, self._SpanKind.INTERNAL

    def _map_response_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map response span data to OTel format.

        Captures response metadata per OTel GenAI semantic conventions.
        """
        name = "gen_ai.response"
        response = getattr(span_data, "response", None)

        if response:
            if hasattr(response, "id") and response.id:
                attributes[f"{_ATTR_PREFIX_GEN_AI}.response.id"] = response.id
            if hasattr(response, "model") and response.model:
                attributes[f"{_ATTR_PREFIX_GEN_AI}.response.model"] = response.model

        return name, attributes, self._SpanKind.INTERNAL

    def _map_transcription_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map transcription (speech-to-text) span data to OTel format."""
        name = "audio.transcription"

        if span_data.model:
            attributes["audio.model"] = span_data.model
            name = f"audio.transcription: {span_data.model}"

        return name, attributes, self._SpanKind.CLIENT

    def _map_speech_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map speech (text-to-speech) span data to OTel format."""
        name = "audio.speech"

        if span_data.model:
            attributes["audio.model"] = span_data.model
            name = f"audio.speech: {span_data.model}"

        return name, attributes, self._SpanKind.CLIENT

    def _map_mcp_tools_span(
        self, span_data: Any, attributes: dict[str, Any]
    ) -> tuple[str, dict[str, Any], Any]:
        """Map MCP tools listing span data to OTel format."""
        name = "mcp.list_tools"
        server = getattr(span_data, "server", None)

        if server:
            attributes["mcp.server"] = server
            name = f"mcp.list_tools: {server}"

        return name, attributes, self._SpanKind.CLIENT

    def _update_span_with_final_data(self, otel_span: Any, span_data: Any) -> None:
        """Update an OTel span with final data available at span end.

        Args:
            otel_span: The OTel span to update.
            span_data: The SDK SpanData object with final values.
        """
        span_type = span_data.type

        if span_type == "generation":
            self._update_generation_span(otel_span, span_data)
        elif span_type == "function":
            self._update_function_span(otel_span, span_data)
        elif span_type == "transcription":
            output = getattr(span_data, "output", None)
            if output:
                otel_span.set_attribute(
                    "audio.transcription.output", _truncate_string(output, 2048)
                )
        elif span_type == "mcp_tools":
            result = getattr(span_data, "result", None)
            if result:
                otel_span.set_attribute("mcp.tools.count", len(result))
                otel_span.set_attribute("mcp.tools.list", json.dumps(result[:20]))

    def _update_generation_span(self, otel_span: Any, span_data: Any) -> None:
        """Update generation span with final usage metrics and output."""
        usage = getattr(span_data, "usage", None)
        if usage:
            if "input_tokens" in usage:
                otel_span.set_attribute(
                    f"{_ATTR_PREFIX_GEN_AI}.usage.input_tokens", usage["input_tokens"]
                )
            if "output_tokens" in usage:
                otel_span.set_attribute(
                    f"{_ATTR_PREFIX_GEN_AI}.usage.output_tokens", usage["output_tokens"]
                )
            if "total_tokens" in usage:
                otel_span.set_attribute(
                    f"{_ATTR_PREFIX_GEN_AI}.usage.total_tokens", usage["total_tokens"]
                )

        # Add truncated input/output preview for debugging
        input_data = getattr(span_data, "input", None)
        if input_data:
            try:
                input_preview = json.dumps(list(input_data)[:3])
                otel_span.set_attribute(
                    f"{_ATTR_PREFIX_GEN_AI}.input.preview", _truncate_string(input_preview, 1024)
                )
            except Exception:
                pass

        output_data = getattr(span_data, "output", None)
        if output_data:
            try:
                output_preview = json.dumps(list(output_data)[:3])
                otel_span.set_attribute(
                    f"{_ATTR_PREFIX_GEN_AI}.output.preview", _truncate_string(output_preview, 1024)
                )
            except Exception:
                pass

    def _update_function_span(self, otel_span: Any, span_data: Any) -> None:
        """Update function span with output.

        Uses OTel GenAI semantic convention for tool call result.
        """
        output = getattr(span_data, "output", None)
        if output:
            otel_span.set_attribute(
                f"{_ATTR_PREFIX_GEN_AI}.tool.call.result",
                _truncate_string(str(output), self._config.max_attribute_length),
            )

    def _add_span_events(self, otel_span: Any, span_data: Any) -> None:
        """Add span events for content capture based on span type and config.

        Events provide timeline visibility within spans and handle large payloads
        better than attributes due to separate size limits.

        Args:
            otel_span: The OTel span to add events to.
            span_data: The SDK SpanData object with content.
        """
        span_type = span_data.type

        if span_type == "generation":
            self._add_generation_events(otel_span, span_data)
        elif span_type == "function":
            self._add_function_events(otel_span, span_data)
        elif span_type == "guardrail":
            self._add_guardrail_event(otel_span, span_data)
        elif span_type == "handoff":
            self._add_handoff_event(otel_span, span_data)

    def _add_generation_events(self, otel_span: Any, span_data: Any) -> None:
        """Add prompt and completion events to generation span."""
        input_data = getattr(span_data, "input", None)
        if input_data and self._config.capture_prompts:
            try:
                content = json.dumps(list(input_data))
            except (TypeError, ValueError):
                content = str(input_data)
            content = self._apply_content_filter(content, "prompt")
            otel_span.add_event(
                "gen_ai.content.prompt",
                attributes={
                    f"{_ATTR_PREFIX_GEN_AI}.prompt": _truncate_string(
                        content, self._config.max_event_length
                    )
                },
            )

        output_data = getattr(span_data, "output", None)
        if output_data and self._config.capture_completions:
            try:
                content = json.dumps(list(output_data))
            except (TypeError, ValueError):
                content = str(output_data)
            content = self._apply_content_filter(content, "completion")
            otel_span.add_event(
                "gen_ai.content.completion",
                attributes={
                    f"{_ATTR_PREFIX_GEN_AI}.completion": _truncate_string(
                        content, self._config.max_event_length
                    )
                },
            )

    def _add_function_events(self, otel_span: Any, span_data: Any) -> None:
        """Add input and output events to function/tool span."""
        input_data = getattr(span_data, "input", None)
        if input_data and self._config.capture_tool_inputs:
            content = self._apply_content_filter(str(input_data), "tool_input")
            otel_span.add_event(
                "gen_ai.tool.input",
                attributes={
                    f"{_ATTR_PREFIX_GEN_AI}.tool.call.arguments": _truncate_string(
                        content, self._config.max_event_length
                    )
                },
            )

        output_data = getattr(span_data, "output", None)
        if output_data and self._config.capture_tool_outputs:
            content = self._apply_content_filter(str(output_data), "tool_output")
            otel_span.add_event(
                "gen_ai.tool.output",
                attributes={
                    f"{_ATTR_PREFIX_GEN_AI}.tool.call.result": _truncate_string(
                        content, self._config.max_event_length
                    )
                },
            )

    def _add_guardrail_event(self, otel_span: Any, span_data: Any) -> None:
        """Add guardrail evaluated event."""
        otel_span.add_event(
            "guardrail.evaluated",
            attributes={
                "guardrail.name": span_data.name,
                "guardrail.triggered": span_data.triggered,
            },
        )

    def _add_handoff_event(self, otel_span: Any, span_data: Any) -> None:
        """Add handoff executed event."""
        from_agent = getattr(span_data, "from_agent", None) or "unknown"
        to_agent = getattr(span_data, "to_agent", None) or "unknown"
        otel_span.add_event(
            "handoff.executed",
            attributes={
                "handoff.from": from_agent,
                "handoff.to": to_agent,
            },
        )

    def _add_baggage_attributes(self, otel_span: Any) -> None:
        """Read baggage from OpenTelemetry context and add as span attributes.

        Baggage enables context propagation across services. The application must
        set baggage values upstream using the OpenTelemetry baggage API:

            from opentelemetry import baggage, context
            ctx = baggage.set_baggage("user.id", user_id)
            with context.attach(ctx):
                await Runner.run(agent, input)

        Args:
            otel_span: The OpenTelemetry span to add baggage attributes to.
        """
        if not self._config.baggage_keys:
            return

        try:
            for key in self._config.baggage_keys:
                value = self._baggage.get_baggage(key)
                if value:
                    # Use the baggage key as the attribute name
                    otel_span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Failed to read baggage: {e}")

    def _apply_content_filter(self, content: str, context: str) -> str:
        """Apply content filter if configured.

        Args:
            content: The content to filter.
            context: The context type (e.g., "prompt", "completion", "tool_input").

        Returns:
            The filtered content, or original content if no filter or filter fails.
        """
        if self._config.content_filter is not None:
            try:
                return self._config.content_filter(content, context)
            except Exception as e:
                logger.warning(f"Content filter failed for {context}: {e}")
        return content

    def _record_span_metrics(self, span_data: Any, error: Any) -> None:
        """Record metrics based on span type.

        Args:
            span_data: The SDK SpanData object.
            error: Optional error object if the span had an error.
        """
        if not self._enable_metrics:
            return

        span_type = span_data.type

        if span_type == "generation":
            # Record token usage metrics
            usage = getattr(span_data, "usage", None)
            model = getattr(span_data, "model", None)
            if usage:
                self._record_token_usage(usage, model)

        elif span_type == "function":
            # Record tool invocation metric
            tool_name = getattr(span_data, "name", "unknown")
            self._record_tool_invocation(tool_name)

        elif span_type == "handoff":
            # Record handoff metric
            from_agent = getattr(span_data, "from_agent", None) or "unknown"
            to_agent = getattr(span_data, "to_agent", None) or "unknown"
            self._record_handoff(from_agent, to_agent)

        elif span_type == "guardrail":
            # Record guardrail trigger metric
            guardrail_name = getattr(span_data, "name", "unknown")
            triggered = getattr(span_data, "triggered", False)
            self._record_guardrail_trigger(guardrail_name, triggered)

    # --- Metrics Recording Methods ---

    def _record_token_usage(
        self, usage: dict[str, Any], model: str | None, duration_s: float | None = None
    ) -> None:
        """Record token usage metrics with required attributes.

        Args:
            usage: Dictionary containing token counts (input_tokens, output_tokens).
            model: The model name for the request.
            duration_s: Optional duration in seconds for the operation.
        """
        if not self._enable_metrics or self._token_histogram is None:
            return

        base_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
        }
        if model:
            base_attrs["gen_ai.request.model"] = model

        try:
            if "input_tokens" in usage and usage["input_tokens"] is not None:
                self._token_histogram.record(
                    usage["input_tokens"],
                    attributes={**base_attrs, "gen_ai.token.type": "input"},
                )

            if "output_tokens" in usage and usage["output_tokens"] is not None:
                self._token_histogram.record(
                    usage["output_tokens"],
                    attributes={**base_attrs, "gen_ai.token.type": "output"},
                )

            # Record duration if provided
            if duration_s is not None and self._duration_histogram is not None:
                self._duration_histogram.record(duration_s, attributes=base_attrs)

        except Exception as e:
            logger.warning(f"Failed to record token usage metrics: {e}")

    def _record_operation_duration(self, duration_s: float, model: str | None) -> None:
        """Record operation duration metric.

        Args:
            duration_s: Duration in seconds.
            model: The model name for the request.
        """
        if not self._enable_metrics or self._duration_histogram is None:
            return

        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
        }
        if model:
            attrs["gen_ai.request.model"] = model

        try:
            self._duration_histogram.record(duration_s, attributes=attrs)
        except Exception as e:
            logger.warning(f"Failed to record operation duration metric: {e}")

    def _record_tool_invocation(self, tool_name: str) -> None:
        """Record tool invocation metric.

        Args:
            tool_name: Name of the tool that was invoked.
        """
        if not self._enable_metrics or self._tool_counter is None:
            return

        try:
            self._tool_counter.add(1, attributes={"gen_ai.tool.name": tool_name})
        except Exception as e:
            logger.warning(f"Failed to record tool invocation metric: {e}")

    def _record_handoff(self, from_agent: str, to_agent: str) -> None:
        """Record handoff metric.

        Args:
            from_agent: Name of the source agent.
            to_agent: Name of the target agent.
        """
        if not self._enable_metrics or self._handoff_counter is None:
            return

        try:
            self._handoff_counter.add(
                1, attributes={"agent.handoff.from": from_agent, "agent.handoff.to": to_agent}
            )
        except Exception as e:
            logger.warning(f"Failed to record handoff metric: {e}")

    def _record_guardrail_trigger(self, guardrail_name: str, triggered: bool) -> None:
        """Record guardrail trigger metric.

        Args:
            guardrail_name: Name of the guardrail.
            triggered: Whether the guardrail was triggered.
        """
        if not self._enable_metrics or self._guardrail_counter is None:
            return

        # Only count if actually triggered
        if not triggered:
            return

        try:
            self._guardrail_counter.add(1, attributes={"agent.guardrail.name": guardrail_name})
        except Exception as e:
            logger.warning(f"Failed to record guardrail trigger metric: {e}")

    def _record_error(self, error_type: str, span_type: str) -> None:
        """Record error metric.

        Args:
            error_type: Type or category of the error.
            span_type: The type of span where the error occurred.
        """
        if not self._enable_metrics or self._error_counter is None:
            return

        try:
            self._error_counter.add(
                1, attributes={"error.type": error_type, "agent.span.type": span_type}
            )
        except Exception as e:
            logger.warning(f"Failed to record error metric: {e}")


def _safe_attribute_value(value: Any) -> str | int | float | bool:
    """Convert a value to a safe OTel attribute type.

    OpenTelemetry attributes must be primitive types (str, int, float, bool)
    or sequences thereof. This function converts complex types to JSON strings.

    Args:
        value: The value to convert.

    Returns:
        A primitive type suitable for OTel attributes.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _truncate_string(value: str, max_length: int = 4096) -> str:
    """Truncate a string to a maximum length.

    Args:
        value: The string to truncate.
        max_length: Maximum length (default 4096).

    Returns:
        The truncated string with ellipsis if truncated.
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def create_resource(
    service_name: str,
    service_version: str | None = None,
    additional_attributes: dict[str, str] | None = None,
) -> Any:
    """Create an OpenTelemetry Resource with recommended attributes for agent services.

    This helper creates a Resource with standard attributes for agent-based services,
    including telemetry SDK information and optional custom attributes.

    Args:
        service_name: The name of the service (maps to service.name).
        service_version: Optional version of the service (maps to service.version).
        additional_attributes: Optional dict of additional resource attributes.

    Returns:
        An OpenTelemetry Resource object configured with the provided attributes.

    Raises:
        ImportError: If OpenTelemetry SDK packages are not installed.

    Example:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from openai_agents_opentelemetry import create_resource

        resource = create_resource(
            service_name="my-agent-service",
            service_version="1.0.0",
            additional_attributes={"deployment.environment": "production"},
        )

        provider = TracerProvider(resource=resource)
        ```
    """
    try:
        from opentelemetry.sdk.resources import Resource
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry SDK is required for create_resource. "
            "Install it with: pip install opentelemetry-sdk"
        ) from e

    # Try to get agents SDK version
    try:
        from agents import __version__ as agents_version
    except ImportError:
        agents_version = "unknown"

    attributes: dict[str, str] = {
        "service.name": service_name,
        "telemetry.sdk.name": "openai-agents-opentelemetry",
        "telemetry.sdk.version": __version__,
        "telemetry.sdk.language": "python",
        "agent.sdk.name": "openai-agents",
        "agent.sdk.version": agents_version,
    }

    if service_version:
        attributes["service.version"] = service_version

    if additional_attributes:
        attributes.update(additional_attributes)

    return Resource.create(attributes)
