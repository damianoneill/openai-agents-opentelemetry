# OpenTelemetry Enhancement Roadmap

Planned enhancements for `openai-agents-opentelemetry`, organised by phase.

> **Stability Notice**: The OpenTelemetry GenAI Semantic Conventions are currently in **Development** status.
> Instrumentations should be aware of the `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable which controls
> convention versioning. This roadmap follows the latest experimental conventions and will be updated as they stabilize.
> See the [transition plan](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for details.

---

## Phase 1 - Foundation (v0.2.0)

### 1. Instrumentation Scope Versioning

**Status**: Planned
**Complexity**: Low
**Dependencies**: None

Add version and schema URL to the tracer for proper instrumentation identification.

**Current:**

```python
self._tracer = trace.get_tracer(tracer_name)
```

**Proposed:**

```python
from . import __version__

# Schema URL indicates which semantic conventions version this instrumentation follows.
# This should be updated when adopting newer semantic convention versions.
# See: https://opentelemetry.io/docs/specs/otel/schemas/
# Current versions: https://github.com/open-telemetry/semantic-conventions/tree/main/schemas
#
# Note: GenAI semantic conventions are in Development status. Update this URL
# when conventions stabilize or when adopting newer experimental versions.
SCHEMA_URL = "https://opentelemetry.io/schemas/1.28.0"

self._tracer = trace.get_tracer(
    tracer_name,
    instrumenting_library_version=__version__,
    schema_url=SCHEMA_URL
)
```

**Note on Schema URL**: The schema URL version should correspond to the OpenTelemetry semantic conventions version being implemented. As of the current roadmap, GenAI semantic conventions are still in development status. The schema URL enables backends to understand and potentially transform telemetry data as conventions evolve. Track updates at the [semantic-conventions repository](https://github.com/open-telemetry/semantic-conventions/releases).

---

## Phase 2 - Enhanced Observability (v0.3.0)

### 2. GenAI Semantic Conventions

**Status**: Planned
**Complexity**: Medium
**Dependencies**: Instrumentation Scope Versioning (#1) should be completed first to ensure schema URL alignment

Better alignment with [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

**Required Attributes (per OTel spec):**

| Attribute               | Description                           | Requirement Level      | Span Type  |
| ----------------------- | ------------------------------------- | ---------------------- | ---------- |
| `gen_ai.operation.name` | Operation type (`chat`, `embeddings`) | **Required**           | Generation |
| `gen_ai.provider.name`  | Always `"openai"`                     | **Required**           | Generation |
| `gen_ai.request.model`  | Model requested                       | Conditionally Required | Generation |

**Recommended Attributes:**

| Attribute                       | Description           | Span Type  |
| ------------------------------- | --------------------- | ---------- |
| `gen_ai.request.temperature`    | Temperature parameter | Generation |
| `gen_ai.request.max_tokens`     | Max tokens parameter  | Generation |
| `gen_ai.request.top_p`          | Top-p parameter       | Generation |
| `gen_ai.request.stop_sequences` | Stop sequences        | Generation |
| `gen_ai.response.model`         | Model that responded  | Generation |
| `gen_ai.response.id`            | Completion ID         | Generation |
| `gen_ai.usage.input_tokens`     | Input token count     | Generation |
| `gen_ai.usage.output_tokens`    | Output token count    | Generation |

**Span Naming Convention:**

Per the OTel spec, span name SHOULD be `{gen_ai.operation.name} {gen_ai.request.model}`:

```python
name = f"chat {span_data.model}"  # e.g., "chat gpt-4"
```

**Example:**

```python
def _map_generation_span(self, span_data: Any, attributes: dict[str, Any]) -> tuple[str, dict[str, Any], Any]:
    kind = self._SpanKind.CLIENT  # LLM call is an outbound request

    # Required attributes (per OTel GenAI semantic conventions)
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
        if "temperature" in config and config["temperature"] is not None:
            attributes[f"{_ATTR_PREFIX_GEN_AI}.request.temperature"] = config["temperature"]
        if "max_tokens" in config and config["max_tokens"] is not None:
            attributes[f"{_ATTR_PREFIX_GEN_AI}.request.max_tokens"] = config["max_tokens"]
        if "top_p" in config and config["top_p"] is not None:
            attributes[f"{_ATTR_PREFIX_GEN_AI}.request.top_p"] = config["top_p"]

    return name, attributes, kind
```

**Opt-In Content Attributes:**

The OTel GenAI conventions define these as Opt-In attributes for capturing full content:

| Attribute                    | Description         | Notes                            |
| ---------------------------- | ------------------- | -------------------------------- |
| `gen_ai.input.messages`      | Chat history/prompt | May be JSON string or structured |
| `gen_ai.output.messages`     | Model responses     | May be JSON string or structured |
| `gen_ai.system_instructions` | System prompt       | Separate from chat history       |

These attributes are disabled by default due to privacy/size concerns. See Phase 3 for configuration options.

---

### 3. Span Events for Milestones

**Status**: Planned
**Complexity**: Medium
**Dependencies**: Should be implemented alongside or after GenAI Semantic Conventions (#2) for consistent naming

> **Design Note**: The OTel GenAI conventions use span attributes (`gen_ai.input.messages`, `gen_ai.output.messages`)
> for content capture. This implementation uses span events as an alternative approach that:
>
> - Handles large payloads better (events have separate size limits)
> - Provides timeline visibility within spans
> - Allows for streaming chunk events
>
> This is a deliberate design choice for the OpenAI Agents SDK use case. Users requiring strict OTel convention
> compliance can disable events and use attribute-based capture via configuration.

**Proposed Events:**

| Span Type  | Event Name                  | Attributes                              |
| ---------- | --------------------------- | --------------------------------------- |
| Generation | `gen_ai.content.prompt`     | `gen_ai.prompt` (truncated)             |
| Generation | `gen_ai.content.completion` | `gen_ai.completion` (truncated)         |
| Function   | `gen_ai.tool.input`         | `gen_ai.tool.call.arguments`            |
| Function   | `gen_ai.tool.output`        | `gen_ai.tool.call.result`               |
| Guardrail  | `guardrail.evaluated`       | `guardrail.name`, `guardrail.triggered` |
| Handoff    | `handoff.executed`          | `handoff.from`, `handoff.to`            |

**Example:**

```python
def _update_generation_span(self, otel_span: Any, span_data: Any) -> None:
    input_data = getattr(span_data, "input", None)
    if input_data and self._config.capture_prompts:
        content = self._apply_content_filter(str(input_data))
        otel_span.add_event(
            "gen_ai.content.prompt",
            attributes={"gen_ai.prompt": _truncate_string(content, self._config.max_event_length)}
        )

    output_data = getattr(span_data, "output", None)
    if output_data and self._config.capture_completions:
        content = self._apply_content_filter(str(output_data))
        otel_span.add_event(
            "gen_ai.content.completion",
            attributes={"gen_ai.completion": _truncate_string(content, self._config.max_event_length)}
        )
```

---

## Phase 3 - Configuration & Resources (v0.4.0)

### 4. Configurable Content Capture

**Status**: Planned
**Complexity**: Medium
**Dependencies**: Should be implemented before or with Span Events (#3) to control what content is captured

Control what content is captured for privacy and compliance.

```python
from typing import Callable, Any

# Type for content filter callback
ContentFilter = Callable[[str, str], str]  # (content, context) -> filtered_content

@dataclass
class ProcessorConfig:
    # Content capture toggles
    capture_prompts: bool = True
    capture_completions: bool = True
    capture_tool_inputs: bool = True
    capture_tool_outputs: bool = True

    # Size limits
    max_attribute_length: int = 4096
    max_event_length: int = 8192

    # Custom content filter callback for redaction/transformation
    # Receives (content, context) where context is e.g., "prompt", "completion", "tool_input"
    # Returns filtered content string
    content_filter: ContentFilter | None = None


# Example: Custom redaction callback
def redact_pii(content: str, context: str) -> str:
    """Custom PII redaction callback."""
    import re
    # Redact SSNs
    content = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", content)
    # Redact email addresses
    content = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL REDACTED]", content)
    return content


processor = OpenTelemetryTracingProcessor(
    config=ProcessorConfig(
        capture_prompts=True,
        capture_completions=True,
        max_attribute_length=1024,
        content_filter=redact_pii,
    )
)
```

**Applying the filter:**

```python
def _apply_content_filter(self, content: str, context: str) -> str:
    """Apply content filter if configured."""
    if self._config.content_filter is not None:
        try:
            return self._config.content_filter(content, context)
        except Exception as e:
            logger.warning(f"Content filter failed for {context}: {e}")
    return content
```

---

### 5. Resource Attributes

**Status**: Planned
**Complexity**: Low
**Dependencies**: None (can be done independently)

Helper for standard resource attributes.

```python
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

resource = Resource.create({
    SERVICE_NAME: "my-agent-service",
    SERVICE_VERSION: "1.0.0",
    "telemetry.sdk.name": "openai-agents-opentelemetry",
    "telemetry.sdk.version": __version__,
    "telemetry.sdk.language": "python",
    "agent.sdk.name": "openai-agents",
    "agent.sdk.version": agents_sdk_version,
})
```

```python
def create_resource(
    service_name: str,
    service_version: str | None = None,
    additional_attributes: dict[str, str] | None = None
) -> Resource:
    """Create a Resource with recommended attributes for agent services."""
    ...
```

---

## Phase 4 - Metrics (v0.5.0)

### 6. Metrics Support

**Status**: Planned
**Complexity**: High
**Dependencies**: GenAI Semantic Conventions (#2) for consistent attribute naming on metrics

**Standard OTel GenAI Metrics:**

| Metric Name                        | Type      | Unit      | Description       |
| ---------------------------------- | --------- | --------- | ----------------- |
| `gen_ai.client.token.usage`        | Histogram | `{token}` | Token consumption |
| `gen_ai.client.operation.duration` | Histogram | `s`       | LLM call duration |

> **Note**: Per OTel GenAI semantic conventions, `gen_ai.client.token.usage` is a **Histogram** (not Counter)
> and requires the `gen_ai.token.type` attribute to distinguish input vs output tokens.

**OpenAI Agents SDK-Specific Metrics:**

These metrics are specific to the OpenAI Agents SDK and provide observability into agent-specific operations
not covered by the standard GenAI conventions:

| Metric Name                | Type    | Unit           | Description                      |
| -------------------------- | ------- | -------------- | -------------------------------- |
| `agent.tool.invocations`   | Counter | `{invocation}` | Tool/function call count by name |
| `agent.handoffs`           | Counter | `{handoff}`    | Agent handoff count              |
| `agent.guardrail.triggers` | Counter | `{trigger}`    | Guardrail trigger count by name  |
| `agent.errors`             | Counter | `{error}`      | Error count by type              |

**Example:**

```python
from opentelemetry import metrics

# Recommended histogram buckets per OTel GenAI conventions
TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]

class OpenTelemetryTracingProcessor(TracingProcessor):
    def __init__(self, tracer_name: str = DEFAULT_TRACER_NAME, enable_metrics: bool = False):
        # ... existing init ...

        if enable_metrics:
            meter = metrics.get_meter(
                tracer_name,
                version=__version__,
                schema_url=SCHEMA_URL
            )

            # Standard OTel GenAI metrics
            self._token_histogram = meter.create_histogram(
                "gen_ai.client.token.usage",
                unit="{token}",
                description="Number of input and output tokens used"
            )
            self._duration_histogram = meter.create_histogram(
                "gen_ai.client.operation.duration",
                unit="s",
                description="Duration of GenAI operations"
            )

            # OpenAI Agents SDK-specific metrics
            self._tool_counter = meter.create_counter(
                "agent.tool.invocations",
                unit="{invocation}",
                description="Number of tool invocations"
            )
            self._handoff_counter = meter.create_counter(
                "agent.handoffs",
                unit="{handoff}",
                description="Number of agent handoffs"
            )
            self._guardrail_counter = meter.create_counter(
                "agent.guardrail.triggers",
                unit="{trigger}",
                description="Number of guardrail triggers"
            )
            self._error_counter = meter.create_counter(
                "agent.errors",
                unit="{error}",
                description="Number of errors by type"
            )

    def _record_token_usage(self, usage: dict, model: str) -> None:
        """Record token usage metrics with required attributes."""
        base_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": model,
        }

        if "input_tokens" in usage:
            self._token_histogram.record(
                usage["input_tokens"],
                attributes={**base_attrs, "gen_ai.token.type": "input"}
            )

        if "output_tokens" in usage:
            self._token_histogram.record(
                usage["output_tokens"],
                attributes={**base_attrs, "gen_ai.token.type": "output"}
            )
```

---

## Phase 5 - Context Propagation (v0.6.0)

### 7. Baggage Support

**Status**: Planned
**Complexity**: Low
**Dependencies**: None

Read OpenTelemetry baggage from the current context and add to spans.

> **Note**: Baggage must be set upstream by the application and requires appropriate propagators
> (e.g., `W3CBaggagePropagator`) to be configured for cross-service propagation.

```python
from opentelemetry import baggage

def on_span_start(self, span: AgentSpan[Any]) -> None:
    # ... existing code ...

    # Read baggage from current OpenTelemetry context
    # Baggage must be set by the application, e.g.:
    #   ctx = baggage.set_baggage("user.id", user_id)
    #   with context.use_context(ctx):
    #       await Runner.run(agent, input)

    user_id = baggage.get_baggage("user.id")
    if user_id:
        otel_span.set_attribute("enduser.id", user_id)

    session_id = baggage.get_baggage("session.id")
    if session_id:
        otel_span.set_attribute("session.id", session_id)

    # Allow custom baggage keys to be configured
    for key in self._config.baggage_keys:
        value = baggage.get_baggage(key)
        if value:
            otel_span.set_attribute(key, value)
```

---

## Future Considerations (v1.0.0+)

### 8. Logging Correlation

**Dependencies**: None

Integrate with OpenTelemetry logging to correlate logs with traces.

> **Note**: The OpenTelemetry Python logging API (`opentelemetry._logs`) uses an underscore prefix
> indicating it may have limited stability. Monitor for API changes.

```python
import logging
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# Set up log provider
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
set_logger_provider(logger_provider)

# Add handler to capture logs with trace context
handler = LoggingHandler(logger_provider=logger_provider)
logging.getLogger().addHandler(handler)
```

---

### 9. Sampling Configuration

**Dependencies**: None

Custom sampling strategies for production use cases:

- Sample all errors at 100%
- Sample successful requests at a configurable rate
- Head-based vs tail-based sampling considerations
- Parent-based sampling for distributed traces

```python
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio, ALWAYS_ON

# Example: Sample 10% of traces, but always sample errors
sampler = ParentBasedTraceIdRatio(0.1)

# For tail-based sampling, consider using the OpenTelemetry Collector
# with the tailsamplingprocessor
```

---

## Timeline Summary

| Phase   | Items                                             | Target  |
| ------- | ------------------------------------------------- | ------- |
| Phase 1 | Instrumentation Scope Versioning                  | v0.2.0  |
| Phase 2 | GenAI Semantic Conventions, Span Events           | v0.3.0  |
| Phase 3 | Configurable Content Capture, Resource Attributes | v0.4.0  |
| Phase 4 | Metrics Support                                   | v0.5.0  |
| Phase 5 | Baggage Support                                   | v0.6.0  |
| Future  | Logging Correlation, Sampling                     | v1.0.0+ |

---

## Dependency Graph

```
Phase 1: Instrumentation Scope Versioning
            │
            ▼
Phase 2: GenAI Semantic Conventions ◄──┐
            │                          │
            ▼                          │ (consistent naming)
         Span Events ──────────────────┘
            │
            ▼
Phase 3: Configurable Content Capture (controls what events capture)
         Resource Attributes (independent)
            │
            ▼
Phase 4: Metrics Support (uses same attribute conventions)
            │
            ▼
Phase 5: Baggage Support (independent)
            │
            ▼
Future:  Logging Correlation, Sampling (independent)
```

---

## Contributing

To work on any of these:

1. Check [GitHub Issues](https://github.com/damianoneill/openai-agents-opentelemetry/issues) for existing discussions
2. Open an issue to discuss your approach
3. See the [Contributing Guide](../CONTRIBUTING.md)

## References

- [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenTelemetry GenAI Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry GenAI Metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry Schema Versions](https://github.com/open-telemetry/semantic-conventions/tree/main/schemas)
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
