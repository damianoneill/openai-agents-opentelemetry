# openai-agents-opentelemetry

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-opentelemetry.svg)](https://pypi.org/project/openai-agents-opentelemetry/)
[![CI](https://github.com/damianoneill/openai-agents-opentelemetry/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-opentelemetry/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-opentelemetry/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-opentelemetry/actions/workflows/compatibility.yml)

OpenTelemetry tracing processor for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

Bridges agent traces to any OTLP-compatible backend (Jaeger, Datadog, Honeycomb, Grafana Tempo, Langfuse, etc.).

## Installation

```bash
pip install openai-agents-opentelemetry
```

## Quick Start

```python
from agents import Agent, Runner, add_trace_processor
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

# Create and register the OpenTelemetry processor
otel_processor = OpenTelemetryTracingProcessor()
add_trace_processor(otel_processor)

# Now all agent traces will be exported to your configured OTel backend
agent = Agent(name="Assistant", instructions="You are helpful.")
result = await Runner.run(agent, "Hello!")
```

## Using OpenTelemetry Only (No OpenAI Backend)

If you want traces to go only to your OpenTelemetry backend:

```python
from agents import set_trace_processors
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

# Replace the default processor entirely
otel_processor = OpenTelemetryTracingProcessor()
set_trace_processors([otel_processor])
```

## Span Mapping

The processor maps SDK spans to OpenTelemetry spans following [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| SDK Span Type | OTel Span Name            | Key Attributes                                                                            |
| ------------- | ------------------------- | ----------------------------------------------------------------------------------------- |
| Agent         | `agent: {name}`           | `agent.name`, `agent.tools`, `agent.handoffs`                                             |
| Generation    | `chat {model}`            | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.*` |
| Function      | `execute_tool {name}`     | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`               |
| Handoff       | `handoff: {from} -> {to}` | `agent.handoff.from`, `agent.handoff.to`                                                  |
| Guardrail     | `guardrail: {name}`       | `agent.guardrail.triggered`                                                               |
| Response      | `gen_ai.response`         | `gen_ai.response.id`, `gen_ai.response.model`                                             |

## Configuration

### Content Capture Configuration

Control what content is captured for privacy and compliance using `ProcessorConfig`:

```python
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

config = ProcessorConfig(
    capture_prompts=True,        # Capture prompt content as span events
    capture_completions=True,    # Capture completion content as span events
    capture_tool_inputs=True,    # Capture tool input arguments
    capture_tool_outputs=True,   # Capture tool output results
    max_attribute_length=4096,   # Max length for span attributes
    max_event_length=8192,       # Max length for span event attributes
    baggage_keys=["user.id", "session.id"],  # Propagate context to spans
)

processor = OpenTelemetryTracingProcessor(config=config)
```

### Content Filtering / PII Redaction

Apply custom filtering to redact sensitive data before capture:

```python
import re
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

def redact_pii(content: str, context: str) -> str:
    """Custom PII redaction callback."""
    # Redact SSNs
    content = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", content)
    # Redact email addresses
    content = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL REDACTED]", content)
    return content

config = ProcessorConfig(
    capture_prompts=True,
    capture_completions=True,
    content_filter=redact_pii,
)

processor = OpenTelemetryTracingProcessor(config=config)
```

### OpenTelemetry SDK Configuration

The processor uses the globally configured OpenTelemetry `TracerProvider`. Configure it as you normally would:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Then add the Agents SDK processor
from agents import add_trace_processor
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

add_trace_processor(OpenTelemetryTracingProcessor())
```

### Resource Helper

Use the `create_resource` helper for standard resource attributes:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from openai_agents_opentelemetry import create_resource

resource = create_resource(
    service_name="my-agent-service",
    service_version="1.0.0",
    additional_attributes={"deployment.environment": "production"},
)

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
```

## Span Events

The processor automatically adds span events for content capture (controlled by `ProcessorConfig`):

| Span Type  | Event Name                  | Attributes                              |
| ---------- | --------------------------- | --------------------------------------- |
| Generation | `gen_ai.content.prompt`     | `gen_ai.prompt`                         |
| Generation | `gen_ai.content.completion` | `gen_ai.completion`                     |
| Function   | `gen_ai.tool.input`         | `gen_ai.tool.call.arguments`            |
| Function   | `gen_ai.tool.output`        | `gen_ai.tool.call.result`               |
| Guardrail  | `guardrail.evaluated`       | `guardrail.name`, `guardrail.triggered` |
| Handoff    | `handoff.executed`          | `handoff.from`, `handoff.to`            |

## Metrics

Enable metrics collection for token usage, operation duration, and agent-specific counters:

```python
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

# Enable metrics collection
processor = OpenTelemetryTracingProcessor(enable_metrics=True)
```

### Standard OTel GenAI Metrics

| Metric Name                        | Type      | Unit      | Description                      |
| ---------------------------------- | --------- | --------- | -------------------------------- |
| `gen_ai.client.token.usage`        | Histogram | `{token}` | Token consumption (input/output) |
| `gen_ai.client.operation.duration` | Histogram | `s`       | LLM call duration                |

### Agent-Specific Metrics

| Metric Name                | Type    | Unit           | Description                      |
| -------------------------- | ------- | -------------- | -------------------------------- |
| `agent.tool.invocations`   | Counter | `{invocation}` | Tool/function call count by name |
| `agent.handoffs`           | Counter | `{handoff}`    | Agent handoff count              |
| `agent.guardrail.triggers` | Counter | `{trigger}`    | Guardrail trigger count by name  |
| `agent.errors`             | Counter | `{error}`      | Error count by type              |

### Configuring Metrics with OpenTelemetry SDK

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Configure metrics export
exporter = OTLPMetricExporter(endpoint="http://localhost:4317")
reader = PeriodicExportingMetricReader(exporter)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Then enable metrics in the processor
from agents import add_trace_processor
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor

add_trace_processor(OpenTelemetryTracingProcessor(enable_metrics=True))
```

## Context Propagation (Baggage)

Propagate context like user IDs or session IDs across all agent spans using OpenTelemetry baggage:

```python
from opentelemetry import baggage, context
from openai_agents_opentelemetry import OpenTelemetryTracingProcessor, ProcessorConfig

# Configure which baggage keys to read and add as span attributes
config = ProcessorConfig(
    baggage_keys=["user.id", "session.id", "tenant.id"]
)
processor = OpenTelemetryTracingProcessor(config=config)

# Set baggage before running agents
ctx = baggage.set_baggage("user.id", "user-123")
ctx = baggage.set_baggage("session.id", "session-456", context=ctx)

with context.attach(ctx):
    # All spans created during this agent run will have user.id and session.id attributes
    result = await Runner.run(agent, "Hello!")
```

This enables filtering traces by user, session, or tenant in your observability backend.

## Compatibility

This package is tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

## License

MIT
