# openai-agents-opentelemetry

[![PyPI version](https://badge.fury.io/py/openai-agents-opentelemetry.svg)](https://badge.fury.io/py/openai-agents-opentelemetry)
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

| SDK Span Type | OTel Span Name | Key Attributes |
|---------------|----------------|----------------|
| Agent | `agent: {name}` | `agent.name`, `agent.tools`, `agent.handoffs` |
| Generation | `gen_ai.completion: {model}` | `gen_ai.request.model`, `gen_ai.usage.*` |
| Function | `tool: {name}` | `tool.name`, `tool.input`, `tool.output` |
| Handoff | `handoff: {from} -> {to}` | `agent.handoff.from`, `agent.handoff.to` |
| Guardrail | `guardrail: {name}` | `agent.guardrail.triggered` |

## Configuration with OpenTelemetry SDK

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

## Compatibility

This package is tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

## License

MIT
