# OpenTelemetry Enhancement Roadmap

Enhancements for `openai-agents-opentelemetry`, organised by phase.

> **Stability Notice**: The OpenTelemetry GenAI Semantic Conventions are currently in **Development** status.
> This library follows the latest experimental conventions and will be updated as they stabilize.
> See the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for details.

---

## Completed Phases

### Phase 1 - Instrumentation Scope Versioning ✅

Added version and schema URL to the tracer for proper instrumentation identification.

### Phase 2 - GenAI Semantic Conventions ✅

Aligned span names and attributes with OpenTelemetry Semantic Conventions for GenAI:

- Generation spans: `chat {model}` with `gen_ai.*` attributes
- Function spans: `execute_tool {name}` with `gen_ai.tool.*` attributes
- Standard operation and provider attributes

### Phase 3 - Configurable Content Capture ✅

Added `ProcessorConfig` for controlling content capture:

- Toggle capture of prompts, completions, tool inputs/outputs
- Size limits for attributes and events
- Content filter callback for PII redaction
- Span events for content capture
- `create_resource` helper for standard resource attributes

### Phase 4 - Metrics Support ✅

Added optional metrics collection (`enable_metrics=True`):

- `gen_ai.client.token.usage` - Token consumption histogram
- `gen_ai.client.operation.duration` - LLM call duration histogram
- `agent.tool.invocations` - Tool call counter
- `agent.handoffs` - Agent handoff counter
- `agent.guardrail.triggers` - Guardrail trigger counter
- `agent.errors` - Error counter by type

### Phase 5 - Baggage Support ✅

Added context propagation via OpenTelemetry baggage:

- Configure `baggage_keys` in `ProcessorConfig`
- Automatically reads baggage and adds as span attributes
- Enables propagation of user.id, session.id, tenant.id across agent spans

---

## Future Considerations (v1.0.0+)

### Logging Correlation

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

### Sampling Configuration

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

| Phase   | Description                      | Status      |
| ------- | -------------------------------- | ----------- |
| Phase 1 | Instrumentation Scope Versioning | ✅ Complete |
| Phase 2 | GenAI Semantic Conventions       | ✅ Complete |
| Phase 3 | Configurable Content Capture     | ✅ Complete |
| Phase 4 | Metrics Support                  | ✅ Complete |
| Phase 5 | Baggage Support                  | ✅ Complete |
| Future  | Logging Correlation, Sampling    | Planned     |

---

## Contributing

To work on future enhancements:

1. Check [GitHub Issues](https://github.com/damianoneill/openai-agents-opentelemetry/issues) for existing discussions
2. Open an issue to discuss your approach
3. See the [Contributing Guide](../CONTRIBUTING.md)

## References

- [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenTelemetry GenAI Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry GenAI Metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
