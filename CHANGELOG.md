# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-31

### ⚠ BREAKING CHANGES

- Span names and attribute names changed to follow OpenTelemetry Semantic Conventions for GenAI
  - Generation span names now follow `chat {model}` pattern
  - Function span names now follow `execute_tool {name}` pattern
  - Attributes use `gen_ai.*` namespace per OTel conventions

### Added

- **Phase 1 - Instrumentation Scope Versioning**: Added version and schema URL to tracer for proper instrumentation identification
- **Phase 2 - GenAI Semantic Conventions**: Full alignment with [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
  - Standard span naming: `chat {model}`, `execute_tool {name}`
  - Required attributes: `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`
  - Recommended attributes: temperature, max_tokens, top_p, usage metrics
- **Phase 3 - Configurable Content Capture**: New `ProcessorConfig` class for controlling content capture
  - Toggle capture of prompts, completions, tool inputs/outputs
  - Configurable size limits for attributes and events
  - Content filter callback for PII redaction
  - Span events for content capture (`gen_ai.content.prompt`, `gen_ai.content.completion`, etc.)
  - `create_resource()` helper for standard resource attributes
- **Phase 4 - Metrics Support**: Optional metrics collection via `enable_metrics=True`
  - `gen_ai.client.token.usage` - Token consumption histogram
  - `gen_ai.client.operation.duration` - LLM call duration histogram
  - `agent.tool.invocations` - Tool call counter
  - `agent.handoffs` - Agent handoff counter
  - `agent.guardrail.triggers` - Guardrail trigger counter
  - `agent.errors` - Error counter by type
- **Phase 5 - Baggage Support**: Context propagation via OpenTelemetry baggage
  - Configure `baggage_keys` in `ProcessorConfig`
  - Automatically reads baggage and adds as span attributes
  - Enables propagation of user.id, session.id, tenant.id across agent spans

## [0.1.1] - 2026-01-29

### Fixed

- Remove email from package metadata

## [0.1.0] - 2026-01-29

### Added

- Initial release of `openai-agents-opentelemetry`
- `OpenTelemetryTracingProcessor` class that bridges OpenAI Agents SDK traces to OpenTelemetry
- Support for all SDK span types:
  - Agent spans with name, tools, and handoffs attributes
  - Generation spans with model and usage metrics
  - Function/tool spans with input and output
  - Handoff spans with source and target agent
  - Guardrail spans with triggered status
  - Response, transcription, speech, and MCP tools spans
- Thread-safe span context management for parallel operations
- Explicit parent context propagation (avoids global context issues with overlapping spans)
- Comprehensive error handling and serialization
- 90%+ test coverage
- CI/CD with GitHub Actions:
  - Multi-Python version testing (3.9, 3.11, 3.13)
  - Weekly SDK compatibility checks
  - Automated PyPI publishing on release
- Pre-commit hooks for code quality enforcement
- Conventional commit message enforcement

[Unreleased]: https://github.com/damianoneill/openai-agents-opentelemetry/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/damianoneill/openai-agents-opentelemetry/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/damianoneill/openai-agents-opentelemetry/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/damianoneill/openai-agents-opentelemetry/releases/tag/v0.1.0
