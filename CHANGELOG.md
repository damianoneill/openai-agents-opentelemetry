# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


### Added

- Initial release of `openai-agents-opentelemetry`
- `OpenTelemetryTracingProcessor` class that bridges OpenAI Agents SDK traces to OpenTelemetry
- Support for all SDK span types:
  - Agent spans with name, tools, and handoffs attributes
  - Generation spans with model and usage metrics (following GenAI semantic conventions)
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

[Unreleased]: https://github.com/damianoneill/openai-agents-opentelemetry/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/damianoneill/openai-agents-opentelemetry/releases/tag/v0.1.0

## v0.2.0 (2026-01-31)

### BREAKING CHANGE

- Span names and attribute names changed to follow
OpenTelemetry Semantic Conventions for GenAI.

### Feat

- add baggage support for context propagation (Phase 5)
- add metrics support for token usage and agent operations (Phase 4)
- add configurable content capture and span events (Phase 3)
- align with OTel GenAI semantic conventions (Phase 2)
- add instrumentation scope versioning (Phase 1)

## v0.1.1 (2026-01-29)

### Fix

- remove email from package metadata

## v0.1.0 (2026-01-29)

### Feat

- initial release of openai-agents-opentelemetry
