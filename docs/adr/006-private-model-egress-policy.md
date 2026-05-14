# ADR-006: Private Model Egress Policy

**Status:** Accepted
**Owner:** Shaiv Pandya
**Date:** 2026-05-14
**Revisit trigger:** Regulatory requirement to prevent private data from reaching external AI providers; or a provider changes data retention/training policies.

## Context

The AI agent, thesis generation, and report workflows send prompts containing portfolio data, research notes, and financial analysis to external LLM providers (Anthropic, OpenAI, Google). This data may include position sizes, P&L, investment theses, and proprietary research — all classified as private by the DLP system.

The model gateway classifies payload sensitivity and currently operates in `allow_with_warning` mode: private data is allowed to egress to external providers, but every call is logged with its sensitivity classification, DLP findings, and a warning-level audit record.

## Decision

**Allow private data egress with warnings** (`allow_with_warning`). The system logs and classifies all model calls but does not block private data from reaching external providers. A `deny` mode is available as a configuration option but is not the default.

This is acceptable because:
- The tool is for personal use only (ADR-001, ADR-004).
- Major LLM providers have data processing agreements and do not train on API inputs.
- The owner accepts the risk of sending private financial data to external providers.
- Full audit trails exist for every egress decision.

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Deny by default (block private egress) | Maximum data protection | Agent cannot answer portfolio questions without a local model |
| Allow with warning (current) | Full agent functionality, audit trail | Private data leaves the owner's infrastructure |
| Local-only models | No data egress | Inferior model quality, GPU infrastructure required |
| Provider allowlist per sensitivity | Granular control | Complexity, maintenance burden |

## Risks

- Provider data handling policies may change.
- A data breach at a provider could expose private financial data.
- If the tool becomes multi-user (ADR-001 changes), per-user consent for data egress would be needed.

## References

- [agent_governance.py](../../../api/agent_governance.py) — model gateway implementation
- [llm_settings.py](../../../api/llm_settings.py) — gateway policy configuration
- SHA-17 (harden egress mode) adds the `deny` option and test coverage.
