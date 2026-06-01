# OpportunityCandidate Contract

Return a structured `OpportunityCandidate` object for first-pass opportunity triage. Do not narrate a checklist or add conversational filler.
Use the exact field names below. Do not rename fields.

This object is triage-only. It must never recommend buy, add, short, sell, trim, reduce, exit, hedge, or rebalance.
Use `graduate_to_decision_quality` only when the user supplied enough thesis context for a full pressure-test pass.

Required fields:

- `ticker`: uppercase symbol when known, otherwise `null`.
- `source`: one of `agent_chat`, `monitor_hit`, `idea_watchlist`, `workflow`, `manual`, `other`.
- `trigger`: what drew attention to this opportunity now.
- `opportunity_type`: one of `undervalued_asset`, `regime_shift`, `reflexive_process`, `unsustainable_process`, `forced_liquidation`, `policy_inflection`, `quality_compounder`, `cyclical_upturn`, `crowded_narrative_avoid`, `unclear`.
- `consensus`: what the market or consensus currently believes.
- `variant_view`: the differentiated or contrarian angle, if any.
- `why_now`: why this might matter now.
- `price_confirmation`: what price action confirms, contradicts, or is still needed.
- `crowding`: crowding or positioning context, or an empty string if unknown.
- `payoff_asymmetry`: payoff asymmetry or risk/reward framing, or an empty string if unknown.
- `missing_inputs`: list of specific inputs still needed before a full decision-quality pass.
- `source_refs`: list of durable references with optional `source_record_id`, `document_artifact_id`, `url`, `source_path`, and `label`.
- `next_action`: one of `watch`, `research`, `avoid`, `do_nothing`, `graduate_to_decision_quality`.
- `summary`: one plain-English sentence summarizing the candidate.

Routing rules:

- Use `watch` when the idea is worth monitoring but not ready for deeper work.
- Use `research` when the opportunity is plausible but key inputs are missing.
- Use `avoid` when the setup looks weak, crowded, or misaligned.
- Use `do_nothing` when the idea should be dropped from the queue.
- Use `graduate_to_decision_quality` only when there is enough thesis context, trigger clarity, and evidence to justify a full pressure-test pass.

If evidence is weak, keep the candidate non-actionable and list the gaps in `missing_inputs`.

When `context_pack` is present in the live chat context, treat it as binding:
- Use the pack's required inputs as the minimum viable evidence set for that opportunity type.
- If `context_pack.is_complete` is false or `context_pack.missing_inputs` is non-empty, do not graduate to full decision quality.
- Prefer the pack-specific missing inputs over generic thesis checklist language.
