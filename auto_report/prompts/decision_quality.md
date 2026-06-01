# Decision Quality Contract

Return a structured `decision_quality` object. Do not narrate a checklist or add conversational filler.
Use the exact field names below. Do not rename fields.

The object must make the decision testable.

Before setting `recommended_action` or `actionability.status`, separate three layers explicitly in the reasoning (without adding new fields):

- **Asset quality**: durability of the business, balance sheet, franchise, or instrument.
- **Thesis quality**: whether the variant view is coherent, evidence-backed, and falsifiable.
- **Trade quality**: whether there is enough mispricing, timing, payoff asymmetry, risk control, and source confidence to act now.

A high-quality asset or plausible thesis does **not** justify `actionable` unless trade quality clears the bar. Strong asset quality with weak trade quality should map to `watch_only`, `missing_inputs`, or `do_nothing` — not a lazy buy/add/short.

Required trade-quality checks before any actionable stance:

- **Reason-now / catalyst**: populated in `catalyst_or_reason_now.why_now` and `event_or_condition`.
- **Variant view**: populated in `mispricing.variant_view` and `why_consensus_is_wrong`.
- **Price confirmation**: reflected in `price_action_read` with honest `confirms_thesis` and `interpretation`.
- **Payoff asymmetry**: bounded downside and meaningful upside must be visible in evidence, invalidation, and sizing context.

Examples:

- Quality compounder + extended valuation or broken entry → `watch`/`research`, not `buy`/`add`.
- Cheap valuation + no catalyst or price confirmation → `research`/`watch`, not `buy`.
- Crowded consensus + thin variant view → `avoid`/`watch`/`do_nothing`, not momentum-chasing action.

Field contract:

- `simple_thesis`: one plain-English sentence.
- `opportunity_type`: one of `undervalued_asset`, `regime_shift`, `reflexive_process`, `unsustainable_process`, `forced_liquidation`, `policy_inflection`, `quality_compounder`, `cyclical_upturn`, `crowded_narrative_avoid`, `unclear`.
- `mispricing`: use `consensus_view`, `variant_view`, `pricing_evidence`, `why_consensus_is_wrong`.
- `catalyst_or_reason_now`: use `event_or_condition`, `expected_timeframe`, `why_now`, `source_evidence`.
- `invalidation`: use `observable`, `metric_or_event`, `threshold`, `timeframe`, `implication`. Vague statements like "if the thesis is wrong" are invalid.
- `evidence_for` and `evidence_against`: each item must use `claim`, `support`, `source_refs`.
- `price_action_read`: use `observed_behavior`, `interpretation`, `confirms_thesis`, `data_needed`. `confirms_thesis` must be `true`, `false`, or `null`, not a sentence.
- `actionability.status`: one of `actionable`, `missing_inputs`, `blocked_by_policy`, `watch_only`, `do_nothing`.
- `recommended_action`: one of `buy`, `add`, `short`, `sell`, `trim`, `reduce`, `exit`, `hedge`, `rebalance`, `hold`, `watch`, `research`, `avoid`, `do_nothing`.
- `recommended_action` is the broad investment decision, not the mechanical trade verb. Use `short` for a bearish thesis or a trade that profits from an asset, currency, credit, or spread complex deteriorating, even if the implementation is buying CDS protection or put options. Use `buy` for a bullish long thesis. Use `add` only when the decision itself is to increase an existing broad position; do not use `add` merely because `sizing_delta.direction` is `increase` or the trade should be pressed. Express press/add/trim sizing in `sizing_delta`.
- `expression`: use `primary`, `instrument_type`, `directness`, `alternatives`, `follow_on`.
- `conviction`: use `level`, `max_level`, `raw_target_weight`, `upgrade_condition`. `level` is 1-5, `max_level` is always 5, and conviction is not the same thing as confidence.
- `confidence`: number from 0 to 1, or `null` only if confidence cannot be estimated.
- `confidence_reason`: plain-English confidence calibration.
- `sizing_context`: use `starting_size`, `add_conditions`, `liquidity_constraints`, `portfolio_constraints`, and `sizing_delta`.
- `sizing_delta`: use `direction`, `amount`, `unit`, `basis`, `condition`.
- `trade_after_trade`: use `if_right`, `if_wrong`, `next_review_trigger`.

`actionability.status` must match the action:

- Use `actionable` only for `buy`, `add`, `short`, `sell`, `trim`, `reduce`, `exit`, `hedge`, or `rebalance` when trade quality clears the bar above. Do not use `actionable` merely because asset quality or thesis quality is strong.
- Use `do_nothing` for `avoid` or `do_nothing`.
- Use `watch_only` for `hold` or `watch`.
- Use `missing_inputs` for `research`.

If inputs are missing, put them in `actionability.missing_inputs`. Do not invent facts to make a decision look complete.
`actionability.missing_inputs` is not only for blocked or research decisions. If a decision is still actionable but important nonblocking inputs are absent, list them anyway and reflect that uncertainty in `confidence_reason` and `sizing_context`.

Common missing inputs to surface include portfolio exposure, ADV/liquidity, borrow cost or short interest for shorts, executed/current size, target-size inputs, valuation comps, next event date, and relative-performance benchmark. Missing inputs do not automatically make a decision non-actionable; they make the remaining uncertainty explicit.
