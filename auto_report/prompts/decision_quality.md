# Decision Quality Contract

Return a structured `decision_quality` object. Do not narrate a checklist or add conversational filler.

The object must make the decision testable:

- `simple_thesis`: one plain-English sentence.
- `mispricing`: what consensus believes, what the variant view is, what is priced, and why consensus is wrong.
- `catalyst_or_reason_now`: the event or condition, expected timeframe, why now, and source evidence.
- `invalidation`: observable, metric/event, threshold, timeframe, and implication. Vague statements like "if the thesis is wrong" are invalid.
- `evidence_for` and `evidence_against`: concrete claims with support and source references.
- `price_action_read`: what price did, what it implies, whether it confirms the thesis, and what data is missing.
- `actionability.status`: one of `actionable`, `missing_inputs`, `blocked_by_policy`, `watch_only`, `do_nothing`.
- `recommended_action`: one of `buy`, `add`, `short`, `sell`, `trim`, `reduce`, `exit`, `hedge`, `rebalance`, `hold`, `watch`, `research`, `avoid`, `do_nothing`.
- `conviction`: `level` is 1-5, `max_level` is always 5, and conviction is not the same thing as confidence.
- `sizing_context`: include prose context and structured `sizing_delta` with direction, amount, unit, basis, and condition.
- `trade_after_trade`: what to do if right, what to do if wrong, and the next review trigger.

If inputs are missing, put them in `actionability.missing_inputs`. Do not invent facts to make a decision look complete.
