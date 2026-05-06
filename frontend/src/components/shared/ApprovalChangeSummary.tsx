import type { ApprovalRecord } from "@/lib/api"

type DetailRow = {
  label: string
  value: string
}

type PositionChangeField = {
  field?: unknown
  before?: unknown
  after?: unknown
}

type PositionChange = {
  ticker?: unknown
  change_type?: unknown
  before?: unknown
  after?: unknown
  fields?: unknown
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function humanizeKey(value: string): string {
  return value
    .replace(/[_-]/g, " ")
    .replace(/\./g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, char => char.toUpperCase())
}

function sentenceCase(value: string): string {
  const text = humanizeKey(value)
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : ""
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(value)
}

function formatMoney(value: number, currency?: unknown): string {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: typeof currency === "string" && currency ? currency : "USD",
      maximumFractionDigits: 0,
    }).format(value)
  } catch {
    return formatNumber(value)
  }
}

function formatValue(value: unknown): string {
  if (value == null || value === "") return "-"
  if (typeof value === "boolean") return value ? "Yes" : "No"
  if (typeof value === "number" && Number.isFinite(value)) return formatNumber(value)
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(", ") : "None"
  if (typeof value === "object") {
    const record = asRecord(value)
    const entries = Object.entries(record).filter(([, item]) => item != null && item !== "")
    if (!entries.length) return "-"
    return entries.map(([key, item]) => `${humanizeKey(key)}: ${formatValue(item)}`).join("; ")
  }
  return String(value).replace(/_/g, " ")
}

function formatDateTime(value: unknown): string {
  if (typeof value !== "string" || !value) return formatValue(value)
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter(item => item && typeof item === "object" && !Array.isArray(item)) as Record<string, unknown>[]
    : []
}

function compactJson(value: Record<string, unknown>): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function actionTitle(actionId: string | null | undefined, entityType: string): string {
  switch (actionId) {
    case "update_watch_trigger_check":
      return "Record watch trigger check"
    case "fire_watch_trigger":
      return "Mark watch trigger as fired"
    case "cancel_watch_trigger":
      return "Cancel watch trigger"
    case "update_watch_trigger_definition":
      return "Update watch trigger definition"
    case "create_watch_trigger":
      return "Create watch trigger"
    case "create_action_item":
      return "Create internal action item"
    case "complete_action_item":
      return "Complete internal action item"
    case "dismiss_action_item":
      return "Dismiss internal action item"
    case "change_thesis_status":
      return "Change thesis status"
    case "create_recommendation":
      return "Create recommendation"
    default:
      return sentenceCase(actionId || entityType || "Proposed change")
  }
}

function positionDescriptor(value: unknown): string {
  const record = asRecord(value)
  const bits: string[] = []
  if (record.direction || record.asset) bits.push([record.direction, record.asset].filter(Boolean).map(formatValue).join(" "))
  const quantity = record.quantity ?? record.shares
  if (quantity != null) {
    bits.push(`${formatValue(quantity)} ${record.instrument_type === "future" ? "contracts" : "shares"}`)
  }
  if (record.contract_multiplier != null && record.instrument_type === "future") {
    bits.push(`multiplier ${formatValue(record.contract_multiplier)}`)
  }
  if (record.cost_basis != null) bits.push(`cost basis ${formatValue(record.cost_basis)}`)
  if (record.currency) bits.push(`currency ${formatValue(record.currency)}`)
  if (record.country || record.exchange) {
    bits.push([record.country, record.exchange].filter(Boolean).map(formatValue).join(" / "))
  }
  if (typeof record.notional_base === "number" && Number.isFinite(record.notional_base)) {
    bits.push(`${formatMoney(record.notional_base, record.base_currency)} base notional`)
  }
  if (record.valuation_status && record.valuation_status !== "ok") {
    bits.push(`valuation ${formatValue(record.valuation_status)}`)
  }
  if (record.conviction != null) bits.push(`conviction ${formatValue(record.conviction)}`)
  if (record.contrarian === true) bits.push("contrarian")
  return bits.filter(Boolean).join(", ") || "-"
}

function positionChangeSummary(change: Record<string, unknown>): { summary: string; rows: DetailRow[] } {
  const changes = asRecordArray(change.position_changes) as PositionChange[]
  const countSummary = asRecord(change.position_change_summary)
  const beforeCount = countSummary.before_count
  const afterCount = countSummary.after_count
  const countText =
    typeof beforeCount === "number" && typeof afterCount === "number"
      ? ` Captured book size: ${formatValue(beforeCount)} before, ${formatValue(afterCount)} after.`
      : ""

  if (!changes.length) {
    const proposedRows = asRecordArray(change.positions)
    const proposedCount = proposedRows.length || (Array.isArray(change.positions) ? change.positions.length : null)
    return {
      summary: proposedCount == null
        ? "No structured position list was stored on this approval."
        : `This approval sets the portfolio book to ${formatValue(proposedCount)} proposed positions.`,
      rows: proposedRows.length
        ? proposedRows.map((row, index) => ({
            label: formatValue(row.ticker || `Position ${index + 1}`),
            value: positionDescriptor(row),
          }))
        : proposedCount == null
          ? []
          : [{ label: "Proposed count", value: formatValue(proposedCount) }],
    }
  }

  const rows = changes.map(item => {
    const ticker = formatValue(item.ticker)
    const type = String(item.change_type || "").toLowerCase()
    if (type === "added") return { label: `${ticker} added`, value: positionDescriptor(item.after) }
    if (type === "removed") return { label: `${ticker} removed`, value: positionDescriptor(item.before) }
    const fieldChanges = asRecordArray(item.fields) as PositionChangeField[]
    const value = fieldChanges.length
      ? fieldChanges.map(field => `${humanizeKey(String(field.field || "Field"))}: ${formatValue(field.before)} to ${formatValue(field.after)}`).join("; ")
      : `${positionDescriptor(item.before)} to ${positionDescriptor(item.after)}`
    return { label: `${ticker} updated`, value }
  })

  return {
    summary: `This applies captured portfolio position changes after approval.${countText}`,
    rows,
  }
}

function watchTriggerCheckSummary(change: Record<string, unknown>): { summary: string; rows: DetailRow[] } {
  const result = asRecord(change.result)
  const triggerId = formatValue(change.trigger_id)
  const fired = result.fired === true
  const field = typeof result.field === "string" ? result.field : null
  const operator = typeof result.operator === "string" ? result.operator : null
  const expected = result.expected
  const actual = result.actual
  const evidence = typeof change.evidence === "string" ? change.evidence : typeof result.evidence === "string" ? result.evidence : ""
  const rows: DetailRow[] = [
    { label: "Target", value: `Watch trigger #${triggerId}` },
    { label: "Outcome", value: fired ? "Condition met. The trigger will be marked fired." : "Condition not met. The trigger remains active." },
  ]
  if (result.type) rows.push({ label: "Trigger type", value: formatValue(result.type) })
  if (field) rows.push({ label: "Metric checked", value: humanizeKey(field) })
  if (actual != null) rows.push({ label: "Observed value", value: formatValue(actual) })
  if (operator || expected != null) {
    rows.push({ label: "Trigger rule", value: `${field ? `${humanizeKey(field)} ` : ""}${operator || ""} ${formatValue(expected)}`.trim() })
  }
  if (evidence) rows.push({ label: "Evidence", value: evidence })
  if (result.as_of) rows.push({ label: "Checked at", value: formatDateTime(result.as_of) })
  return {
    summary: fired
      ? `This records that watch trigger #${triggerId} was checked and fired.`
      : `This records that watch trigger #${triggerId} was checked and did not fire.`,
    rows,
  }
}

function createWatchTriggerSummary(change: Record<string, unknown>): { summary: string; rows: DetailRow[] } {
  const rows: DetailRow[] = [
    { label: "Condition", value: formatValue(change.condition) },
    { label: "Trigger type", value: formatValue(change.trigger_type) },
  ]
  if (change.ticker) rows.unshift({ label: "Ticker", value: formatValue(change.ticker) })
  if (change.expires_at) rows.push({ label: "Expires", value: formatDateTime(change.expires_at) })
  if (Object.keys(asRecord(change.definition)).length) rows.push({ label: "Rule definition", value: formatValue(change.definition) })
  return { summary: "This creates a new watch trigger after approval.", rows }
}

function actionItemSummary(change: Record<string, unknown>, actionId?: string | null): { summary: string; rows: DetailRow[] } {
  if (actionId === "complete_action_item") {
    const rows = [
      { label: "Action item", value: `#${formatValue(change.item_id)}` },
      { label: "Resolution note", value: formatValue(change.resolution_note) },
    ]
    return { summary: "This marks an internal action item complete.", rows }
  }
  if (actionId === "dismiss_action_item") {
    return {
      summary: "This dismisses an internal action item.",
      rows: [{ label: "Action item", value: `#${formatValue(change.item_id)}` }],
    }
  }
  const rows: DetailRow[] = [
    { label: "Description", value: formatValue(change.description) },
    { label: "Type", value: formatValue(change.action_type) },
    { label: "Urgency", value: formatValue(change.urgency) },
  ]
  if (change.ticker) rows.unshift({ label: "Ticker", value: formatValue(change.ticker) })
  if (change.recommendation_id) rows.push({ label: "Recommendation", value: `#${formatValue(change.recommendation_id)}` })
  return { summary: "This creates an internal follow-up item for the team.", rows }
}

function recommendationSummary(change: Record<string, unknown>): { summary: string; rows: DetailRow[] } {
  const record = asRecord(change.record)
  const rows: DetailRow[] = []
  for (const key of ["ticker", "action", "conviction", "confidence", "risk_flag", "reason", "summary"]) {
    if (record[key] != null && record[key] !== "") rows.push({ label: humanizeKey(key), value: formatValue(record[key]) })
  }
  return {
    summary: "This stores a recommendation record after approval.",
    rows: rows.length ? rows : genericRows(record),
  }
}

function genericRows(change: Record<string, unknown>): DetailRow[] {
  return Object.entries(change)
    .filter(([, value]) => value != null && value !== "")
    .slice(0, 8)
    .map(([key, value]) => ({
      label: humanizeKey(key),
      value: key.endsWith("_at") || key === "expires_at" ? formatDateTime(value) : formatValue(value),
    }))
}

function proposedChangeSummary(approval: ApprovalRecord): { title: string; summary: string; rows: DetailRow[] } {
  const actionId = approval.action_id
  const change = approval.proposed_change
  const title = actionTitle(actionId, approval.entity_type)
  switch (actionId) {
    case "update_portfolio_positions":
      return { title, ...positionChangeSummary(change) }
    case "update_watch_trigger_check":
      return { title, ...watchTriggerCheckSummary(change) }
    case "create_watch_trigger":
      return { title, ...createWatchTriggerSummary(change) }
    case "create_action_item":
    case "complete_action_item":
    case "dismiss_action_item":
      return { title, ...actionItemSummary(change, actionId) }
    case "create_recommendation":
      return { title, ...recommendationSummary(change) }
    default:
      return {
        title,
        summary: "This applies the proposed internal state change after approval.",
        rows: genericRows(change),
      }
  }
}

export function ApprovalChangeSummary({ approval }: { approval: ApprovalRecord }) {
  const { title, summary, rows } = proposedChangeSummary(approval)
  return (
    <div className="space-y-3">
      <div>
        <p className="text-sm font-semibold text-app">{title}</p>
        <p className="mt-1 text-xs text-muted">{summary}</p>
      </div>
      {rows.length > 0 && (
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-[140px_minmax(0,1fr)]">
          {rows.map(row => (
            <div key={`${row.label}-${row.value}`} className="contents">
              <dt className="text-[11px] font-medium uppercase tracking-wide text-subtle">{row.label}</dt>
              <dd className="min-w-0 break-words text-xs text-app">{row.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <details className="group rounded border border-app bg-[hsl(var(--background-card))]">
        <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-muted">
          Technical details
        </summary>
        <pre className="max-h-56 overflow-auto whitespace-pre-wrap border-t border-app p-3 font-mono text-[11px] text-app">
          {compactJson(approval.proposed_change)}
        </pre>
      </details>
    </div>
  )
}
