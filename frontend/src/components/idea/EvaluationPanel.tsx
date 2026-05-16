import { CheckCircle2, XCircle } from "lucide-react"

import { Toggle } from "@/components/shared/FormControls"
import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import type {
  AnalyzerRiskFlags,
  AnalyzerRiskParts,
  IdeaAnalyzerContext,
  IdeaEvaluation,
  IdeaFactorScore,
  IdeaMissingInformation,
} from "@/lib/api"
import { formatDate, formatLabel, scoreText } from "@/lib/ideaUtils"

const ACTION_TONE: Record<string, StatusTone> = {
  buy: "success",
  add: "success",
  short: "error",
  sell: "error",
  trim: "warning",
  reduce: "warning",
  exit: "error",
  hedge: "warning",
  rebalance: "warning",
  hold: "neutral",
  watch: "info",
  research: "info",
  avoid: "error",
  do_nothing: "neutral",
}

const GATE_TONE: Record<string, StatusTone> = {
  pass: "success",
  downgraded: "warning",
  blocked: "error",
  invalid: "error",
}

const CANONICAL_FACTORS = [
  "macro_support",
  "industry_attractiveness",
  "business_quality",
  "management_quality",
  "valuation_asymmetry",
  "portfolio_fit",
]

const RISK_FLAG_LABELS: Record<string, string> = {
  short_squeeze_risk: "Squeeze",
  drawdown_risk: "Drawdown",
  contrarian_not_eligible: "Contrarian",
  short_squeeze_data_missing: "Squeeze data",
  drawdown_data_missing: "Drawdown data",
  risk_data_missing: "Risk data",
}

const RISK_FLAG_TONES: Record<string, StatusTone> = {
  short_squeeze_risk: "warning",
  drawdown_risk: "warning",
  contrarian_not_eligible: "warning",
  short_squeeze_data_missing: "neutral",
  drawdown_data_missing: "neutral",
  risk_data_missing: "neutral",
}

export function ActionPill({ action }: { action?: string | null }) {
  if (!action) return <span className="text-sm text-subtle">N/A</span>
  return <StatusBadge tone={ACTION_TONE[action] ?? "neutral"}>{formatLabel(action)}</StatusBadge>
}

export function ConfidencePill({ level, confidence }: { level?: string | null; confidence?: number | null }) {
  const normalized = String(level || "").toLowerCase()
  const tone: StatusTone = normalized === "high" ? "success" : normalized === "medium" ? "warning" : "neutral"
  const confidenceLabel = confidence == null ? "N/A" : `${Math.round(Number(confidence) * 100)}%`
  return (
    <span className="flex flex-wrap items-center gap-2">
      <StatusBadge tone={tone}>{formatLabel(normalized || "low")}</StatusBadge>
      <span className="font-mono text-xs text-subtle">{confidenceLabel}</span>
    </span>
  )
}

function DecisionQualityGatePanel({ evaluation }: { evaluation: IdeaEvaluation }) {
  const gate = evaluation.decision_quality_gate
  if (!gate) return null
  const reasons = Array.isArray(gate.reasons) ? gate.reasons : []
  return (
    <div className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Decision Quality</span>
        <StatusBadge tone={GATE_TONE[String(gate.status)] ?? "neutral"}>{formatLabel(String(gate.status || "unknown"))}</StatusBadge>
        {gate.original_action !== gate.final_action && (
          <span className="text-xs text-subtle">
            {formatLabel(gate.original_action)} to {formatLabel(gate.final_action)}
          </span>
        )}
      </div>
      {reasons.length > 0 && (
        <div className="mt-3 space-y-2">
          {reasons.slice(0, 4).map((reason, index) => (
            <p key={`${reason.code}-${index}`} className="leading-6">
              <span className="font-mono text-xs text-subtle">{reason.code}</span>
              {reason.message ? `: ${reason.message}` : ""}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

function EvaluatorDiagnosticNotice({ evaluation }: { evaluation: IdeaEvaluation }) {
  const diagnostic = evaluation.data_quality?.evaluator
  if (!diagnostic || typeof diagnostic !== "object") return null
  const status = String(diagnostic.status || "").toLowerCase()
  if (!status || status === "ok") return null
  const tone: StatusTone = status === "repaired" ? "warning" : "error"
  const attempts = diagnostic.attempts == null ? null : Number(diagnostic.attempts)
  const detailParts = [
    diagnostic.provider ? String(diagnostic.provider) : "",
    diagnostic.model ? String(diagnostic.model) : "",
    diagnostic.web_search_status ? `search ${formatLabel(String(diagnostic.web_search_status))}` : "",
    attempts != null && Number.isFinite(attempts) ? `${attempts} attempt${attempts === 1 ? "" : "s"}` : "",
  ].filter(Boolean)
  const reason = String(diagnostic.failure_reason || "")
  return (
    <div className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Evaluator Diagnostics</span>
        <StatusBadge tone={tone}>{formatLabel(status)}</StatusBadge>
        {detailParts.length > 0 && <span className="text-xs text-subtle">{detailParts.join(" / ")}</span>}
      </div>
      {reason && <p className="mt-2 leading-6">{reason}</p>}
    </div>
  )
}

function DecisionQualitySummary({ evaluation }: { evaluation: IdeaEvaluation }) {
  const dq = evaluation.decision_quality
  if (!dq) return null
  const catalyst = dq.catalyst_or_reason_now || {}
  const invalidation = dq.invalidation || {}
  const catalystTimeframe = String(catalyst.expected_timeframe || "")
  const invalidationTimeframe = String(invalidation.timeframe || "")
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Structured Catalyst</h3>
        <p className="mt-2 text-sm leading-6 text-muted">{String(catalyst.event_or_condition || evaluation.catalyst || "N/A")}</p>
        {catalystTimeframe && <p className="mt-1 text-xs text-subtle">{catalystTimeframe}</p>}
      </div>
      <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Structured Invalidation</h3>
        <p className="mt-2 text-sm leading-6 text-muted">{String(invalidation.threshold || evaluation.invalidation || "N/A")}</p>
        {invalidationTimeframe && <p className="mt-1 text-xs text-subtle">{invalidationTimeframe}</p>}
      </div>
    </div>
  )
}

function factorDisplayLabel(name: string, evaluation: IdeaEvaluation) {
  const instrument = evaluation.data_quality?.instrument
  const asset = String(instrument?.asset || "equity")
  const instrumentType = String(instrument?.instrument_type || "security")
  const equitySecurity = asset === "equity" && instrumentType === "security"
  if (equitySecurity) return formatLabel(name)
  if (name === "industry_attractiveness") return "Market Structure"
  if (name === "business_quality") return "Thesis Quality"
  if (name === "management_quality") return "Issuer / Manager"
  if (name === "valuation_asymmetry") return "Asymmetry"
  return formatLabel(name)
}

function FactorScore({ name, factor, evaluation }: { name: string; factor: IdeaFactorScore; evaluation: IdeaEvaluation }) {
  const score = typeof factor?.score === "number" ? factor.score : null
  return (
    <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{factorDisplayLabel(name, evaluation)}</span>
        <span className="font-mono text-sm font-semibold text-app">{scoreText(score)}</span>
      </div>
      {factor?.status && <p className="mt-1 text-xs text-muted">{formatLabel(factor.status)}</p>}
      {factor?.rationale && <p className="mt-2 text-xs leading-5 text-subtle">{factor.rationale}</p>}
    </div>
  )
}

function MissingRows({ rows }: { rows: IdeaMissingInformation[] }) {
  if (!rows.length) {
    return <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">No missing information flagged.</p>
  }
  return (
    <div className="space-y-2">
      {rows.map((row, index) => (
        <div key={`${row.field}-${index}`} className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-app">{formatLabel(row.field)}</span>
            <StatusBadge tone={row.severity === "critical" ? "error" : row.severity === "high" ? "warning" : "neutral"}>
              {formatLabel(row.severity)}
            </StatusBadge>
          </div>
          {row.reason && <p className="mt-2 text-sm leading-6 text-muted">{row.reason}</p>}
        </div>
      ))}
    </div>
  )
}

function formatMaybeNumber(value: unknown, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A"
}

function formatAvailability(value: boolean | undefined) {
  if (value === true) return "Available"
  if (value === false) return "Unavailable"
  return "N/A"
}

function activeRiskFlags(flags?: AnalyzerRiskFlags | null) {
  if (!flags) return []
  return Object.entries(RISK_FLAG_LABELS).filter(([key]) => flags[key] === true)
}

function nonZeroRiskParts(parts?: AnalyzerRiskParts | null) {
  if (!parts) return []
  return Object.entries(parts)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]) && Math.abs(entry[1]) > 1e-9)
    .sort(([a], [b]) => a.localeCompare(b))
}

export function AnalyzerRiskBadges({
  context,
  emptyLabel = "No risk flags",
}: {
  context?: IdeaAnalyzerContext | null
  emptyLabel?: string
}) {
  if (!context || context.status !== "available") {
    return <StatusBadge tone="neutral">{formatLabel(String(context?.status || "inactive"))}</StatusBadge>
  }
  const flags = activeRiskFlags(context.risk_flags)
  if (!flags.length) {
    return <StatusBadge tone="neutral">{emptyLabel}</StatusBadge>
  }
  return (
    <span className="flex flex-wrap gap-1.5">
      {flags.map(([key, label]) => (
        <StatusBadge key={key} tone={RISK_FLAG_TONES[key] ?? "neutral"}>{label}</StatusBadge>
      ))}
    </span>
  )
}

function AnalyzerRiskDetails({ context }: { context: IdeaAnalyzerContext }) {
  const parts = nonZeroRiskParts(context.risk_parts)
  return (
    <div className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Risk Brakes</span>
        <AnalyzerRiskBadges context={context} />
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <div className="rounded-md border border-app bg-card px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Short Cover Risk</span>
          <p className="mt-1 font-mono text-sm text-app">{formatMaybeNumber(context.short_cover_risk)}</p>
        </div>
        <div className="rounded-md border border-app bg-card px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Long Risk Penalty</span>
          <p className="mt-1 font-mono text-sm text-app">{formatMaybeNumber(context.long_risk_penalty)}</p>
        </div>
      </div>
      {parts.length > 0 && (
        <div className="mt-3 grid gap-2 md:grid-cols-3">
          {parts.map(([key, value]) => (
            <div key={key} className="rounded-md border border-app bg-card px-3 py-2">
              <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{formatLabel(key)}</span>
              <p className="mt-1 font-mono text-sm text-app">{formatMaybeNumber(value)}</p>
            </div>
          ))}
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-xs text-subtle">
        <span>Drawdown metrics: {formatAvailability(context.drawdown_metrics_available)}</span>
        <span>Squeeze metrics: {formatAvailability(context.short_squeeze_metrics_available)}</span>
      </div>
    </div>
  )
}

function AnalyzerContextPanel({ context }: { context: IdeaAnalyzerContext }) {
  const status = String(context.status || "unavailable")
  const coverage = context.coverage && typeof context.coverage === "object" ? context.coverage : {}
  const diagnostics =
    context.diagnostic_subfactors && typeof context.diagnostic_subfactors === "object"
      ? context.diagnostic_subfactors
      : {}
  const gates = Array.isArray(context.gate_reasons) ? context.gate_reasons : []
  const warnings = Array.isArray(context.warnings) ? context.warnings : []

  return (
    <section className="space-y-3">
      <h3 className="section-title text-sm">Analyzer Context</h3>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Action</span>
          <p className="mt-2 text-sm font-semibold text-app">{String(context.action_label || formatLabel(status))}</p>
        </div>
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Scenario</span>
          <p className="mt-2 font-mono text-sm font-semibold text-app">{formatMaybeNumber(context.scenario_score)}</p>
        </div>
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Confidence</span>
          <p className="mt-2 font-mono text-sm font-semibold text-app">
            {typeof context.confidence === "number" ? `${Math.round(context.confidence * 100)}%` : "N/A"}
          </p>
        </div>
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Coverage</span>
          <p className="mt-2 font-mono text-sm font-semibold text-app">
            {typeof coverage.ratio === "number" ? `${Math.round(coverage.ratio * 100)}%` : "N/A"}
          </p>
        </div>
      </div>
      <div className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          <span>Gate: {String(context.gate_status || "N/A")}</span>
          <span>Source: {context.source_timestamp ? formatDate(String(context.source_timestamp)) : "N/A"}</span>
        </div>
        {status === "available" && <div className="mt-3"><AnalyzerRiskDetails context={context} /></div>}
        {gates.length > 0 && <p className="mt-2 leading-6">Gates: {gates.map(String).join("; ")}</p>}
        {warnings.length > 0 && <p className="mt-2 leading-6">Warnings: {warnings.map(String).join("; ")}</p>}
        {Object.keys(diagnostics).length > 0 && (
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {Object.entries(diagnostics).map(([name, row]) => (
              <div key={name} className="rounded-md border border-app bg-card px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{formatLabel(name)}</span>
                <p className="mt-1 font-mono text-sm text-app">
                  Signal {formatMaybeNumber(row.signal)} / Score {formatMaybeNumber(row.mapped_score, 1)}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function portfolioContextUsage(evaluation: IdeaEvaluation): boolean | null {
  const value = evaluation.data_quality?.portfolio_context_used
  return typeof value === "boolean" ? value : null
}

function PortfolioContextBadge({ evaluation }: { evaluation: IdeaEvaluation }) {
  const used = portfolioContextUsage(evaluation)
  if (used === true) return <StatusBadge tone="info">Portfolio used</StatusBadge>
  if (used === false) return <StatusBadge tone="neutral">Portfolio excluded</StatusBadge>
  return <StatusBadge tone="neutral">Portfolio unknown</StatusBadge>
}

export function EvaluationPanel({
  evaluation,
  onAccept,
  onReject,
  accepting,
  rejecting,
  portfolioContextEnabled,
  onPortfolioContextChange,
  portfolioContextUpdating,
}: {
  evaluation: IdeaEvaluation | null
  onAccept: () => void
  onReject: () => void
  accepting: boolean
  rejecting: boolean
  portfolioContextEnabled?: boolean
  onPortfolioContextChange?: (checked: boolean) => void
  portfolioContextUpdating?: boolean
}) {
  if (!evaluation) {
    return (
      <div className="space-y-3">
        {typeof portfolioContextEnabled === "boolean" && onPortfolioContextChange && (
          <div className="rounded-lg border border-app bg-card-muted px-3 py-2">
            <Toggle
              label="Portfolio Influence"
              checked={portfolioContextEnabled}
              onChange={onPortfolioContextChange}
              disabled={portfolioContextUpdating}
              description={portfolioContextEnabled ? "Used for future evaluations." : "Excluded from future evaluations."}
            />
          </div>
        )}
        <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">No evaluation yet.</p>
      </div>
    )
  }

  const factors = CANONICAL_FACTORS
    .map(name => [name, evaluation.factor_scores?.[name]] as const)
    .filter((entry): entry is readonly [string, IdeaFactorScore] => Boolean(entry[1]))
  const evidence = evaluation.evidence || []
  const disconfirming = evaluation.disconfirming_evidence || []
  const accepted = Boolean(evaluation.accepted_at || evaluation.recommendation_id)
  const analyzerContext =
    evaluation.analyzer_context && Object.keys(evaluation.analyzer_context).length > 0
      ? evaluation.analyzer_context
      : null

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <ActionPill action={evaluation.action} />
        <span className="font-mono text-sm font-semibold text-app">Score {scoreText(evaluation.score)}</span>
        <span className="text-sm text-subtle">Confidence {evaluation.confidence == null ? "N/A" : `${Math.round(Number(evaluation.confidence) * 100)}%`}</span>
        <span className="text-sm text-subtle">{formatDate(evaluation.evaluated_at)}</span>
        <PortfolioContextBadge evaluation={evaluation} />
        {accepted && <StatusBadge tone="success">Accepted</StatusBadge>}
      </div>

      {typeof portfolioContextEnabled === "boolean" && onPortfolioContextChange && (
        <div className="rounded-lg border border-app bg-card-muted px-3 py-2">
          <Toggle
            label="Portfolio Influence"
            checked={portfolioContextEnabled}
            onChange={onPortfolioContextChange}
            disabled={portfolioContextUpdating}
            description={portfolioContextEnabled ? "Used for future evaluations." : "Excluded from future evaluations."}
          />
        </div>
      )}

      {evaluation.thesis_statement && (
        <p className="text-base font-medium leading-7 text-app">{evaluation.thesis_statement}</p>
      )}
      {evaluation.rationale && <p className="text-sm leading-6 text-muted">{evaluation.rationale}</p>}

      <EvaluatorDiagnosticNotice evaluation={evaluation} />

      <DecisionQualityGatePanel evaluation={evaluation} />

      {factors.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {factors.map(([name, factor]) => <FactorScore key={name} name={name} factor={factor} evaluation={evaluation} />)}
        </div>
      )}

      {analyzerContext && <AnalyzerContextPanel context={analyzerContext} />}

      <section className="space-y-3">
        <h3 className="section-title text-sm">Missing Information</h3>
        <MissingRows rows={evaluation.missing_information || []} />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="space-y-3">
          <h3 className="section-title text-sm">Supporting Evidence</h3>
          {evidence.length ? evidence.slice(0, 6).map((item, index) => (
            <div key={index} className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
              <span className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                {String(item.source || "Evidence")}
              </span>
              <span className="mt-1 block leading-6">{String(item.summary || item.url || "Evidence item")}</span>
            </div>
          )) : <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">No supporting evidence stored.</p>}
        </section>
        <section className="space-y-3">
          <h3 className="section-title text-sm">Disconfirming Evidence</h3>
          {disconfirming.length ? disconfirming.slice(0, 6).map((item, index) => (
            <div key={index} className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">
              <span className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                {String(item.source || "Risk")}
              </span>
              <span className="mt-1 block leading-6">{String(item.summary || item.url || "Evidence item")}</span>
            </div>
          )) : <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">No disconfirming evidence stored.</p>}
        </section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Catalyst</h3>
          <p className="mt-2 text-sm leading-6 text-muted">{evaluation.catalyst || "N/A"}</p>
        </div>
        <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">Invalidation</h3>
          <p className="mt-2 text-sm leading-6 text-muted">{evaluation.invalidation || "N/A"}</p>
        </div>
      </div>

      <DecisionQualitySummary evaluation={evaluation} />

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onAccept}
          disabled={accepted || accepting}
          className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
        >
          <CheckCircle2 size={16} aria-hidden="true" />
          {accepting ? "Accepting" : accepted ? "Accepted" : "Accept"}
        </button>
        <button
          type="button"
          onClick={onReject}
          disabled={rejecting}
          className="theme-button-base theme-button-secondary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
        >
          <XCircle size={16} aria-hidden="true" />
          {rejecting ? "Rejecting" : "Reject Idea"}
        </button>
      </div>
    </div>
  )
}
