import { CheckCircle2, XCircle } from "lucide-react"

import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import type { IdeaEvaluation, IdeaFactorScore, IdeaMissingInformation } from "@/lib/api"
import { formatDate, formatLabel, scoreText } from "@/lib/ideaUtils"

const ACTION_TONE: Record<string, StatusTone> = {
  buy: "success",
  watch: "info",
  avoid: "error",
  do_nothing: "neutral",
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

function FactorScore({ name, factor }: { name: string; factor: IdeaFactorScore }) {
  const score = typeof factor?.score === "number" ? factor.score : null
  return (
    <div className="rounded-lg border border-app bg-card-muted px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{formatLabel(name)}</span>
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

export function EvaluationPanel({
  evaluation,
  onAccept,
  onReject,
  accepting,
  rejecting,
}: {
  evaluation: IdeaEvaluation | null
  onAccept: () => void
  onReject: () => void
  accepting: boolean
  rejecting: boolean
}) {
  if (!evaluation) {
    return <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">No evaluation yet.</p>
  }

  const factors = Object.entries(evaluation.factor_scores || {})
  const evidence = evaluation.evidence || []
  const disconfirming = evaluation.disconfirming_evidence || []
  const accepted = Boolean(evaluation.accepted_at || evaluation.recommendation_id)

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <ActionPill action={evaluation.action} />
        <span className="font-mono text-sm font-semibold text-app">Score {scoreText(evaluation.score)}</span>
        <span className="text-sm text-subtle">Confidence {evaluation.confidence == null ? "N/A" : `${Math.round(Number(evaluation.confidence) * 100)}%`}</span>
        <span className="text-sm text-subtle">{formatDate(evaluation.evaluated_at)}</span>
        {accepted && <StatusBadge tone="success">Accepted</StatusBadge>}
      </div>

      {evaluation.thesis_statement && (
        <p className="text-base font-medium leading-7 text-app">{evaluation.thesis_statement}</p>
      )}
      {evaluation.rationale && <p className="text-sm leading-6 text-muted">{evaluation.rationale}</p>}

      {factors.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {factors.map(([name, factor]) => <FactorScore key={name} name={name} factor={factor} />)}
        </div>
      )}

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
