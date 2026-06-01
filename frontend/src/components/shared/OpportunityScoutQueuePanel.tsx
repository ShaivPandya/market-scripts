import { useState } from "react"
import { Link } from "react-router-dom"
import { Eye, FlaskConical, Radar, Search, Sparkles, X } from "lucide-react"

import { StagedProposalNotice } from "@/components/shared/StagedProposalNotice"
import { DecisionStateBadge } from "@/components/shared/DecisionStateBadge"
import {
  createMonitorForOpportunityCandidate,
  dismissOpportunityCandidate,
  type OpportunityCandidateRecord,
  promoteOpportunityCandidate,
  requestResearchForOpportunityCandidate,
  watchOpportunityCandidate,
} from "@/lib/api"

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—"
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

interface OpportunityScoutQueuePanelProps {
  items: OpportunityCandidateRecord[]
  onUpdated?: () => void
}

type FeedbackResult = {
  label: string
  approval?: { approval_id?: string | number; decision_state?: string; effect_scope?: string; review_route?: string }
}

export function OpportunityScoutQueuePanel({ items, onUpdated }: OpportunityScoutQueuePanelProps) {
  const [pendingId, setPendingId] = useState<string | null>(null)
  const [feedback, setFeedback] = useState<Record<string, FeedbackResult>>({})

  async function runAction(
    candidateId: string,
    label: string,
    action: () => Promise<{ status?: string; approval?: FeedbackResult["approval"]; status_proposal?: FeedbackResult["approval"]; watch_proposal?: FeedbackResult["approval"]; research_proposal?: FeedbackResult["approval"]; promote_proposal?: FeedbackResult["approval"]; monitor_proposal?: FeedbackResult["approval"] }>,
  ) {
    setPendingId(candidateId)
    try {
      const result = await action()
      const approval =
        result.approval ||
        result.status_proposal ||
        result.watch_proposal ||
        result.research_proposal ||
        result.promote_proposal ||
        result.monitor_proposal
      setFeedback(current => ({
        ...current,
        [candidateId]: { label, approval },
      }))
      onUpdated?.()
    } finally {
      setPendingId(null)
    }
  }

  if (!items.length) return null

  return (
    <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4 max-md:max-h-[min(40rem,calc(100dvh-12rem))] lg:col-span-2">
      <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
        <Radar size={14} className="text-violet-500" />
        OpportunityScout
        <span className="ml-auto text-xs text-subtle">{items.length} candidate{items.length !== 1 ? "s" : ""}</span>
      </h2>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {items.map(candidate => {
          const candidateId = candidate.candidate_id || candidate.id
          const isPending = pendingId === candidateId
          const result = feedback[candidateId]
          return (
            <div key={candidateId} className="rounded-lg border border-app/60 px-3 py-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                {candidate.ticker ? (
                  <Link
                    to={`/dossier/${encodeURIComponent(candidate.ticker)}`}
                    state={{ from: "workspace" }}
                    className="font-semibold text-app hover:underline"
                  >
                    {candidate.ticker}
                  </Link>
                ) : (
                  <span className="font-semibold text-app">Sector / Thematic</span>
                )}
                <span className="rounded bg-violet-50 px-1.5 py-0.5 text-xs font-medium text-violet-800 dark:bg-violet-950 dark:text-violet-200">
                  {candidate.opportunity_type.replace(/_/g, " ")}
                </span>
                <span className="text-xs uppercase tracking-wide text-subtle">{candidate.source_kind.replace(/_/g, " ")}</span>
                <DecisionStateBadge state={candidate.gate_status === "blocked" ? "blocked" : "generated"} />
                <span className="ml-auto text-xs text-subtle">Score {candidate.rank_score?.toFixed(1) ?? "—"}</span>
              </div>

              <p className="mt-2 text-app">{candidate.trigger}</p>
              <div className="mt-2 grid gap-2 text-xs text-muted md:grid-cols-2">
                <p><span className="font-medium text-subtle">Why now:</span> {candidate.why_now || "—"}</p>
                <p><span className="font-medium text-subtle">Variant view:</span> {candidate.variant_view || "—"}</p>
                <p><span className="font-medium text-subtle">Price confirmation:</span> {candidate.price_confirmation || "—"}</p>
                <p><span className="font-medium text-subtle">Next action:</span> {candidate.next_action.replace(/_/g, " ")}</p>
              </div>
              {candidate.missing_inputs.length > 0 && (
                <p className="mt-2 text-xs text-subtle">
                  Missing: {candidate.missing_inputs.join(" · ")}
                </p>
              )}
              {candidate.updated_at && (
                <p className="mt-1 text-[11px] text-subtle">Updated {formatTime(candidate.updated_at)}</p>
              )}

              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => runAction(candidateId, "Dismiss staged", () => dismissOpportunityCandidate({ candidate_id: candidateId }))}
                  className="inline-flex items-center gap-1 rounded border border-app px-2 py-1 text-xs hover:bg-app/5 disabled:opacity-50"
                  aria-label={`Dismiss ${candidate.ticker || "candidate"}`}
                >
                  <X size={12} /> Dismiss
                </button>
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => runAction(candidateId, "Watch staged", () => watchOpportunityCandidate({ candidate_id: candidateId }))}
                  className="inline-flex items-center gap-1 rounded border border-app px-2 py-1 text-xs hover:bg-app/5 disabled:opacity-50"
                >
                  <Eye size={12} /> Watch
                </button>
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => runAction(candidateId, "Research staged", () => requestResearchForOpportunityCandidate({ candidate_id: candidateId }))}
                  className="inline-flex items-center gap-1 rounded border border-app px-2 py-1 text-xs hover:bg-app/5 disabled:opacity-50"
                >
                  <Search size={12} /> Request Research
                </button>
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => runAction(candidateId, "Promote staged", () => promoteOpportunityCandidate({ candidate_id: candidateId }))}
                  className="inline-flex items-center gap-1 rounded border border-app px-2 py-1 text-xs hover:bg-app/5 disabled:opacity-50"
                >
                  <Sparkles size={12} /> Promote to DQ
                </button>
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => runAction(candidateId, "Monitor staged", () => createMonitorForOpportunityCandidate({ candidate_id: candidateId }))}
                  className="inline-flex items-center gap-1 rounded border border-app px-2 py-1 text-xs hover:bg-app/5 disabled:opacity-50"
                >
                  <FlaskConical size={12} /> Create Monitor
                </button>
              </div>

              {result && (
                <div className="mt-2">
                  <StagedProposalNotice proposal={result.approval} showReviewLink>
                    {result.label}. Approval is required before app state changes.
                  </StagedProposalNotice>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
