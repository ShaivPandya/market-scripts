import { GitBranch } from "lucide-react"

import { TraceTriggerButton } from "@/components/shared/TraceTriggerButton"
import { useDecisionTrace } from "@/contexts/DecisionTraceContext"
import type { EvidenceLedgerSummary, ProvenanceSelector } from "@/lib/api"
import { cn } from "@/lib/utils"

function formatTime(iso: string | null | undefined): string {
  const value = String(iso ?? "").trim()
  if (!value) return "Unknown time"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
}

function sourceQualityClass(quality: string | null | undefined): string {
  const normalized = String(quality ?? "").toLowerCase()
  if (normalized === "ok") return "theme-badge-success"
  if (normalized === "degraded" || normalized === "warning") return "theme-badge-warning"
  if (normalized === "error" || normalized === "failed" || normalized === "stale") return "theme-badge-error"
  return "theme-badge-neutral"
}

function EvidenceBundleList({
  title,
  items,
  onTrace,
}: {
  title: string
  items: EvidenceLedgerSummary["claims"][number]["supporting_evidence"]
  onTrace: (selector: ProvenanceSelector, label?: string) => void
}) {
  if (!items.length) return null
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-subtle">{title}</p>
      {items.map((bundle, index) => {
        const evidence = bundle.evidence
        const summary = String(evidence.summary ?? evidence.title ?? "Evidence item").trim()
        const sourceRecord = bundle.source_record
        const sourceRecordId = String(evidence.source_record_id ?? sourceRecord?.source_record_id ?? "").trim()
        return (
          <div key={`${summary}-${index}`} className="rounded-lg border border-app px-3 py-2 text-sm">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                {evidence.title && <p className="font-medium text-app">{evidence.title}</p>}
                <p className="mt-1 text-muted">{summary}</p>
                {evidence.observed_at && (
                  <p className="mt-1 text-xs text-subtle">Observed {formatTime(evidence.observed_at)}</p>
                )}
              </div>
              {sourceRecordId && (
                <button
                  type="button"
                  className="inline-flex shrink-0 items-center gap-1 rounded border border-app px-2 py-1 text-xs text-muted hover:text-app"
                  onClick={() => onTrace({ source_record_id: sourceRecordId }, summary)}
                  title="Trace source lineage"
                >
                  <GitBranch className="h-3.5 w-3.5" />
                  Trace
                </button>
              )}
            </div>
            {sourceRecord && (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-subtle">{sourceRecord.source_name ?? sourceRecord.vendor ?? "Source"}</span>
                {sourceRecord.quality && (
                  <span className={cn("rounded px-1.5 py-0.5 font-medium", sourceQualityClass(sourceRecord.quality))}>
                    {sourceRecord.quality}
                  </span>
                )}
                {sourceRecord.as_of && <span className="text-subtle">as of {formatTime(sourceRecord.as_of)}</span>}
              </div>
            )}
            {bundle.citations.length > 0 && (
              <div className="mt-2 space-y-1 border-t border-app pt-2">
                {bundle.citations.map((citation, citationIndex) => (
                  <div key={`${citation.citation_id ?? citationIndex}`} className="text-xs text-muted">
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline dark:text-blue-400">
                        {citation.title ?? citation.url}
                      </a>
                    ) : (
                      <span>{citation.title ?? citation.source_path ?? "Citation"}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function EvidenceLedgerPanel({ ledger, ticker }: { ledger: EvidenceLedgerSummary | null | undefined; ticker: string }) {
  const { openDecisionTrace } = useDecisionTrace()

  function handleTrace(selector: ProvenanceSelector, label?: string) {
    const sourceRecordId = String(selector.source_record_id ?? "").trim()
    if (!sourceRecordId) return
    openDecisionTrace({
      kind: "source_record",
      record: { source_record_id: sourceRecordId, label },
    })
  }

  if (!ledger) {
    return <p className="text-sm text-muted">Evidence ledger unavailable for {ticker}.</p>
  }

  const claimCount = ledger.counts?.claims ?? ledger.claims.length
  const evidenceCount = ledger.counts?.evidence_items ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-app">Evidence Ledger</h2>
          <p className="text-sm text-muted">
            Structured source and evidence lineage for thesis claims and recommendations.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-subtle">
          <span className="rounded border border-app px-2 py-1">{claimCount} claims</span>
          <span className="rounded border border-app px-2 py-1">{evidenceCount} evidence items</span>
        </div>
      </div>

      {!ledger.claims.length && !ledger.recommendations.length ? (
        <p className="text-sm text-muted">No structured evidence linked yet for {ticker}.</p>
      ) : (
        <div className="space-y-6">
          {ledger.claims.map(claim => (
            <section key={claim.claim_id ?? claim.claim} className="theme-surface rounded-xl p-4">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-app">{claim.claim ?? "Thesis claim"}</h3>
                {claim.status && (
                  <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-muted dark:bg-gray-800">
                    {claim.status}
                  </span>
                )}
              </div>
              {(claim.expected_evidence_text || claim.disconfirming_evidence_text) && (
                <div className="mb-3 grid gap-2 text-xs text-muted md:grid-cols-2">
                  {claim.expected_evidence_text && (
                    <div>
                      <p className="font-semibold uppercase text-subtle">Expected evidence (text)</p>
                      <p className="mt-1">{claim.expected_evidence_text}</p>
                    </div>
                  )}
                  {claim.disconfirming_evidence_text && (
                    <div>
                      <p className="font-semibold uppercase text-subtle">Disconfirming evidence (text)</p>
                      <p className="mt-1">{claim.disconfirming_evidence_text}</p>
                    </div>
                  )}
                </div>
              )}
              <div className="grid gap-4 md:grid-cols-2">
                <EvidenceBundleList
                  title="Supporting evidence"
                  items={claim.supporting_evidence}
                  onTrace={handleTrace}
                />
                <EvidenceBundleList
                  title="Disconfirming evidence"
                  items={claim.disconfirming_evidence}
                  onTrace={handleTrace}
                />
              </div>
            </section>
          ))}

          {ledger.recommendations.length > 0 && (
            <section className="theme-surface rounded-xl p-4">
              <h3 className="mb-3 text-sm font-semibold text-app">Recommendation evidence</h3>
              <div className="space-y-4">
                {ledger.recommendations.map(recommendation => (
                  <div key={recommendation.recommendation_id ?? `${recommendation.action}-${recommendation.as_of}`} className="rounded-lg border border-app p-3">
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                      <span className="font-medium text-app">{recommendation.action ?? "Recommendation"}</span>
                      {recommendation.as_of && <span className="text-subtle">{formatTime(recommendation.as_of)}</span>}
                      {recommendation.status && (
                        <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-muted dark:bg-gray-800">
                          {recommendation.status}
                        </span>
                      )}
                      {recommendation.recommendation_id && recommendation.action && (
                        <TraceTriggerButton
                          compact
                          label="Trace recommendation evidence"
                          onClick={() =>
                            openDecisionTrace({
                              kind: "recommendation",
                              record: {
                                id: recommendation.recommendation_id,
                                action: recommendation.action,
                                as_of: recommendation.as_of,
                              },
                            })
                          }
                        />
                      )}
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      <EvidenceBundleList
                        title="Supporting evidence"
                        items={recommendation.supporting_evidence}
                        onTrace={handleTrace}
                      />
                      <EvidenceBundleList
                        title="Disconfirming evidence"
                        items={recommendation.disconfirming_evidence}
                        onTrace={handleTrace}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  )
}
