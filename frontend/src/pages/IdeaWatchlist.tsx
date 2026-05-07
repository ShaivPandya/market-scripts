import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Play, Plus, RefreshCw, Trash2 } from "lucide-react"

import { ActionPill, AnalyzerRiskBadges, ConfidencePill } from "@/components/idea/EvaluationPanel"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import { ActionButton, TextInput } from "@/components/shared/FormControls"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  createIdea,
  deleteIdea,
  fetchIdeaComparisonEvaluationJob,
  fetchIdeaComparisonRuns,
  fetchIdeaEvaluationJob,
  fetchIdeas,
  startIdeaComparisonEvaluationJob,
  updateIdea,
  type IdeaAnalyzerDirection,
  type IdeaAnalyzerContext,
  type IdeaComparisonJobResponse,
  type IdeaComparisonRun,
  type IdeaListResponse,
  type IdeaStatus,
} from "@/lib/api"
import { AnalyzerWorkbench } from "@/pages/PortfolioAnalyzer"
import {
  formatDate,
  formatLabel,
  latestEvaluation,
  missingCount,
  readActiveJobs,
  scoreText,
  writeActiveJobs,
} from "@/lib/ideaUtils"
import { cn } from "@/lib/utils"

const ACTIVE_COMPARISON_JOB_KEY = "idea-watchlist-active-comparison-job-v1"
const IDEA_WATCHLIST_ANALYZER_STATE_KEY = ["idea-watchlist", "analyzer", "state-v3"] as const
const ACTIONABLE_IDEA_STATUSES = new Set<IdeaStatus | string>(["watching", "researching", "ready_for_review"])
const ANALYZER_DIRECTIONS: { value: IdeaAnalyzerDirection; label: string }[] = [
  { value: "inactive", label: "Inactive" },
  { value: "long", label: "Long" },
  { value: "short", label: "Short" },
]

const STATUS_TONE: Record<string, StatusTone> = {
  watching: "neutral",
  researching: "info",
  ready_for_review: "warning",
  accepted: "success",
  rejected: "error",
  archived: "neutral",
}

function readActiveComparisonJob(): string | null {
  try {
    const jobId = window.localStorage.getItem(ACTIVE_COMPARISON_JOB_KEY)
    return jobId && jobId.trim() ? jobId : null
  } catch {
    return null
  }
}

function writeActiveComparisonJob(jobId: string | null) {
  if (jobId) {
    window.localStorage.setItem(ACTIVE_COMPARISON_JOB_KEY, jobId)
  } else {
    window.localStorage.removeItem(ACTIVE_COMPARISON_JOB_KEY)
  }
}

function StatusPill({ status }: { status: string }) {
  return <StatusBadge tone={STATUS_TONE[status] ?? "neutral"}>{formatLabel(status)}</StatusBadge>
}

function analyzerDirection(idea: { metadata?: Record<string, unknown> }): IdeaAnalyzerDirection {
  const direction = String(idea.metadata?.analyzer_direction || "inactive").toLowerCase()
  return direction === "long" || direction === "short" ? direction : "inactive"
}

function ComparativeRankingPanel({
  run,
  riskByIdeaId,
}: {
  run: IdeaComparisonRun | null
  riskByIdeaId: Record<string, IdeaAnalyzerContext | null>
}) {
  const rankings = run?.rankings ?? []
  return (
    <section className="mb-4 rounded-lg border border-app bg-card-muted p-3">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="section-title text-sm">Comparative Ranking</h3>
          <p className="mt-1 text-xs text-subtle">
            {run ? `${rankings.length} ranked ideas / ${formatDate(run.created_at)}` : "No comparative run yet."}
          </p>
        </div>
        {run?.run_id && <span className="font-mono text-xs text-subtle">{run.run_id.slice(0, 18)}</span>}
      </div>
      {run?.summary && <p className="mb-3 text-sm leading-6 text-muted">{run.summary}</p>}
      {rankings.length ? (
        <div className="overflow-x-auto rounded-lg border border-app bg-card">
          <table className="w-full min-w-[860px] border-collapse text-sm">
            <thead className="bg-card-muted">
              <tr>
                {["Rank", "Ticker", "Action", "Risk", "Score", "Confidence", "Rationale"].map(label => (
                  <th key={label} className="border-b border-app px-3 py-2 text-left text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rankings.map(row => (
                <tr key={`${row.run_id}-${row.rank}-${row.idea_id}`} className="border-b border-app last:border-b-0">
                  <td className="px-3 py-3 font-mono text-app">#{row.rank}</td>
                  <td className="px-3 py-3 font-semibold text-app">{row.ticker}</td>
                  <td className="px-3 py-3"><ActionPill action={row.action} /></td>
                  <td className="px-3 py-3"><AnalyzerRiskBadges context={riskByIdeaId[String(row.idea_id)]} /></td>
                  <td className="px-3 py-3 font-mono text-app">{scoreText(row.score)}</td>
                  <td className="px-3 py-3"><ConfidencePill level={row.confidence_level} confidence={row.confidence} /></td>
                  <td className="max-w-[28rem] px-3 py-3 text-muted">{row.rationale || "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="rounded-lg border border-app bg-card px-3 py-4 text-sm text-muted">Run Evaluate All to rank active ideas together.</p>
      )}
    </section>
  )
}

export function IdeaWatchlist() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [activeJobs, setActiveJobs] = useState<Record<string, string>>(() => readActiveJobs())
  const [activeComparisonJob, setActiveComparisonJob] = useState<string | null>(() => readActiveComparisonJob())
  const [comparisonJobSnapshot, setComparisonJobSnapshot] = useState<IdeaComparisonJobResponse | null>(null)
  const [comparisonJobError, setComparisonJobError] = useState<string | null>(null)
  const [ticker, setTicker] = useState("")
  const [companyName, setCompanyName] = useState("")
  const [tags, setTags] = useState("")
  const [notes, setNotes] = useState("")
  const [deletingIdeaIds, setDeletingIdeaIds] = useState<Set<string>>(() => new Set())
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const ideasQuery = useApiQuery(["ideas"], () => fetchIdeas({ include_archived: false, limit: 300 }), 30_000)
  const ideas = useMemo(() => ideasQuery.data?.ideas ?? [], [ideasQuery.data?.ideas])
  const actionableIdeas = useMemo(
    () => ideas.filter(idea => ACTIONABLE_IDEA_STATUSES.has(idea.status)),
    [ideas],
  )
  const comparisonQuery = useApiQuery(["idea-comparison-runs"], () => fetchIdeaComparisonRuns({ limit: 1 }), 30_000)
  const latestComparisonRun = useMemo(() => {
    const persisted = comparisonQuery.data?.runs?.[0] ?? null
    if (persisted) return persisted
    return comparisonJobSnapshot?.status === "done" ? comparisonJobSnapshot.result?.run ?? null : null
  }, [comparisonJobSnapshot, comparisonQuery.data?.runs])

  function setActiveJobsAndPersist(next: Record<string, string>) {
    setActiveJobs(next)
    writeActiveJobs(next)
  }

  function setActiveComparisonJobAndPersist(jobId: string | null) {
    setActiveComparisonJob(jobId)
    writeActiveComparisonJob(jobId)
  }

  useEffect(() => {
    if (!Object.keys(activeJobs).length) return
    let stopped = false

    async function pollJobs() {
      const entries = Object.entries(activeJobs)
      const next = { ...activeJobs }
      let changed = false
      await Promise.all(entries.map(async ([ideaId, jobId]) => {
        try {
          const job = await fetchIdeaEvaluationJob(jobId)
          if (job.status === "done") {
            delete next[ideaId]
            changed = true
            await qc.invalidateQueries({ queryKey: ["ideas"] })
            await qc.invalidateQueries({ queryKey: ["idea", ideaId] })
          } else if (job.status === "error" || job.status === "cancelled") {
            delete next[ideaId]
            changed = true
          }
        } catch {
          // transient polling errors are surfaced on the detail page
        }
      }))
      if (!stopped && changed) setActiveJobsAndPersist(next)
    }

    void pollJobs()
    const handle = window.setInterval(() => void pollJobs(), 2500)
    return () => {
      stopped = true
      window.clearInterval(handle)
    }
  }, [activeJobs, qc])

  useEffect(() => {
    if (!activeComparisonJob) return
    let stopped = false

    async function pollComparisonJob() {
      try {
        const job = await fetchIdeaComparisonEvaluationJob(activeComparisonJob as string)
        setComparisonJobSnapshot(job)
        if (job.status === "done") {
          setComparisonJobError(null)
          if (!stopped) setActiveComparisonJobAndPersist(null)
          await qc.invalidateQueries({ queryKey: ["ideas"] })
          await qc.invalidateQueries({ queryKey: ["idea-comparison-runs"] })
        } else if (job.status === "error" || job.status === "cancelled") {
          if (!stopped) setActiveComparisonJobAndPersist(null)
          setComparisonJobError(job.error || "Comparative evaluation failed.")
        }
      } catch (err) {
        setComparisonJobError(err instanceof Error ? err.message : "Unable to poll comparative evaluation.")
      }
    }

    void pollComparisonJob()
    const handle = window.setInterval(() => void pollComparisonJob(), 2500)
    return () => {
      stopped = true
      window.clearInterval(handle)
    }
  }, [activeComparisonJob, qc])

  const createMutation = useMutation({
    mutationFn: () => createIdea({
      ticker,
      company_name: companyName || null,
      user_notes: notes || null,
      tags: tags.split(",").map(t => t.trim()).filter(Boolean),
    }),
    onSuccess: data => {
      setTicker("")
      setCompanyName("")
      setTags("")
      setNotes("")
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      navigate(`/ideas/${data.idea.id}`, { state: { from: "ideas" } })
    },
  })

  const evaluateAllMutation = useMutation({
    mutationFn: () => startIdeaComparisonEvaluationJob(),
    onSuccess: job => {
      setComparisonJobError(null)
      setComparisonJobSnapshot(job)
      if (job.status === "done") {
        void qc.invalidateQueries({ queryKey: ["ideas"] })
        void qc.invalidateQueries({ queryKey: ["idea-comparison-runs"] })
        return
      }
      setActiveComparisonJobAndPersist(job.job_id)
      void qc.invalidateQueries({ queryKey: ["ideas"] })
    },
    onError: err => setComparisonJobError(err instanceof Error ? err.message : "Comparative evaluation failed."),
  })

  const deleteMutation = useMutation({
    mutationFn: (ideaId: string) => deleteIdea(ideaId),
    onMutate: ideaId => {
      setDeleteError(null)
      setDeletingIdeaIds(prev => new Set(prev).add(String(ideaId)))
    },
    onSuccess: (_data, ideaId) => {
      qc.setQueryData<IdeaListResponse>(["ideas"], current => {
        if (!current) return current
        const nextIdeas = current.ideas.filter(idea => String(idea.id) !== String(ideaId))
        return { ...current, ideas: nextIdeas, count: nextIdeas.length }
      })
      setActiveJobs(prev => {
        const next = { ...prev }
        delete next[String(ideaId)]
        writeActiveJobs(next)
        return next
      })
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", ideaId] })
      void qc.invalidateQueries({ queryKey: ["idea-comparison-runs"] })
    },
    onError: err => setDeleteError(err instanceof Error ? err.message : "Could not delete idea."),
    onSettled: (_data, _error, ideaId) => {
      setDeletingIdeaIds(prev => {
        const next = new Set(prev)
        if (ideaId) next.delete(String(ideaId))
        return next
      })
    },
  })

  const directionMutation = useMutation({
    mutationFn: ({ ideaId, direction }: { ideaId: string; direction: IdeaAnalyzerDirection }) =>
      updateIdea(ideaId, { analyzer_direction: direction }),
    onSuccess: (_data, variables) => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", variables.ideaId] })
      qc.removeQueries({ queryKey: IDEA_WATCHLIST_ANALYZER_STATE_KEY })
    },
  })

  const rows = useMemo(() => ideas.map(idea => {
    const evaluation = latestEvaluation(idea, null)
    const activeJob = activeJobs[String(idea.id)]
    return { idea, evaluation, activeJob }
  }), [ideas, activeJobs])
  const riskByIdeaId = useMemo(
    () => Object.fromEntries(rows.map(({ idea, evaluation }) => [String(idea.id), evaluation?.analyzer_context ?? null])),
    [rows],
  )

  function currentComparisonJobResponse(): { jobId: string; message: string } | null {
    if (!activeComparisonJob) return null
    const progress = comparisonJobSnapshot?.progress
    const phase = typeof progress?.phase === "string" ? formatLabel(progress.phase) : "running"
    const done = typeof progress?.done === "number" ? progress.done : null
    const total = typeof progress?.total === "number" && progress.total > 0 ? progress.total : null
    const suffix = done != null && total != null ? ` ${done}/${total}` : ""
    return { jobId: activeComparisonJob, message: `Comparative evaluation ${phase}${suffix}` }
  }

  const comparisonJob = currentComparisonJobResponse()

  return (
    <main className="mx-auto w-full max-w-[1500px] px-4 py-5 sm:px-6 lg:px-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-app">Idea Watchlist</h1>
          <p className="mt-1 text-sm text-subtle">{ideas.length} active ideas / {actionableIdeas.length} actionable</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => evaluateAllMutation.mutate()}
            disabled={!actionableIdeas.length || Boolean(activeComparisonJob) || evaluateAllMutation.isPending}
            className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
          >
            <Play size={16} aria-hidden="true" />
            {activeComparisonJob || evaluateAllMutation.isPending ? "Evaluating All" : "Evaluate All"}
          </button>
          <button
            type="button"
            onClick={() => void ideasQuery.refetch()}
            className="theme-button-base theme-button-secondary min-h-10 px-4 text-sm"
          >
            <RefreshCw size={16} aria-hidden="true" />
            Refresh
          </button>
        </div>
      </div>

      {comparisonJob && (
        <div className="mb-5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-3 text-sm text-blue-900 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
          {comparisonJob.message} ({comparisonJob.jobId.slice(0, 8)})
        </div>
      )}
      {comparisonJobError && (
        <div className="mb-5 rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {comparisonJobError}
        </div>
      )}
      {deleteError && (
        <div className="mb-5 rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {deleteError}
        </div>
      )}

      <section className="theme-surface mb-5 rounded-lg p-4">
        <form
          className="grid gap-3 lg:grid-cols-[minmax(8rem,10rem)_minmax(10rem,1fr)_minmax(10rem,1fr)_minmax(16rem,2fr)_auto]"
          onSubmit={event => {
            event.preventDefault()
            createMutation.mutate()
          }}
        >
          <TextInput label="Ticker" value={ticker} onChange={setTicker} uppercase placeholder="AAPL" />
          <TextInput label="Company" value={companyName} onChange={setCompanyName} placeholder="Apple" />
          <TextInput label="Tags" value={tags} onChange={setTags} placeholder="quality, ai" />
          <TextInput label="Notes" value={notes} onChange={setNotes} placeholder="Reason for review" />
          <div className="flex items-end">
            <ActionButton type="submit" disabled={!ticker.trim()} loading={createMutation.isPending} className="min-h-11">
              <Plus size={16} aria-hidden="true" />
              Add
            </ActionButton>
          </div>
        </form>
        {createMutation.error && <p className="mt-3 text-sm text-red-600">{createMutation.error.message}</p>}
      </section>

      <section className="theme-surface rounded-lg p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="section-title text-sm">Watchlist</h2>
          {(ideasQuery.isLoading || comparisonQuery.isLoading) && <LoadingSpinner message="Loading ideas" />}
        </div>

        <ComparativeRankingPanel run={latestComparisonRun} riskByIdeaId={riskByIdeaId} />
        {comparisonQuery.error && (
          <div className="mb-4">
            <ErrorMessage message={`Could not load comparison runs: ${comparisonQuery.error.message}`} />
          </div>
        )}

        {ideasQuery.error ? (
          <ErrorMessage message={`Could not load ideas: ${ideasQuery.error.message}`} />
        ) : rows.length === 0 ? (
          <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">No ideas.</p>
        ) : (
          <div className="max-h-[19rem] overflow-auto rounded-lg border border-app bg-card">
            <table className="w-full min-w-[1080px] border-collapse text-sm">
              <thead className="sticky top-0 z-10 bg-card-muted">
                <tr>
                  {["Ticker", "Status", "Analyzer", "Latest Action", "Risk", "Score", "Gaps", "Last Evaluated", "Accepted", "Actions"].map(label => (
                    <th key={label} className="border-b border-app px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(({ idea, evaluation, activeJob }) => (
                  (() => {
                    const ideaId = String(idea.id)
                    const isDeleting = deletingIdeaIds.has(ideaId)
                    return (
                      <tr
                        key={idea.id}
                        onClick={() => navigate(`/ideas/${idea.id}`, { state: { from: "ideas" } })}
                        className={cn("cursor-pointer border-b border-app transition-colors hover:bg-hover", isDeleting && "opacity-60")}
                      >
                        <td className="px-3 py-3">
                          <div className="font-semibold text-app">{idea.ticker}</div>
                          <div className="max-w-[14rem] truncate text-xs text-subtle">{idea.company_name || "N/A"}</div>
                        </td>
                        <td className="px-3 py-3"><StatusPill status={activeJob ? "researching" : idea.status} /></td>
                        <td className="px-3 py-3">
                          <select
                            value={analyzerDirection(idea)}
                            onClick={event => event.stopPropagation()}
                            onChange={event => {
                              event.stopPropagation()
                              directionMutation.mutate({
                                ideaId,
                                direction: event.target.value as IdeaAnalyzerDirection,
                              })
                            }}
                            disabled={directionMutation.isPending && directionMutation.variables?.ideaId === ideaId}
                            className="h-9 rounded-md border border-app bg-card px-2 text-sm text-app"
                            aria-label={`Analyzer direction for ${idea.ticker}`}
                          >
                            {ANALYZER_DIRECTIONS.map(option => (
                              <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                          </select>
                        </td>
                        <td className="px-3 py-3">{activeJob ? <StatusBadge tone="info">Running</StatusBadge> : <ActionPill action={evaluation?.action} />}</td>
                        <td className="px-3 py-3"><AnalyzerRiskBadges context={evaluation?.analyzer_context ?? null} /></td>
                        <td className="px-3 py-3 font-mono text-app">{scoreText(evaluation?.score)}</td>
                        <td className="px-3 py-3 text-app">{missingCount(evaluation)}</td>
                        <td className="px-3 py-3 text-subtle">{formatDate(evaluation?.evaluated_at)}</td>
                        <td className="px-3 py-3">
                          {idea.accepted_recommendation_id ? <StatusBadge tone="success">Yes</StatusBadge> : <span className="text-subtle">No</span>}
                        </td>
                        <td className="px-3 py-3">
                          <button
                            type="button"
                            onClick={event => {
                              event.stopPropagation()
                              deleteMutation.mutate(ideaId)
                            }}
                            disabled={isDeleting}
                            className="theme-button-base theme-button-secondary min-h-9 px-2 text-sm disabled:pointer-events-none disabled:opacity-50"
                            aria-label={`Delete ${idea.ticker}`}
                            title="Delete idea"
                          >
                            <Trash2 size={15} aria-hidden="true" />
                            <span className="sr-only">Delete</span>
                          </button>
                        </td>
                      </tr>
                    )
                  })()
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {directionMutation.error && (
        <div className="mt-4">
          <ErrorMessage message={directionMutation.error instanceof Error ? directionMutation.error.message : "Could not update analyzer direction."} />
        </div>
      )}

      <section className="mt-5">
        <AnalyzerWorkbench
          universeMode="portfolio_plus_ideas"
          title="Idea Analyzer"
          subtitle="Manual analyzer run using current portfolio rows plus watchlist ideas marked long or short."
          stateKey={IDEA_WATCHLIST_ANALYZER_STATE_KEY}
        />
      </section>
    </main>
  )
}
