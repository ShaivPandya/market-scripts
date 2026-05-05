import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Archive,
  CheckCircle2,
  FileUp,
  Play,
  Plus,
  RefreshCw,
  Save,
  XCircle,
} from "lucide-react"

import { EquityOverviewReadView } from "@/components/overview/EquityOverviewReadView"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import { ActionButton, SelectInput, TextInput } from "@/components/shared/FormControls"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  acceptIdeaEvaluation,
  archiveIdea,
  createIdea,
  fetchIdea,
  fetchIdeaEvaluationJob,
  fetchIdeas,
  rejectIdea,
  startIdeaEvaluationJob,
  updateIdea,
  uploadOverviewDocument,
  type IdeaAction,
  type IdeaDetailResponse,
  type IdeaEvaluation,
  type IdeaEvaluationJobResponse,
  type IdeaFactorScore,
  type IdeaMissingInformation,
  type IdeaStatus,
  type InvestmentIdea,
} from "@/lib/api"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"
import { cn } from "@/lib/utils"

const ACTIVE_JOBS_KEY = "idea-watchlist-active-jobs-v1"

const IDEA_STATUSES: { value: IdeaStatus; label: string }[] = [
  { value: "watching", label: "Watching" },
  { value: "researching", label: "Researching" },
  { value: "ready_for_review", label: "Ready" },
  { value: "accepted", label: "Accepted" },
  { value: "rejected", label: "Rejected" },
  { value: "archived", label: "Archived" },
]

const ACTION_TONE: Record<string, StatusTone> = {
  buy: "success",
  watch: "info",
  avoid: "error",
  do_nothing: "neutral",
}

const STATUS_TONE: Record<string, StatusTone> = {
  watching: "neutral",
  researching: "info",
  ready_for_review: "warning",
  accepted: "success",
  rejected: "error",
  archived: "neutral",
}

function readActiveJobs(): Record<string, string> {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(ACTIVE_JOBS_KEY) || "{}")
    if (!parsed || typeof parsed !== "object") return {}
    return Object.fromEntries(
      Object.entries(parsed).filter(([, jobId]) => typeof jobId === "string" && jobId),
    ) as Record<string, string>
  } catch {
    return {}
  }
}

function writeActiveJobs(jobs: Record<string, string>) {
  window.localStorage.setItem(ACTIVE_JOBS_KEY, JSON.stringify(jobs))
}

function formatDate(value?: string | null) {
  if (!value) return "Never"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

function formatLabel(value?: string | null) {
  return String(value || "").replace(/_/g, " ") || "N/A"
}

function scoreText(value?: number | null) {
  return value == null || Number.isNaN(Number(value)) ? "N/A" : `${Math.round(Number(value))}`
}

function latestEvaluation(idea: InvestmentIdea, detail?: IdeaDetailResponse | null): IdeaEvaluation | null {
  const evaluations = detail?.evaluations ?? []
  return evaluations.find(e => e.id === idea.latest_evaluation_id) ?? evaluations[0] ?? idea.latest_evaluation ?? null
}

function missingCount(evaluation?: IdeaEvaluation | null) {
  return evaluation?.missing_information?.length ?? 0
}

function StatusPill({ status }: { status: string }) {
  return <StatusBadge tone={STATUS_TONE[status] ?? "neutral"}>{formatLabel(status)}</StatusBadge>
}

function ActionPill({ action }: { action?: string | null }) {
  if (!action) return <span className="text-sm text-subtle">N/A</span>
  return <StatusBadge tone={ACTION_TONE[action] ?? "neutral"}>{formatLabel(action)}</StatusBadge>
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

function EvaluationPanel({
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

export function IdeaWatchlist() {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [activeJobs, setActiveJobs] = useState<Record<string, string>>(() => readActiveJobs())
  const [jobSnapshots, setJobSnapshots] = useState<Record<string, IdeaEvaluationJobResponse>>({})
  const [jobErrors, setJobErrors] = useState<Record<string, string>>({})
  const [ticker, setTicker] = useState("")
  const [companyName, setCompanyName] = useState("")
  const [tags, setTags] = useState("")
  const [notes, setNotes] = useState("")
  const [editNotes, setEditNotes] = useState("")
  const [editTags, setEditTags] = useState("")
  const [editStatus, setEditStatus] = useState<IdeaStatus>("watching")
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [acceptMessage, setAcceptMessage] = useState<string | null>(null)

  const ideasQuery = useApiQuery(["ideas"], () => fetchIdeas({ include_archived: false, limit: 300 }), 30_000)
  const ideas = useMemo(() => ideasQuery.data?.ideas ?? [], [ideasQuery.data?.ideas])

  useEffect(() => {
    if (selectedId == null && ideas.length > 0) setSelectedId(ideas[0].id)
  }, [ideas, selectedId])

  const detailQuery = useQuery({
    queryKey: ["idea", selectedId],
    queryFn: () => fetchIdea(selectedId as number),
    enabled: selectedId != null,
    staleTime: 15_000,
  })
  const detail = detailQuery.data ?? null
  const selectedIdea = detail?.idea ?? ideas.find(idea => idea.id === selectedId) ?? null
  const selectedEvaluation = selectedIdea ? latestEvaluation(selectedIdea, detail) : null

  useEffect(() => {
    if (!selectedIdea) return
    setEditNotes(selectedIdea.user_notes || "")
    setEditTags((selectedIdea.tags || []).join(", "))
    setEditStatus((selectedIdea.status as IdeaStatus) || "watching")
  }, [selectedIdea])

  function setActiveJobsAndPersist(next: Record<string, string>) {
    setActiveJobs(next)
    writeActiveJobs(next)
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
          setJobSnapshots(prev => ({ ...prev, [ideaId]: job }))
          if (job.status === "done") {
            delete next[ideaId]
            changed = true
            await qc.invalidateQueries({ queryKey: ["ideas"] })
            await qc.invalidateQueries({ queryKey: ["idea", Number(ideaId)] })
          } else if (job.status === "error" || job.status === "cancelled") {
            delete next[ideaId]
            changed = true
            setJobErrors(prev => ({ ...prev, [ideaId]: job.error || "Evaluation failed." }))
          }
        } catch (err) {
          setJobErrors(prev => ({ ...prev, [ideaId]: err instanceof Error ? err.message : "Unable to poll job." }))
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

  const createMutation = useMutation({
    mutationFn: () => createIdea({
      ticker,
      company_name: companyName || null,
      user_notes: notes || null,
      tags: tags.split(",").map(t => t.trim()).filter(Boolean),
    }),
    onSuccess: data => {
      setSelectedId(data.idea.id)
      setTicker("")
      setCompanyName("")
      setTags("")
      setNotes("")
      void qc.invalidateQueries({ queryKey: ["ideas"] })
    },
  })

  const updateMutation = useMutation({
    mutationFn: () => {
      if (!selectedIdea) throw new Error("No idea selected.")
      return updateIdea(selectedIdea.id, {
        user_notes: editNotes,
        tags: editTags.split(",").map(t => t.trim()).filter(Boolean),
        status: editStatus,
      })
    },
    onSuccess: data => {
      setSelectedId(data.idea.id)
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", data.idea.id] })
    },
  })

  const evaluateMutation = useMutation({
    mutationFn: ({ ideaId, forceRefresh }: { ideaId: number; forceRefresh?: boolean }) =>
      startIdeaEvaluationJob(ideaId, { force_refresh: Boolean(forceRefresh) }),
    onSuccess: (job, variables) => {
      setJobErrors(prev => {
        const next = { ...prev }
        delete next[String(variables.ideaId)]
        return next
      })
      setJobSnapshots(prev => ({ ...prev, [String(variables.ideaId)]: job }))
      if (job.status === "done") {
        void qc.invalidateQueries({ queryKey: ["ideas"] })
        void qc.invalidateQueries({ queryKey: ["idea", variables.ideaId] })
        return
      }
      setActiveJobsAndPersist({ ...activeJobs, [String(variables.ideaId)]: job.job_id })
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", variables.ideaId] })
    },
  })

  const archiveMutation = useMutation({
    mutationFn: (ideaId: number) => archiveIdea(ideaId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      setSelectedId(null)
    },
  })

  const uploadMutation = useMutation({
    mutationFn: ({ idea, file }: { idea: InvestmentIdea; file: File }) => uploadOverviewDocument(idea.ticker, file),
    onSuccess: data => {
      setUploadMessage(`Overview saved for ${data.ticker}.`)
      if (selectedIdea) void qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
    },
    onError: err => setUploadMessage(err instanceof Error ? err.message : "Upload failed."),
  })

  const acceptMutation = useMutation({
    mutationFn: () => {
      if (!selectedIdea || !selectedEvaluation) throw new Error("No evaluation selected.")
      return acceptIdeaEvaluation(selectedIdea.id, selectedEvaluation.id)
    },
    onSuccess: result => {
      setAcceptMessage(
        result.action_proposal
          ? `Recommendation accepted; approval #${result.action_proposal.approval_id} staged.`
          : "Recommendation accepted.",
      )
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      if (selectedIdea) void qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
      void invalidateApprovalSummaries(qc)
    },
    onError: err => setAcceptMessage(err instanceof Error ? err.message : "Accept failed."),
  })

  const rejectMutation = useMutation({
    mutationFn: () => {
      if (!selectedIdea) throw new Error("No idea selected.")
      return rejectIdea(selectedIdea.id)
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      if (selectedIdea) void qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
    },
  })

  const rows = useMemo(() => ideas.map(idea => {
    const isSelected = idea.id === selectedId
    const detailForRow = isSelected ? detail : null
    const evaluation = latestEvaluation(idea, detailForRow)
    const activeJob = activeJobs[String(idea.id)]
    return { idea, evaluation, activeJob }
  }), [ideas, selectedId, detail, activeJobs])

  function currentJobResponse(ideaId: number): { jobId: string; message: string } | null {
    const jobId = activeJobs[String(ideaId)]
    if (!jobId) return null
    const progress = jobSnapshots[String(ideaId)]?.progress
    const phase = typeof progress?.phase === "string" ? formatLabel(progress.phase) : "running"
    const done = typeof progress?.done === "number" ? progress.done : null
    const total = typeof progress?.total === "number" && progress.total > 0 ? progress.total : null
    const suffix = done != null && total != null ? ` ${done}/${total}` : ""
    return { jobId, message: `Evaluation ${phase}${suffix}` }
  }

  return (
    <main className="mx-auto w-full max-w-[1500px] px-4 py-5 sm:px-6 lg:px-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-app">Idea Watchlist</h1>
          <p className="mt-1 text-sm text-subtle">{ideas.length} active ideas</p>
        </div>
        <button
          type="button"
          onClick={() => void ideasQuery.refetch()}
          className="theme-button-base theme-button-secondary min-h-10 px-4 text-sm"
        >
          <RefreshCw size={16} aria-hidden="true" />
          Refresh
        </button>
      </div>

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

      <div className="grid gap-5 xl:grid-cols-[minmax(38rem,1fr)_minmax(28rem,0.9fr)]">
        <section className="theme-surface rounded-lg p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="section-title text-sm">Watchlist</h2>
            {ideasQuery.isLoading && <LoadingSpinner message="Loading ideas" />}
          </div>

          {ideasQuery.error ? (
            <ErrorMessage message={`Could not load ideas: ${ideasQuery.error.message}`} />
          ) : rows.length === 0 ? (
            <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">No ideas.</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-app bg-card">
              <table className="w-full min-w-[760px] border-collapse text-sm">
                <thead className="bg-card-muted">
                  <tr>
                    {["Ticker", "Status", "Latest Action", "Score", "Gaps", "Last Evaluated", "Accepted"].map(label => (
                      <th key={label} className="border-b border-app px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ idea, evaluation, activeJob }) => (
                    <tr
                      key={idea.id}
                      onClick={() => setSelectedId(idea.id)}
                      className={cn(
                        "cursor-pointer border-b border-app transition-colors hover:bg-hover",
                        selectedId === idea.id && "bg-card-muted",
                      )}
                    >
                      <td className="px-3 py-3">
                        <div className="font-semibold text-app">{idea.ticker}</div>
                        <div className="max-w-[14rem] truncate text-xs text-subtle">{idea.company_name || "N/A"}</div>
                      </td>
                      <td className="px-3 py-3"><StatusPill status={activeJob ? "researching" : idea.status} /></td>
                      <td className="px-3 py-3">{activeJob ? <StatusBadge tone="info">Running</StatusBadge> : <ActionPill action={evaluation?.action} />}</td>
                      <td className="px-3 py-3 font-mono text-app">{scoreText(evaluation?.score)}</td>
                      <td className="px-3 py-3 text-app">{missingCount(evaluation)}</td>
                      <td className="px-3 py-3 text-subtle">{formatDate(evaluation?.evaluated_at)}</td>
                      <td className="px-3 py-3">
                        {idea.accepted_recommendation_id ? <StatusBadge tone="success">Yes</StatusBadge> : <span className="text-subtle">No</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="theme-surface rounded-lg p-4">
          {!selectedIdea ? (
            <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">Select an idea.</p>
          ) : (
            <div className="space-y-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-xl font-semibold text-app">{selectedIdea.ticker}</h2>
                    <StatusPill status={selectedIdea.status} />
                  </div>
                  <p className="mt-1 text-sm text-subtle">{selectedIdea.company_name || "Company name not set"}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => evaluateMutation.mutate({ ideaId: selectedIdea.id, forceRefresh: true })}
                    disabled={Boolean(activeJobs[String(selectedIdea.id)]) || evaluateMutation.isPending}
                    className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
                  >
                    <Play size={16} aria-hidden="true" />
                    Run Evaluation
                  </button>
                  <button
                    type="button"
                    onClick={() => archiveMutation.mutate(selectedIdea.id)}
                    disabled={archiveMutation.isPending}
                    className="theme-button-base theme-button-secondary min-h-10 px-3 text-sm disabled:pointer-events-none disabled:opacity-50"
                    aria-label="Archive idea"
                  >
                    <Archive size={16} aria-hidden="true" />
                  </button>
                </div>
              </div>

              {currentJobResponse(selectedIdea.id) && (
                <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-3 text-sm text-blue-900 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
                  {currentJobResponse(selectedIdea.id)?.message} ({currentJobResponse(selectedIdea.id)?.jobId.slice(0, 8)})
                </div>
              )}
              {jobErrors[String(selectedIdea.id)] && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                  {jobErrors[String(selectedIdea.id)]}
                </div>
              )}

              <div className="grid gap-3 md:grid-cols-[1fr_11rem]">
                <TextInput label="Tags" value={editTags} onChange={setEditTags} placeholder="quality, cyclical" />
                <SelectInput label="Status" value={editStatus} onChange={value => setEditStatus(value as IdeaStatus)} options={IDEA_STATUSES} />
              </div>
              <div>
                <label className="theme-field-label" htmlFor="idea-notes">Notes</label>
                <textarea
                  id="idea-notes"
                  value={editNotes}
                  onChange={event => setEditNotes(event.target.value)}
                  className="theme-input mt-1 min-h-[120px] w-full resize-y"
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => updateMutation.mutate()}
                  disabled={updateMutation.isPending}
                  className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
                >
                  <Save size={16} aria-hidden="true" />
                  Save
                </button>
                {updateMutation.error && <span className="self-center text-sm text-red-600">{updateMutation.error.message}</span>}
              </div>

              <section className="rounded-lg border border-app bg-card-muted p-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="section-title text-sm">Overview</h3>
                    <p className="mt-1 text-xs text-subtle">
                      {detail?.documents?.overview_present ? "Stored" : "Missing"}
                      {detail?.documents?.thesis_present ? " / thesis stored" : ""}
                    </p>
                  </div>
                  <label className="theme-button-base theme-button-secondary min-h-10 cursor-pointer px-4 text-sm">
                    <FileUp size={16} aria-hidden="true" />
                    Upload
                    <input
                      type="file"
                      accept=".md,.markdown,.pdf,text/markdown,application/pdf"
                      className="hidden"
                      onChange={event => {
                        const file = event.target.files?.[0]
                        event.currentTarget.value = ""
                        if (file) uploadMutation.mutate({ idea: selectedIdea, file })
                      }}
                    />
                  </label>
                </div>
                {uploadMessage && <p className="mt-2 text-xs text-subtle">{uploadMessage}</p>}
                {detail?.documents?.overview_error && (
                  <p className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                    {detail.documents.overview_error}
                  </p>
                )}
                {detail?.documents?.overview_content ? (
                  <div className="mt-4 max-h-[46rem] overflow-y-auto pr-1">
                    <EquityOverviewReadView
                      content={detail.documents.overview_content}
                      parsed={detail.documents.overview_parsed ?? null}
                      ticker={selectedIdea.ticker}
                    />
                  </div>
                ) : (
                  <p className="mt-3 rounded-lg border border-app bg-card px-3 py-4 text-sm text-muted">No overview stored.</p>
                )}
              </section>

              {detailQuery.isLoading ? (
                <LoadingSpinner message="Loading detail" />
              ) : detailQuery.error ? (
                <ErrorMessage message={`Could not load idea: ${detailQuery.error.message}`} />
              ) : (
                <EvaluationPanel
                  evaluation={selectedEvaluation}
                  onAccept={() => acceptMutation.mutate()}
                  onReject={() => rejectMutation.mutate()}
                  accepting={acceptMutation.isPending}
                  rejecting={rejectMutation.isPending}
                />
              )}
              {acceptMessage && <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">{acceptMessage}</p>}

              {detail?.evaluations && detail.evaluations.length > 1 && (
                <section className="space-y-3">
                  <h3 className="section-title text-sm">History</h3>
                  <div className="space-y-2">
                    {detail.evaluations.slice(1, 8).map(evaluation => (
                      <button
                        key={evaluation.id}
                        type="button"
                        className="w-full rounded-lg border border-app bg-card-muted px-3 py-3 text-left transition-colors hover:bg-hover"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <ActionPill action={evaluation.action as IdeaAction} />
                          <span className="font-mono text-sm text-app">{scoreText(evaluation.score)}</span>
                          <span className="text-sm text-subtle">{formatDate(evaluation.evaluated_at)}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}
