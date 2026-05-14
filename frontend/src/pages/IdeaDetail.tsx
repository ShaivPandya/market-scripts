import { useEffect, useMemo, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { FileUp, Loader2, Play, Save, Trash2 } from "lucide-react"

import { EquityOverviewReadView } from "@/components/overview/EquityOverviewReadView"
import { ActionPill, EvaluationPanel } from "@/components/idea/EvaluationPanel"
import { ManagementQualityPreview } from "@/components/idea/ManagementQualityPreview"
import { PositionValuationTab } from "@/components/valuation/PositionValuationTab"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { SelectInput, TextInput } from "@/components/shared/FormControls"
import { formatApprovalDisplayLabel } from "@/components/shared/StagedProposalNotice"
import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import { useDocumentGenerationUpload } from "@/hooks/useDocumentGenerationUpload"
import {
  acceptIdeaEvaluation,
  deleteIdea,
  fetchIdea,
  fetchIdeaEvaluationJob,
  rejectIdea,
  startIdeaEvaluationJob,
  updateIdea,
  type IdeaAction,
  type IdeaAnalyzerDirection,
  type IdeaEvaluationJobResponse,
  type IdeaStatus,
  type InvestmentIdea,
  type StagedMutationResponse,
} from "@/lib/api"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"
import {
  formatDate,
  formatLabel,
  latestEvaluation,
  readActiveJobs,
  scoreText,
  writeActiveJobs,
} from "@/lib/ideaUtils"
import { cn } from "@/lib/utils"

const TABS = ["Overview", "Management Quality", "Valuation", "Thesis", "Evaluation"] as const
type Tab = typeof TABS[number]

const IDEA_STATUSES: { value: IdeaStatus; label: string }[] = [
  { value: "watching", label: "Watching" },
  { value: "researching", label: "Researching" },
  { value: "ready_for_review", label: "Ready" },
  { value: "accepted", label: "Accepted" },
  { value: "rejected", label: "Rejected" },
]

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

function StatusPill({ status }: { status: string }) {
  return <StatusBadge tone={STATUS_TONE[status] ?? "neutral"}>{formatLabel(status)}</StatusBadge>
}

function normalizeTagsString(value: string): string {
  return value.split(",").map(t => t.trim()).filter(Boolean).join(",")
}

function analyzerDirection(idea: InvestmentIdea | null): IdeaAnalyzerDirection {
  const direction = String(idea?.metadata?.analyzer_direction || "inactive").toLowerCase()
  return direction === "long" || direction === "short" ? direction : "inactive"
}

function portfolioContextEnabledForIdea(idea: InvestmentIdea | null): boolean {
  return idea?.metadata?.use_portfolio_context !== false
}

export function IdeaDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [tab, setTab] = useState<Tab>("Overview")
  const [editNotes, setEditNotes] = useState("")
  const [editTags, setEditTags] = useState("")
  const [editStatus, setEditStatus] = useState<IdeaStatus>("watching")
  const [editAnalyzerDirection, setEditAnalyzerDirection] = useState<IdeaAnalyzerDirection>("inactive")
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [uploadMessageIsError, setUploadMessageIsError] = useState(false)
  const [managementUploadMessage, setManagementUploadMessage] = useState<string | null>(null)
  const [managementUploadMessageIsError, setManagementUploadMessageIsError] = useState(false)
  const [acceptMessage, setAcceptMessage] = useState<string | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [jobSnapshot, setJobSnapshot] = useState<IdeaEvaluationJobResponse | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)

  const detailQuery = useQuery({
    queryKey: ["idea", id],
    queryFn: () => fetchIdea(id as string),
    enabled: id != null,
    staleTime: 15_000,
  })
  const detail = detailQuery.data ?? null
  const selectedIdea = detail?.idea ?? null
  const selectedEvaluation = selectedIdea ? latestEvaluation(selectedIdea, detail) : null

  const originalNotes = selectedIdea?.user_notes || ""
  const originalTagsString = (selectedIdea?.tags || []).join(", ")
  const originalStatus = (selectedIdea?.status as IdeaStatus) || "watching"
  const originalAnalyzerDirection = analyzerDirection(selectedIdea)
  const originalUsePortfolioContext = portfolioContextEnabledForIdea(selectedIdea)

  const isDirty = useMemo(() => {
    if (!selectedIdea) return false
    return (
      editNotes !== originalNotes ||
      normalizeTagsString(editTags) !== normalizeTagsString(originalTagsString) ||
      editStatus !== originalStatus ||
      editAnalyzerDirection !== originalAnalyzerDirection
    )
  }, [
    selectedIdea,
    editNotes,
    editTags,
    editStatus,
    editAnalyzerDirection,
    originalNotes,
    originalTagsString,
    originalStatus,
    originalAnalyzerDirection,
  ])

  useEffect(() => {
    if (!selectedIdea) return
    setEditNotes(selectedIdea.user_notes || "")
    setEditTags((selectedIdea.tags || []).join(", "))
    setEditStatus((selectedIdea.status as IdeaStatus) || "watching")
    setEditAnalyzerDirection(analyzerDirection(selectedIdea))
  }, [selectedIdea])

  useEffect(() => {
    if (!id) return
    const jobs = readActiveJobs()
    const persisted = jobs[String(id)] ?? null
    setActiveJobId(persisted)
  }, [id])

  function clearActiveJob() {
    if (!id) return
    const jobs = readActiveJobs()
    if (jobs[String(id)]) {
      delete jobs[String(id)]
      writeActiveJobs(jobs)
    }
    setActiveJobId(null)
  }

  function persistActiveJob(jobId: string) {
    if (!id) return
    const jobs = readActiveJobs()
    jobs[String(id)] = jobId
    writeActiveJobs(jobs)
    setActiveJobId(jobId)
  }

  useEffect(() => {
    if (!id || !activeJobId) return
    let stopped = false

    async function poll() {
      try {
        const job = await fetchIdeaEvaluationJob(activeJobId as string)
        if (stopped) return
        setJobSnapshot(job)
        if (job.status === "done") {
          clearActiveJob()
          await qc.invalidateQueries({ queryKey: ["ideas"] })
          await qc.invalidateQueries({ queryKey: ["idea", id] })
        } else if (job.status === "error" || job.status === "cancelled") {
          clearActiveJob()
          setJobError(job.error || "Evaluation failed.")
        }
      } catch (err) {
        if (stopped) return
        setJobError(err instanceof Error ? err.message : "Unable to poll job.")
      }
    }

    void poll()
    const handle = window.setInterval(() => void poll(), 2500)
    return () => {
      stopped = true
      window.clearInterval(handle)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, activeJobId, qc])

  const updateMutation = useMutation({
    mutationFn: () => {
      if (!selectedIdea) throw new Error("No idea selected.")
      return updateIdea(selectedIdea.id, {
        user_notes: editNotes,
        tags: editTags.split(",").map(t => t.trim()).filter(Boolean),
        status: editStatus,
        analyzer_direction: editAnalyzerDirection,
      })
    },
    onSuccess: data => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", data.idea.id] })
    },
  })

  const evaluateMutation = useMutation({
    mutationFn: ({ ideaId, forceRefresh }: { ideaId: string; forceRefresh?: boolean }) =>
      startIdeaEvaluationJob(ideaId, {
        force_refresh: Boolean(forceRefresh),
        use_portfolio_context: portfolioContextEnabledForIdea(selectedIdea),
      }),
    onSuccess: (job, variables) => {
      setJobError(null)
      setJobSnapshot(job)
      if (job.status === "done") {
        void qc.invalidateQueries({ queryKey: ["ideas"] })
        void qc.invalidateQueries({ queryKey: ["idea", variables.ideaId] })
        return
      }
      persistActiveJob(job.job_id)
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", variables.ideaId] })
    },
  })

  const portfolioContextMutation = useMutation({
    mutationFn: ({ ideaId, enabled }: { ideaId: string; enabled: boolean }) =>
      updateIdea(ideaId, { use_portfolio_context: enabled }),
    onSuccess: data => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", data.idea.id] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (ideaId: string) => deleteIdea(ideaId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      navigate("/ideas")
    },
  })

  const overviewUpload = useDocumentGenerationUpload<{ status: "ok"; ticker: string; content: string }>({
    kind: "overview",
    ticker: selectedIdea?.ticker ?? "",
    onSuccess: async data => {
      setUploadMessageIsError(false)
      setUploadMessage(`Overview saved for ${data.ticker}.`)
      if (selectedIdea) await qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
    },
    onError: message => {
      setUploadMessageIsError(true)
      setUploadMessage(message)
    },
  })

  const managementUpload = useDocumentGenerationUpload<StagedMutationResponse>({
    kind: "management_quality",
    ticker: selectedIdea?.ticker ?? "",
    onSuccess: async result => {
      const proposalLabel = formatApprovalDisplayLabel(result.approval_id).toLowerCase()
      setManagementUploadMessageIsError(false)
      setManagementUploadMessage(
        `Management quality ${proposalLabel} staged.`,
      )
      if (selectedIdea) await qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
      await invalidateApprovalSummaries(qc)
    },
    onError: message => {
      setManagementUploadMessageIsError(true)
      setManagementUploadMessage(message)
    },
  })

  const acceptMutation = useMutation({
    mutationFn: () => {
      if (!selectedIdea || !selectedEvaluation) throw new Error("No evaluation selected.")
      return acceptIdeaEvaluation(selectedIdea.id, selectedEvaluation.id)
    },
    onSuccess: result => {
      setAcceptMessage(
        result.action_proposal
          ? `Recommendation accepted; ${formatApprovalDisplayLabel(result.action_proposal.approval_id)} staged.`
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

  function jobBannerMessage(): string | null {
    if (!activeJobId) return null
    const progress = jobSnapshot?.progress as { phase?: unknown; done?: unknown; total?: unknown } | undefined
    const phase = typeof progress?.phase === "string" ? formatLabel(progress.phase) : "running"
    const done = typeof progress?.done === "number" ? progress.done : null
    const total = typeof progress?.total === "number" && progress.total > 0 ? progress.total : null
    const suffix = done != null && total != null ? ` ${done}/${total}` : ""
    return `Evaluation ${phase}${suffix}`
  }

  if (!id) {
    return (
      <main className="mx-auto w-full max-w-[1500px] px-4 py-5 sm:px-6 lg:px-8">
        <ErrorMessage message="No idea id provided." />
      </main>
    )
  }

  return (
    <main className="mx-auto w-full max-w-[1500px] px-4 py-5 sm:px-6 lg:px-8">
      <div className="mb-4">
        <button
          type="button"
          onClick={() => navigate("/ideas")}
          className="shrink-0 text-sm text-muted hover:text-app"
        >
          &larr; Ideas
        </button>
      </div>

      {detailQuery.isLoading ? (
        <LoadingSpinner message="Loading idea" />
      ) : detailQuery.error ? (
        <ErrorMessage message={`Could not load idea: ${detailQuery.error.message}`} />
      ) : !selectedIdea ? (
        <ErrorMessage message="Idea not found." />
      ) : (
        <>
          <section className="theme-surface mb-4 rounded-lg p-4">
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
	                  disabled={Boolean(activeJobId) || evaluateMutation.isPending || portfolioContextMutation.isPending}
	                  className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
	                >
                  <Play size={16} aria-hidden="true" />
                  Run Evaluation
                </button>
                <button
                  type="button"
                  onClick={() => deleteMutation.mutate(selectedIdea.id)}
                  disabled={deleteMutation.isPending}
                  className="theme-button-base theme-button-secondary min-h-10 px-3 text-sm disabled:pointer-events-none disabled:opacity-50"
                  aria-label="Delete idea"
                  title="Delete idea"
                >
                  <Trash2 size={16} aria-hidden="true" />
                </button>
              </div>
            </div>

            {jobBannerMessage() && (
              <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 px-3 py-3 text-sm text-blue-900 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
                {jobBannerMessage()} ({activeJobId?.slice(0, 8)})
              </div>
            )}
            {jobError && (
              <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                {jobError}
              </div>
            )}
          </section>

          <div className="mb-4 flex w-full max-w-full gap-1 overflow-x-auto overscroll-x-contain border-b border-app [-webkit-overflow-scrolling:touch]">
            {TABS.map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  "shrink-0 px-3 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors",
                  tab === t
                    ? "border-blue-500 text-blue-600 dark:text-blue-400"
                    : "border-transparent text-muted hover:text-app",
                )}
              >
                {t}
                {t === "Evaluation" && detail?.evaluations && detail.evaluations.length > 0 && (
                  <span className="ml-1 text-xs text-subtle">({detail.evaluations.length})</span>
                )}
              </button>
            ))}
          </div>

          <div className="theme-surface rounded-lg p-4">
            {tab === "Overview" && (
              <section>
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="section-title text-sm">Overview</h3>
                    <p className="mt-1 text-xs text-subtle">
                      {detail?.documents?.overview_present ? "Stored" : "Missing"}
                      {detail?.documents?.thesis_present ? " / thesis stored" : ""}
                      {detail?.documents?.management_quality_present ? " / management stored" : ""}
                    </p>
                  </div>
                  <label className={cn(
                    "theme-button-base theme-button-secondary min-h-10 cursor-pointer px-4 text-sm",
                    overviewUpload.isUploading && "pointer-events-none opacity-60",
                  )}>
                    {overviewUpload.isUploading ? (
                      <Loader2 size={16} className="animate-spin" aria-hidden="true" />
                    ) : (
                      <FileUp size={16} aria-hidden="true" />
                    )}
                    {overviewUpload.isUploading ? "Uploading" : "Upload"}
                    <input
                      type="file"
                      accept=".md,.markdown,.pdf,text/markdown,application/pdf"
                      className="hidden"
                      disabled={overviewUpload.isUploading}
                      onChange={event => {
                        const file = event.target.files?.[0]
                        event.currentTarget.value = ""
                        if (file) {
                          setUploadMessage(null)
                          setUploadMessageIsError(false)
                          void overviewUpload.startUpload(file)
                        }
                      }}
                    />
                  </label>
                </div>
                {uploadMessage && (
                  uploadMessageIsError ? (
                    <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                      {uploadMessage}
                    </p>
                  ) : (
                    <p className="mb-2 text-xs font-medium text-green-600 dark:text-green-400">{uploadMessage}</p>
                  )
                )}
                {detail?.documents?.overview_error && (
                  <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                    {detail.documents.overview_error}
                  </p>
                )}
                {detail?.documents?.overview_content ? (
                  <EquityOverviewReadView
                    content={detail.documents.overview_content}
                    parsed={detail.documents.overview_parsed ?? null}
                    ticker={selectedIdea.ticker}
                  />
                ) : (
                  <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">No overview stored.</p>
                )}
              </section>
            )}

            {tab === "Management Quality" && (
              <section>
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="section-title text-sm">Management Quality</h3>
                    <p className="mt-1 text-xs text-subtle">
                      {detail?.documents?.management_quality_present ? "Stored" : "Missing"}
                    </p>
                  </div>
                  <label className={cn(
                    "theme-button-base theme-button-secondary min-h-10 cursor-pointer px-4 text-sm",
                    managementUpload.isUploading && "pointer-events-none opacity-60",
                  )}>
                    {managementUpload.isUploading ? (
                      <Loader2 size={16} className="animate-spin" aria-hidden="true" />
                    ) : (
                      <FileUp size={16} aria-hidden="true" />
                    )}
                    {managementUpload.isUploading ? "Uploading" : "Upload"}
                    <input
                      type="file"
                      accept=".md,.markdown,.pdf,text/markdown,application/pdf"
                      className="hidden"
                      disabled={managementUpload.isUploading}
                      onChange={event => {
                        const file = event.target.files?.[0]
                        event.currentTarget.value = ""
                        if (file) {
                          setManagementUploadMessage(null)
                          setManagementUploadMessageIsError(false)
                          void managementUpload.startUpload(file)
                        }
                      }}
                    />
                  </label>
                </div>
                {managementUploadMessage && (
                  managementUploadMessageIsError ? (
                    <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                      {managementUploadMessage}
                    </p>
                  ) : (
                    <p className="mb-2 text-xs text-subtle">{managementUploadMessage}</p>
                  )
                )}
                {detail?.documents?.management_quality_error && (
                  <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                    {detail.documents.management_quality_error}
                  </p>
                )}
                {detail?.documents?.management_quality_content ? (
                  <ManagementQualityPreview
                    content={detail.documents.management_quality_content}
                    parsed={detail.documents.management_quality_parsed ?? null}
                  />
                ) : (
                  <p className="rounded-lg border border-app bg-card-muted px-3 py-4 text-sm text-muted">
                    No management quality assessment stored.
                  </p>
                )}
              </section>
            )}

            {tab === "Valuation" && <PositionValuationTab ticker={selectedIdea.ticker} />}

            {tab === "Thesis" && (
              <section className="space-y-4">
                <div className="grid gap-3 md:grid-cols-[1fr_11rem_11rem]">
                  <TextInput label="Tags" value={editTags} onChange={setEditTags} placeholder="quality, cyclical" />
                  <SelectInput label="Status" value={editStatus} onChange={value => setEditStatus(value as IdeaStatus)} options={IDEA_STATUSES} />
                  <SelectInput
                    label="Analyzer"
                    value={editAnalyzerDirection}
                    onChange={value => setEditAnalyzerDirection(value as IdeaAnalyzerDirection)}
                    options={ANALYZER_DIRECTIONS}
                  />
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
                    disabled={!isDirty || updateMutation.isPending}
                    className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
                  >
                    <Save size={16} aria-hidden="true" />
                    {updateMutation.isPending ? "Saving" : "Save"}
                  </button>
                  {updateMutation.error && <span className="self-center text-sm text-red-600">{updateMutation.error.message}</span>}
                </div>
              </section>
            )}

	            {tab === "Evaluation" && (
	              <section className="space-y-5">
	                <EvaluationPanel
	                  evaluation={selectedEvaluation}
	                  onAccept={() => acceptMutation.mutate()}
	                  onReject={() => rejectMutation.mutate()}
	                  accepting={acceptMutation.isPending}
	                  rejecting={rejectMutation.isPending}
	                  portfolioContextEnabled={originalUsePortfolioContext}
	                  onPortfolioContextChange={enabled => {
	                    if (!selectedIdea) return
	                    portfolioContextMutation.mutate({ ideaId: selectedIdea.id, enabled })
	                  }}
	                  portfolioContextUpdating={portfolioContextMutation.isPending}
	                />
                {acceptMessage && <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-muted">{acceptMessage}</p>}

                {detail?.evaluations && detail.evaluations.length > 1 && (
                  <section className="space-y-3">
                    <h3 className="section-title text-sm">History</h3>
                    <div className="space-y-2">
                      {detail.evaluations.slice(1, 8).map(evaluation => (
                        <div
                          key={evaluation.id}
                          className="w-full rounded-lg border border-app bg-card-muted px-3 py-3"
                        >
                          <div className="flex flex-wrap items-center gap-2">
                            <ActionPill action={evaluation.action as IdeaAction} />
                            <span className="font-mono text-sm text-app">{scoreText(evaluation.score)}</span>
                            <span className="text-sm text-subtle">{formatDate(evaluation.evaluated_at)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </section>
            )}
          </div>
        </>
      )}
    </main>
  )
}
