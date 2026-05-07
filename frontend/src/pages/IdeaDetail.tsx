import { useEffect, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Archive, FileUp, Play, Save } from "lucide-react"

import { EquityOverviewReadView } from "@/components/overview/EquityOverviewReadView"
import { ActionPill, EvaluationPanel } from "@/components/idea/EvaluationPanel"
import { ManagementQualityPreview } from "@/components/idea/ManagementQualityPreview"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { SelectInput, TextInput } from "@/components/shared/FormControls"
import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import {
  acceptIdeaEvaluation,
  archiveIdea,
  fetchIdea,
  fetchIdeaEvaluationJob,
  rejectIdea,
  startIdeaEvaluationJob,
  updateIdea,
  uploadManagementQualityDocument,
  uploadOverviewDocument,
  type IdeaAction,
  type IdeaEvaluationJobResponse,
  type IdeaStatus,
  type InvestmentIdea,
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

const IDEA_STATUSES: { value: IdeaStatus; label: string }[] = [
  { value: "watching", label: "Watching" },
  { value: "researching", label: "Researching" },
  { value: "ready_for_review", label: "Ready" },
  { value: "accepted", label: "Accepted" },
  { value: "rejected", label: "Rejected" },
  { value: "archived", label: "Archived" },
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

export function IdeaDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [editNotes, setEditNotes] = useState("")
  const [editTags, setEditTags] = useState("")
  const [editStatus, setEditStatus] = useState<IdeaStatus>("watching")
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [managementUploadMessage, setManagementUploadMessage] = useState<string | null>(null)
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

  useEffect(() => {
    if (!selectedIdea) return
    setEditNotes(selectedIdea.user_notes || "")
    setEditTags((selectedIdea.tags || []).join(", "))
    setEditStatus((selectedIdea.status as IdeaStatus) || "watching")
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
      })
    },
    onSuccess: data => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      void qc.invalidateQueries({ queryKey: ["idea", data.idea.id] })
    },
  })

  const evaluateMutation = useMutation({
    mutationFn: ({ ideaId, forceRefresh }: { ideaId: string; forceRefresh?: boolean }) =>
      startIdeaEvaluationJob(ideaId, { force_refresh: Boolean(forceRefresh) }),
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

  const archiveMutation = useMutation({
    mutationFn: (ideaId: string) => archiveIdea(ideaId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["ideas"] })
      navigate("/ideas")
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

  const managementUploadMutation = useMutation({
    mutationFn: ({ idea, file }: { idea: InvestmentIdea; file: File }) =>
      uploadManagementQualityDocument(idea.ticker, file),
    onSuccess: result => {
      setManagementUploadMessage(
        result.approval_id
          ? `Management quality proposal #${result.approval_id} staged.`
          : "Management quality proposal staged.",
      )
      if (selectedIdea) void qc.invalidateQueries({ queryKey: ["idea", selectedIdea.id] })
      void invalidateApprovalSummaries(qc)
    },
    onError: err => setManagementUploadMessage(err instanceof Error ? err.message : "Upload failed."),
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
        <section className="theme-surface rounded-lg p-4">
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
                  disabled={Boolean(activeJobId) || evaluateMutation.isPending}
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

            {jobBannerMessage() && (
              <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-3 text-sm text-blue-900 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
                {jobBannerMessage()} ({activeJobId?.slice(0, 8)})
              </div>
            )}
            {jobError && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                {jobError}
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
                    {detail?.documents?.management_quality_present ? " / management stored" : ""}
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

            <section className="rounded-lg border border-app bg-card-muted p-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="section-title text-sm">Management Quality</h3>
                  <p className="mt-1 text-xs text-subtle">
                    {detail?.documents?.management_quality_present ? "Stored" : "Missing"}
                  </p>
                </div>
                <label className={cn(
                  "theme-button-base theme-button-secondary min-h-10 cursor-pointer px-4 text-sm",
                  managementUploadMutation.isPending && "pointer-events-none opacity-60",
                )}>
                  <FileUp size={16} aria-hidden="true" />
                  {managementUploadMutation.isPending ? "Uploading" : "Upload"}
                  <input
                    type="file"
                    accept=".md,.markdown,.pdf,text/markdown,application/pdf"
                    className="hidden"
                    disabled={managementUploadMutation.isPending}
                    onChange={event => {
                      const file = event.target.files?.[0]
                      event.currentTarget.value = ""
                      if (file) managementUploadMutation.mutate({ idea: selectedIdea, file })
                    }}
                  />
                </label>
              </div>
              {managementUploadMessage && <p className="mt-2 text-xs text-subtle">{managementUploadMessage}</p>}
              {detail?.documents?.management_quality_error && (
                <p className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                  {detail.documents.management_quality_error}
                </p>
              )}
              {detail?.documents?.management_quality_content ? (
                <div className="mt-4 max-h-[46rem] overflow-y-auto pr-1">
                  <ManagementQualityPreview
                    content={detail.documents.management_quality_content}
                    parsed={detail.documents.management_quality_parsed ?? null}
                  />
                </div>
              ) : (
                <p className="mt-3 rounded-lg border border-app bg-card px-3 py-4 text-sm text-muted">
                  No management quality assessment stored.
                </p>
              )}
            </section>

            <EvaluationPanel
              evaluation={selectedEvaluation}
              onAccept={() => acceptMutation.mutate()}
              onReject={() => rejectMutation.mutate()}
              accepting={acceptMutation.isPending}
              rejecting={rejectMutation.isPending}
            />
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
        </section>
      )}
    </main>
  )
}
