import axios from "axios"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import {
  fetchDocumentGenerationJob,
  startDocumentGenerationUpload,
  type DocumentGenerationJobResponse,
  type DocumentGenerationKind,
} from "@/lib/api"

const STORAGE_KEY = "document-generation-active-jobs:v1"
const POLL_INTERVAL_MS = 2000
const DEFAULT_TIMEOUT_MS = 1_200_000
const COMPLETION_GRACE_MS = 30_000
const MAX_TRANSIENT_POLL_ERRORS = 5

type ActiveDocumentGenerationStatus = "queued" | "running"

export interface ActiveDocumentGenerationJob {
  kind: DocumentGenerationKind
  ticker: string
  jobId: string
  status: ActiveDocumentGenerationStatus
  startedAt: string
  timeoutS?: number
  filename?: string
}

interface UseDocumentGenerationUploadOptions<TResult> {
  kind: DocumentGenerationKind
  ticker: string
  onSuccess: (result: TResult, job: DocumentGenerationJobResponse<TResult>) => void | Promise<void>
  onError?: (message: string) => void
}

interface UseDocumentGenerationUploadResult {
  activeJob: ActiveDocumentGenerationJob | null
  isUploading: boolean
  isStarting: boolean
  startUpload: (file: File) => Promise<void>
}

function normalizeTicker(ticker: string) {
  return ticker.trim().toUpperCase()
}

function activeJobKey(kind: DocumentGenerationKind, ticker: string) {
  return `${kind}:${normalizeTicker(ticker)}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value)
}

function normalizeStoredJob(value: unknown): ActiveDocumentGenerationJob | null {
  if (!isRecord(value)) return null
  const kind = value.kind
  const ticker = typeof value.ticker === "string" ? normalizeTicker(value.ticker) : ""
  const jobId = typeof value.jobId === "string" ? value.jobId.trim() : ""
  const status = value.status
  const startedAt = typeof value.startedAt === "string" ? value.startedAt : ""
  const timeoutS = typeof value.timeoutS === "number" && Number.isFinite(value.timeoutS)
    ? value.timeoutS
    : undefined
  const filename = typeof value.filename === "string" && value.filename.trim()
    ? value.filename
    : undefined

  if (kind !== "thesis" && kind !== "overview" && kind !== "management_quality") return null
  if (!ticker || !jobId || !startedAt) return null
  if (status !== "queued" && status !== "running") return null
  return { kind, ticker, jobId, status, startedAt, timeoutS, filename }
}

function readStoredJobs(): Record<string, ActiveDocumentGenerationJob> {
  if (typeof window === "undefined") return {}
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (!isRecord(parsed)) return {}
    return Object.entries(parsed).reduce<Record<string, ActiveDocumentGenerationJob>>((acc, [key, value]) => {
      const job = normalizeStoredJob(value)
      if (job) acc[key] = job
      return acc
    }, {})
  } catch {
    return {}
  }
}

function writeStoredJobs(jobs: Record<string, ActiveDocumentGenerationJob>) {
  if (typeof window === "undefined") return
  try {
    if (Object.keys(jobs).length === 0) {
      window.localStorage.removeItem(STORAGE_KEY)
      return
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs))
  } catch {
    // Ignore storage failures; in-memory state still tracks the job for this route lifetime.
  }
}

function readStoredJob(key: string): ActiveDocumentGenerationJob | null {
  return readStoredJobs()[key] ?? null
}

function writeStoredJob(key: string, job: ActiveDocumentGenerationJob | null) {
  const jobs = readStoredJobs()
  if (job) jobs[key] = job
  else delete jobs[key]
  writeStoredJobs(jobs)
}

function jobDeadline(job: ActiveDocumentGenerationJob) {
  const startedAtMs = new Date(job.startedAt).getTime()
  const timeoutMs = Number.isFinite(job.timeoutS)
    ? Math.max(180, Number(job.timeoutS)) * 1000
    : DEFAULT_TIMEOUT_MS
  return (Number.isFinite(startedAtMs) ? startedAtMs : Date.now()) + timeoutMs + COMPLETION_GRACE_MS
}

function isMissingJobError(err: unknown) {
  return (
    (axios.isAxiosError(err) && err.response?.status === 404) ||
    (err instanceof Error && /^404:/.test(err.message))
  )
}

function isRetryableDocumentGenerationError(err: unknown) {
  if (!axios.isAxiosError(err)) return false
  if (!err.response) return true
  return [408, 429, 500, 502, 503, 504].includes(err.response.status)
}

function errorMessage(err: unknown, fallback: string) {
  if (err instanceof Error && err.message.trim()) return err.message
  const text = String(err ?? "").trim()
  return text || fallback
}

function terminalJobMessage(job: DocumentGenerationJobResponse<unknown>) {
  if (job.status === "cancelled") return job.error || "Document generation was cancelled."
  if (job.status === "error") return job.error || "Document generation failed."
  return "Document generation failed."
}

export function useDocumentGenerationUpload<TResult>({
  kind,
  ticker,
  onSuccess,
  onError,
}: UseDocumentGenerationUploadOptions<TResult>): UseDocumentGenerationUploadResult {
  const normalizedTicker = useMemo(() => normalizeTicker(ticker), [ticker])
  const storageKey = useMemo(() => activeJobKey(kind, normalizedTicker), [kind, normalizedTicker])
  const [activeJob, setActiveJob] = useState<ActiveDocumentGenerationJob | null>(() => readStoredJob(storageKey))
  const [isStarting, setIsStarting] = useState(false)
  const mountedRef = useRef(true)
  const onSuccessRef = useRef(onSuccess)
  const onErrorRef = useRef(onError)

  useEffect(() => {
    onSuccessRef.current = onSuccess
  }, [onSuccess])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    setActiveJob(readStoredJob(storageKey))
  }, [storageKey])

  const reportError = useCallback((message: string) => {
    if (!mountedRef.current) return
    onErrorRef.current?.(message)
  }, [])

  const setActiveJobAndPersist = useCallback((next: ActiveDocumentGenerationJob | null) => {
    writeStoredJob(storageKey, next)
    if (mountedRef.current) setActiveJob(next)
  }, [storageKey])

  const handleTerminalJob = useCallback(async (job: DocumentGenerationJobResponse<TResult>) => {
    setActiveJobAndPersist(null)

    if (job.status === "done") {
      if (job.result == null) {
        reportError("Document generation completed without a result.")
        return
      }
      try {
        await onSuccessRef.current(job.result, job)
      } catch (err) {
        reportError(errorMessage(err, "Document generation completed, but the UI failed to refresh."))
      }
      return
    }

    reportError(terminalJobMessage(job as DocumentGenerationJobResponse<unknown>))
  }, [reportError, setActiveJobAndPersist])

  useEffect(() => {
    if (!activeJob) return
    const currentJob = activeJob
    let stopped = false
    let transientPollErrors = 0
    const deadline = jobDeadline(currentJob)

    async function pollActiveJob() {
      if (Date.now() > deadline) {
        setActiveJobAndPersist(null)
        reportError("Timeout: Document generation is taking too long. Try again.")
        return
      }

      try {
        const job = await fetchDocumentGenerationJob<TResult>(currentJob.jobId)
        if (stopped) return
        transientPollErrors = 0

        if (job.status === "queued" || job.status === "running") {
          if (job.status !== currentJob.status || job.timeout_s !== currentJob.timeoutS) {
            setActiveJobAndPersist({
              ...currentJob,
              status: job.status,
              timeoutS: job.timeout_s ?? currentJob.timeoutS,
            })
          }
          return
        }

        await handleTerminalJob(job)
      } catch (err) {
        if (stopped) return
        if (isMissingJobError(err)) {
          setActiveJobAndPersist(null)
          reportError("Document generation job no longer exists. Please upload again.")
          return
        }
        transientPollErrors += 1
        if (!isRetryableDocumentGenerationError(err) || transientPollErrors >= MAX_TRANSIENT_POLL_ERRORS) {
          setActiveJobAndPersist(null)
          reportError(errorMessage(err, "Unable to poll document generation upload."))
        }
      }
    }

    void pollActiveJob()
    const handle = window.setInterval(() => void pollActiveJob(), POLL_INTERVAL_MS)
    return () => {
      stopped = true
      window.clearInterval(handle)
    }
  }, [activeJob, handleTerminalJob, reportError, setActiveJobAndPersist])

  const startUpload = useCallback(async (file: File) => {
    const existing = readStoredJob(storageKey)
    if (existing) {
      if (mountedRef.current) setActiveJob(existing)
      return
    }

    if (mountedRef.current) setIsStarting(true)
    try {
      const started = await startDocumentGenerationUpload<TResult>(kind, normalizedTicker, file)

      if (started.status === "queued" || started.status === "running") {
        const next: ActiveDocumentGenerationJob = {
          kind,
          ticker: normalizedTicker,
          jobId: started.job_id,
          status: started.status,
          startedAt: new Date().toISOString(),
          timeoutS: started.timeout_s,
          filename: file.name || undefined,
        }
        setActiveJobAndPersist(next)
        return
      }

      if (mountedRef.current) await handleTerminalJob(started)
    } catch (err) {
      reportError(errorMessage(err, "Upload failed."))
    } finally {
      if (mountedRef.current) setIsStarting(false)
    }
  }, [handleTerminalJob, kind, normalizedTicker, reportError, setActiveJobAndPersist, storageKey])

  return {
    activeJob,
    isUploading: isStarting || activeJob != null,
    isStarting,
    startUpload,
  }
}
