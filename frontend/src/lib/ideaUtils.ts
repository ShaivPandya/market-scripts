import type { IdeaDetailResponse, IdeaEvaluation, InvestmentIdea } from "@/lib/api"

export const ACTIVE_JOBS_KEY = "idea-watchlist-active-jobs-current"

export function readActiveJobs(): Record<string, string> {
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

export function writeActiveJobs(jobs: Record<string, string>) {
  window.localStorage.setItem(ACTIVE_JOBS_KEY, JSON.stringify(jobs))
}

export function formatDate(value?: string | null) {
  if (!value) return "Never"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

export function formatLabel(value?: string | null) {
  return String(value || "").replace(/_/g, " ") || "N/A"
}

export function scoreText(value?: number | null) {
  return value == null || Number.isNaN(Number(value)) ? "N/A" : `${Math.round(Number(value))}`
}

export function latestEvaluation(idea: InvestmentIdea, detail?: IdeaDetailResponse | null): IdeaEvaluation | null {
  const evaluations = detail?.evaluations ?? []
  return evaluations.find(e => e.id === idea.latest_evaluation_id) ?? evaluations[0] ?? idea.latest_evaluation ?? null
}

export function missingCount(evaluation?: IdeaEvaluation | null) {
  return evaluation?.missing_information?.length ?? 0
}
