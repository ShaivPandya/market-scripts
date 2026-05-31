import { AlertTriangle, Clock, GitBranch, Info } from "lucide-react"
import { Link } from "react-router-dom"
import { cn } from "@/lib/utils"

export interface WhatChangedSummary {
  baseline: {
    kind?: string | null
    source_type?: string | null
    source_id?: string | null
    at?: string | null
    days?: number | null
  }
  generated_at: string
  items: WhatChangedItem[]
  counts: {
    total: number
    by_category?: Record<string, number>
    by_severity?: Record<string, number>
    by_change_kind?: Record<string, number>
  }
}

export interface WhatChangedItem {
  object_type: string
  object_uid: string
  ticker?: string | null
  category: string
  change_kind: string
  severity: string
  changed_at: string
  title: string
  summary: string
  before?: Record<string, unknown>
  after?: Record<string, unknown>
}

interface WhatChangedPanelProps {
  summary?: WhatChangedSummary | null
  className?: string
  from?: string
  maxItems?: number
  title?: string
}

function formatTime(value: string | null | undefined): string {
  const text = String(value ?? "").trim()
  if (!text) return "Unknown time"
  const date = new Date(text)
  if (Number.isNaN(date.getTime())) return text
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

function baselineText(summary: WhatChangedSummary | null | undefined): string {
  const baseline = summary?.baseline
  const at = formatTime(baseline?.at)
  if (baseline?.kind === "last_report_run") return `Since latest report, ${at}`
  if (baseline?.kind === "last_workflow_run") return `Since latest workflow, ${at}`
  if (baseline?.kind === "override") return `Since ${at}`
  if (baseline?.days) return `Last ${baseline.days} days`
  return `Since ${at}`
}

function label(value: string | null | undefined): string {
  const text = String(value || "unknown").replace(/_/g, " ")
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function severityClass(severity: string | null | undefined): string {
  const value = String(severity || "info")
  if (value === "critical") return "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
  if (value === "warning") return "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
  return "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300"
}

function severityIcon(severity: string | null | undefined) {
  const value = String(severity || "info")
  if (value === "critical" || value === "warning") return AlertTriangle
  return Info
}

function diffPairs(item: WhatChangedItem): Array<[string, unknown, unknown]> {
  const before = item.before ?? {}
  const after = item.after ?? {}
  return Object.keys(after).slice(0, 3).map(key => [key, before[key], after[key]])
}

function displayValue(value: unknown): string {
  if (value == null || value === "") return "empty"
  if (Array.isArray(value)) return value.map(displayValue).join(", ")
  if (typeof value === "object") return JSON.stringify(value) ?? String(value)
  return String(value)
}

export function WhatChangedPanel({
  summary,
  className,
  from = "workspace",
  maxItems = 8,
  title = "What Changed",
}: WhatChangedPanelProps) {
  const items = summary?.items ?? []
  const visibleItems = items.slice(0, maxItems)
  const hiddenCount = Math.max(0, items.length - visibleItems.length)
  const criticalCount = summary?.counts?.by_severity?.critical ?? 0
  const warningCount = summary?.counts?.by_severity?.warning ?? 0

  return (
    <section className={cn("theme-surface rounded-xl p-4", className)}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-app">
          <Clock size={14} className="text-blue-500" />
          {title}
        </h2>
        <span className="ml-auto text-xs text-subtle">{baselineText(summary)}</span>
      </div>

      <div className="mb-3 flex flex-wrap gap-2 text-xs text-muted">
        <span>{summary?.counts?.total ?? 0} tracked changes</span>
        {criticalCount ? <span>{criticalCount} critical</span> : null}
        {warningCount ? <span>{warningCount} warning</span> : null}
      </div>

      {visibleItems.length === 0 ? (
        <div className="rounded-lg border border-app px-3 py-2 text-sm text-muted">
          No tracked operational changes since the baseline.
        </div>
      ) : (
        <div className="space-y-2">
          {visibleItems.map(item => {
            const Icon = severityIcon(item.severity)
            const pairs = diffPairs(item)
            return (
              <div key={`${item.object_uid}-${item.changed_at}`} className="rounded-lg border border-app px-3 py-2">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className={cn("inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium", severityClass(item.severity))}>
                    <Icon size={12} />
                    {label(item.severity)}
                  </span>
                  <span className="rounded border border-app px-1.5 py-0.5 text-xs text-subtle">
                    {label(item.category)}
                  </span>
                  {item.ticker ? (
                    <Link
                      to={`/dossier/${encodeURIComponent(item.ticker)}`}
                      state={{ from }}
                      className="font-semibold text-blue-600 hover:underline dark:text-blue-400"
                    >
                      {item.ticker}
                    </Link>
                  ) : null}
                  <span className="min-w-0 flex-1 truncate font-medium text-app">{item.title}</span>
                  <span className="text-xs text-subtle">{formatTime(item.changed_at)}</span>
                </div>
                <p className="mt-1 text-xs text-muted">{item.summary}</p>
                {item.change_kind === "updated" && pairs.length > 0 ? (
                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-subtle">
                    {pairs.map(([key, before, after]) => (
                      <span key={key} className="max-w-full truncate">
                        {label(key)}: {displayValue(before)} {"->"} {displayValue(after)}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            )
          })}
          {hiddenCount > 0 ? (
            <div className="flex items-center gap-2 rounded-lg border border-app px-3 py-2 text-xs text-subtle">
              <GitBranch size={12} />
              {hiddenCount} more tracked changes hidden.
            </div>
          ) : null}
        </div>
      )}
    </section>
  )
}
