import { TraceTriggerButton } from "@/components/shared/TraceTriggerButton"
import { useDecisionTrace } from "@/contexts/DecisionTraceContext"
import type { RecordTimelineEntry } from "@/lib/api"
import { cn } from "@/lib/utils"

function formatTime(iso: string | null | undefined): string {
  const value = String(iso ?? "").trim()
  if (!value) return "Unknown time"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
}

const KIND_COLORS: Record<string, string> = {
  conviction_change: "text-blue-700 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  thesis_status_change: "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
  lifecycle_event: "text-purple-700 bg-purple-50 dark:text-purple-400 dark:bg-purple-950",
  evaluation: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  idea_evaluation: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  recommendation_accepted: "text-teal-700 bg-teal-50 dark:text-teal-400 dark:bg-teal-950",
  approval_applied: "text-gray-700 bg-gray-100 dark:text-gray-300 dark:bg-gray-800",
}

export function RecordEvolutionTimeline({
  entries,
  title = "Record Evolution",
  limit = 12,
  className,
}: {
  entries?: RecordTimelineEntry[] | null
  title?: string
  limit?: number
  className?: string
}) {
  const { openDecisionTrace } = useDecisionTrace()
  const timeline = (entries ?? []).slice(0, limit)
  if (timeline.length === 0) return null

  return (
    <div className={cn("mt-4 pt-4 border-t border-app", className)}>
      <h3 className="text-xs font-semibold text-subtle uppercase mb-2">{title}</h3>
      <div className="space-y-2">
        {timeline.map((entry, index) => {
          const refs = entry.refs ?? {}
          const canTraceApproval = Boolean(refs.approval_id)
          return (
            <div
              key={String(entry.id ?? `${entry.kind}-${entry.changed_at}-${index}`)}
              className="flex flex-wrap items-start gap-3 text-xs text-muted"
            >
              <span className="w-24 shrink-0 text-subtle">{formatTime(entry.changed_at)}</span>
              <span
                className={cn(
                  "shrink-0 rounded px-1.5 py-0.5 font-medium",
                  KIND_COLORS[entry.kind] ?? "text-gray-700 bg-gray-100 dark:text-gray-300 dark:bg-gray-800",
                )}
              >
                {entry.label}
              </span>
              <span className="min-w-0 flex-1 text-app">{entry.summary}</span>
              {canTraceApproval && (
                <TraceTriggerButton
                  compact
                  label="Trace approval"
                  onClick={() =>
                    openDecisionTrace({
                      kind: "approval",
                      record: { id: refs.approval_id },
                    })
                  }
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
