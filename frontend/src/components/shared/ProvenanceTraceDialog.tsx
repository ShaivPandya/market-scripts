import { GitBranch, Link2, ShieldCheck } from "lucide-react"

import { Dialog } from "@/components/shared/Dialog"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchProvenanceTrace, type ProvenanceSelector, type ProvenanceTrace } from "@/lib/api"
import { cn } from "@/lib/utils"

interface ProvenanceTraceDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  selector: ProvenanceSelector | null
  title?: string
}

function selectorLabel(selector: ProvenanceSelector): string {
  const entry = Object.entries(selector).find(([key, value]) => key !== "max_depth" && value != null)
  if (!entry) return "provenance trace"
  const [key, value] = entry
  return `${key.replace(/_/g, " ")} ${String(value)}`
}

function displayTime(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return ""
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
}

function timelineLabel(item: Record<string, unknown>): string {
  const kind = String(item.kind ?? "")
  if (kind === "event") {
    return [item.event_type, item.event_name].filter(Boolean).join(" · ") || "event"
  }
  if (kind === "link") {
    return `${String(item.source_ref_type ?? "")}:${String(item.source_ref_id ?? "")} ${String(item.link_type ?? "")} ${String(item.target_ref_type ?? "")}:${String(item.target_ref_id ?? "")}`
  }
  if (kind === "source_record") {
    return `${String(item.source_name ?? "")}:${String(item.record_kind ?? "")}`
  }
  if (kind === "workflow_artifact") {
    return `workflow artifact ${String(item.artifact_type ?? "")}`
  }
  return kind || "timeline item"
}

function statusClass(status: unknown): string {
  const value = String(status ?? "").toLowerCase()
  if (["succeeded", "completed", "ok", "approved"].includes(value)) return "text-emerald-600 dark:text-emerald-400"
  if (["failed", "error", "denied", "rejected"].includes(value)) return "text-red-600 dark:text-red-400"
  return "text-muted"
}

function TraceBody({ selector }: { selector: ProvenanceSelector }) {
  const { data, isPending, error } = useApiQuery<ProvenanceTrace>(
    ["provenance", selector],
    () => fetchProvenanceTrace({ max_depth: 4, ...selector }),
    30_000,
  )

  if (isPending) return <LoadingSpinner message="Loading provenance trace..." />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const timeline = data.timeline ?? []
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Events", data.events.length],
          ["Links", data.links.length],
          ["Sources", data.source_records.length],
          ["Artifacts", data.workflow_artifacts.length],
        ].map(([label, value]) => (
          <div key={String(label)} className="rounded-lg border border-app px-3 py-2">
            <div className="text-[11px] font-medium uppercase tracking-wide text-subtle">{label}</div>
            <div className="mt-1 text-lg font-semibold text-app">{value}</div>
          </div>
        ))}
      </div>

      <section>
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
          <GitBranch size={15} />
          Timeline
        </h3>
        <div className="max-h-[42vh] space-y-2 overflow-y-auto pr-1">
          {timeline.length === 0 && <p className="text-sm text-muted">No lineage records found.</p>}
          {timeline.map((item, index) => (
            <div key={`${String(item.kind ?? "item")}-${index}`} className="rounded-lg border border-app px-3 py-2">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-xs font-medium uppercase text-subtle">{String(item.kind ?? "item")}</span>
                <span className={cn("text-xs", statusClass(item.status))}>{String(item.status ?? "")}</span>
                <span className="ml-auto text-xs text-subtle">{displayTime(item.timestamp)}</span>
              </div>
              <div className="mt-1 break-words text-sm text-app">{timelineLabel(item)}</div>
            </div>
          ))}
        </div>
      </section>

      {data.events.length > 0 && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
            <ShieldCheck size={15} />
            Event Path
          </h3>
          <div className="space-y-1 text-xs text-muted">
            {data.events.slice(0, 8).map(event => (
              <div key={event.id} className="flex flex-wrap gap-x-2 gap-y-1">
                <span className="font-medium text-app">{event.event_type}</span>
                <span>{event.event_name}</span>
                <span className={statusClass(event.status)}>{event.status}</span>
                <span className="font-mono text-[11px] text-subtle">{event.id}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {data.links.length > 0 && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
            <Link2 size={15} />
            Linked Refs
          </h3>
          <div className="space-y-1 text-xs text-muted">
            {data.links.slice(0, 12).map((link, index) => (
              <div key={link.id ?? index} className="break-words">
                <span className="font-mono text-[11px] text-subtle">
                  {link.source_ref_type}:{link.source_ref_id}
                </span>{" "}
                <span className="font-medium text-app">{link.link_type}</span>{" "}
                <span className="font-mono text-[11px] text-subtle">
                  {link.target_ref_type}:{link.target_ref_id}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export function ProvenanceTraceDialog({ open, onOpenChange, selector, title }: ProvenanceTraceDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title ?? "Lineage Trace"}
      description={selector ? selectorLabel(selector) : undefined}
      maxWidth="max-w-5xl"
    >
      {open && selector ? <TraceBody selector={selector} /> : null}
    </Dialog>
  )
}
