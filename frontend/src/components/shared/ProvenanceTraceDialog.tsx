import { useMemo, useState } from "react"
import { GitBranch, Link2, ShieldCheck } from "lucide-react"

import { Dialog } from "@/components/shared/Dialog"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchProvenanceTrace,
  type ProvenanceGraphEdge,
  type ProvenanceGraphNode,
  type ProvenanceSelector,
  type ProvenanceTrace,
} from "@/lib/api"
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

function traceItems<T>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : []
}

function timelineLabel(item: Record<string, unknown>): string {
  const kind = String(item.kind ?? "")
  if (kind === "node") return [item.node_type, item.label, item.id].filter(Boolean).join(" · ") || "node"
  if (kind === "edge") return [item.edge_type, item.source_node_id, "->", item.target_node_id].filter(Boolean).join(" ")
  return kind || "timeline item"
}

function statusClass(status: unknown): string {
  const value = String(status ?? "").toLowerCase()
  if (["succeeded", "completed", "ok", "approved"].includes(value)) return "text-emerald-600 dark:text-emerald-400"
  if (["failed", "error", "denied", "rejected"].includes(value)) return "text-red-600 dark:text-red-400"
  return "text-muted"
}

function TraceBody({ selector }: { selector: ProvenanceSelector }) {
  const [direction, setDirection] = useState<"both" | "upstream" | "downstream">(selector.direction ?? "both")
  const [depth, setDepth] = useState<number>(selector.max_depth ?? 3)
  const querySelector = useMemo(
    () => ({ ...selector, direction, max_depth: depth }),
    [selector, direction, depth],
  )
  const { data, isPending, error } = useApiQuery<ProvenanceTrace>(
    ["provenance", querySelector],
    () => fetchProvenanceTrace(querySelector),
    30_000,
  )

  if (isPending) return <LoadingSpinner message="Loading provenance trace..." />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const nodes = traceItems<ProvenanceGraphNode>(data.nodes)
  const edges = traceItems<ProvenanceGraphEdge>(data.edges)
  const eventNodes = nodes.filter(node => node.node_type === "event")
  const referenceNodes = nodes.filter(node => node.node_type !== "event")
  const timeline = traceItems<Record<string, unknown>>(data.timeline)
  const warnings = Array.isArray(data.warnings) ? data.warnings : []
  const counts = data.counts
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-app px-3 py-2">
        <label className="flex items-center gap-2 text-xs font-medium text-muted">
          Direction
          <select
            className="rounded-md border border-app bg-app px-2 py-1 text-xs text-app"
            value={direction}
            onChange={event => setDirection(event.target.value as "both" | "upstream" | "downstream")}
          >
            <option value="both">Both</option>
            <option value="upstream">Upstream</option>
            <option value="downstream">Downstream</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs font-medium text-muted">
          Depth
          <select
            className="rounded-md border border-app bg-app px-2 py-1 text-xs text-app"
            value={depth}
            onChange={event => setDepth(Number(event.target.value))}
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map(value => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>
        {data.truncated && <span className="text-xs font-medium text-amber-600 dark:text-amber-400">Trace truncated</span>}
      </div>

      {warnings.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {warnings.map((warning, index) => (
            <span key={`${warning.code}-${index}`} className="rounded-md border border-app px-2 py-1 text-xs text-muted">
              {warning.code.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Nodes", counts?.nodes ?? nodes.length],
          ["Edges", counts?.edges ?? edges.length],
          ["Events", counts?.events ?? eventNodes.length],
          ["References", counts?.references ?? referenceNodes.length],
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
                <span className="ml-auto text-xs text-subtle">{displayTime(item.timestamp ?? item.at)}</span>
              </div>
              <div className="mt-1 break-words text-sm text-app">{timelineLabel(item)}</div>
            </div>
          ))}
        </div>
      </section>

      {eventNodes.length > 0 && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
            <ShieldCheck size={15} />
            Events
          </h3>
          <div className="space-y-1 text-xs text-muted">
            {eventNodes.slice(0, 8).map((event, index) => (
              <div key={event.id} className="flex flex-wrap gap-x-2 gap-y-1">
                <span className="font-medium text-app">{event.event_type ?? "event"}</span>
                <span>{event.event_name ?? event.label ?? `#${index + 1}`}</span>
                <span className={statusClass(event.status)}>{event.status}</span>
                <span className="font-mono text-[11px] text-subtle">{event.id}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {edges.length > 0 && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
            <Link2 size={15} />
            Edges
          </h3>
          <div className="space-y-1 text-xs text-muted">
            {edges.slice(0, 12).map((edge, index) => (
              <div key={String(edge.id ?? index)} className="break-words">
                <span className="font-mono text-[11px] text-subtle">
                  {String(edge.source_ref_type ?? edge.source_node_id ?? "source")}:{String(edge.source_ref_id ?? "")}
                </span>{" "}
                <span className="font-medium text-app">{String(edge.link_type ?? edge.relation_type ?? "relates")}</span>{" "}
                <span className="font-mono text-[11px] text-subtle">
                  {String(edge.target_ref_type ?? edge.target_node_id ?? "target")}:{String(edge.target_ref_id ?? "")}
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
