import * as RadixDialog from "@radix-ui/react-dialog"
import {
  AlertTriangle,
  GitBranch,
  Layers,
  ShieldCheck,
  Wrench,
  X,
} from "lucide-react"

import {
  DecisionStateBadge,
  EffectScopeBadge,
  PolicyStateBadge,
  QualityStateBadge,
} from "@/components/shared/DecisionStateBadge"
import { ProvenanceTraceBody } from "@/components/shared/ProvenanceTraceDialog"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { StatusBadge } from "@/components/shared/StatusBadge"
import { useDecisionTrace } from "@/contexts/DecisionTraceContext"
import type { DecisionTraceBlocker, DecisionTraceGate, DecisionTraceModel } from "@/lib/decisionTrace"
import { humanizeDecisionValue, policyTone, qualityTone } from "@/lib/decisionState"
import { cn } from "@/lib/utils"

function BlockersSection({ blockers }: { blockers: DecisionTraceBlocker[] }) {
  if (!blockers.length) {
    return (
      <section>
        <SectionHeading icon={AlertTriangle} title="Blockers" />
        <p className="text-sm text-muted">No active blockers recorded.</p>
      </section>
    )
  }
  return (
    <section>
      <SectionHeading icon={AlertTriangle} title="Blockers" />
      <div className="space-y-2">
        {blockers.map((blocker, index) => (
          <div key={`${blocker.code ?? "blocker"}-${index}`} className="rounded-lg border border-app px-3 py-2 text-sm">
            {blocker.code && <p className="font-mono text-xs text-subtle">{blocker.code}</p>}
            <p className="mt-1 text-app">{blocker.message}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function GatesSection({ gates }: { gates: DecisionTraceGate[] }) {
  if (!gates.length) {
    return (
      <section>
        <SectionHeading icon={ShieldCheck} title="Gates" />
        <p className="text-sm text-muted">No gate trace available for this record.</p>
      </section>
    )
  }
  return (
    <section>
      <SectionHeading icon={ShieldCheck} title="Gates" />
      <div className="space-y-3">
        {gates.map((gate, index) => (
          <div key={`${gate.label}-${index}`} className="rounded-lg border border-app px-3 py-3 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-app">{gate.label}</span>
              <StatusBadge tone={policyTone(gate.status)}>{humanizeDecisionValue(gate.status)}</StatusBadge>
              {gate.originalAction && gate.finalAction && gate.originalAction !== gate.finalAction && (
                <span className="text-xs text-subtle">
                  {humanizeDecisionValue(gate.originalAction)} → {humanizeDecisionValue(gate.finalAction)}
                </span>
              )}
            </div>
            {gate.reasons.length > 0 && (
              <div className="mt-2 space-y-1 text-xs text-muted">
                {gate.reasons.slice(0, 6).map((reason, reasonIndex) => (
                  <p key={`${reason.code ?? "reason"}-${reasonIndex}`}>
                    {reason.code ? <span className="font-mono text-subtle">{reason.code}: </span> : null}
                    {reason.message}
                  </p>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function SourcesSection({ sources }: { sources: DecisionTraceModel["sources"] }) {
  if (!sources.length) {
    return (
      <section>
        <SectionHeading icon={Layers} title="Sources" />
        <p className="text-sm text-muted">No source health or source references attached.</p>
      </section>
    )
  }
  return (
    <section>
      <SectionHeading icon={Layers} title="Sources" />
      <div className="space-y-2">
        {sources.map((source, index) => (
          <div key={`${source.id ?? source.name}-${index}`} className="rounded-lg border border-app px-3 py-2 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-app">{source.name}</span>
              {source.domain && <span className="text-xs text-subtle">{source.domain}</span>}
              {source.status && <StatusBadge tone={qualityTone(source.qualityState ?? source.status)}>{source.status}</StatusBadge>}
              {source.stale && <span className="text-xs text-amber-600 dark:text-amber-400">Stale</span>}
            </div>
            {(source.detail || source.reason) && (
              <p className="mt-1 text-xs text-muted">{source.reason || source.detail}</p>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function ToolsSection({ tools }: { tools: DecisionTraceModel["tools"] }) {
  if (!tools.length) {
    return (
      <section>
        <SectionHeading icon={Wrench} title="Tools" />
        <p className="text-sm text-muted">No tool execution trace captured.</p>
      </section>
    )
  }
  return (
    <section>
      <SectionHeading icon={Wrench} title="Tools" />
      <div className="flex flex-wrap gap-2">
        {tools.map((tool, index) => (
          <span
            key={`${tool.name}-${index}`}
            className={cn(
              "rounded-full border border-app px-2.5 py-1 text-xs",
              tool.blocksActionable ? "text-red-600 dark:text-red-400" : "text-muted",
            )}
            title={tool.message}
          >
            {tool.name}
            {tool.status ? ` · ${tool.status}` : ""}
          </span>
        ))}
      </div>
    </section>
  )
}

function SummarySection({ trace }: { trace: DecisionTraceModel }) {
  const { summary } = trace
  return (
    <section className="rounded-xl border border-app bg-[hsl(var(--muted-2))] px-4 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold text-app">{summary.title}</h3>
        <span className="rounded bg-app px-2 py-0.5 text-[11px] uppercase tracking-wide text-subtle">
          {humanizeDecisionValue(summary.entityKind)}
        </span>
      </div>
      {summary.subtitle && <p className="mt-1 text-sm text-muted">{summary.subtitle}</p>}
      <div className="mt-3 flex flex-wrap gap-2">
        {summary.decisionState && <DecisionStateBadge state={summary.decisionState} />}
        {summary.effectScope && <EffectScopeBadge scope={summary.effectScope} />}
        {summary.policyState && <PolicyStateBadge state={summary.policyState} />}
        {summary.qualityState && <QualityStateBadge state={summary.qualityState} />}
        {summary.lineageState && (
          <StatusBadge tone={qualityTone(summary.lineageState)}>{humanizeDecisionValue(summary.lineageState)} lineage</StatusBadge>
        )}
      </div>
      {(summary.ticker || summary.asOf) && (
        <div className="mt-2 flex flex-wrap gap-3 text-xs text-subtle">
          {summary.ticker && <span>Ticker {summary.ticker}</span>}
          {summary.asOf && <span>As of {summary.asOf}</span>}
        </div>
      )}
      {trace.notes.length > 0 && (
        <div className="mt-3 space-y-1 text-xs text-muted">
          {trace.notes.slice(0, 3).map((note, index) => (
            <p key={`note-${index}`} className="line-clamp-3">{note}</p>
          ))}
        </div>
      )}
    </section>
  )
}

function SectionHeading({
  icon: Icon,
  title,
}: {
  icon: typeof GitBranch
  title: string
}) {
  return (
    <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-app">
      <Icon size={15} />
      {title}
    </h3>
  )
}

function TraceContent({ trace }: { trace: DecisionTraceModel }) {
  return (
    <div className="space-y-6">
      <SummarySection trace={trace} />
      <BlockersSection blockers={trace.blockers} />
      <GatesSection gates={trace.gates} />
      <SourcesSection sources={trace.sources} />
      <ToolsSection tools={trace.tools} />
      {trace.provenanceSelector && (
        <section>
          <SectionHeading icon={GitBranch} title="Provenance" />
          <ProvenanceTraceBody selector={trace.provenanceSelector} />
        </section>
      )}
    </div>
  )
}

export function DecisionTraceDrawer() {
  const { trace, open, closeDecisionTrace } = useDecisionTrace()

  return (
    <RadixDialog.Root
      open={open}
      onOpenChange={nextOpen => {
        if (!nextOpen) closeDecisionTrace()
      }}
    >
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-[65] bg-[hsl(var(--background-overlay))]/35 backdrop-blur-[1px]" />
        <RadixDialog.Content
          className="theme-floating fixed inset-y-0 right-0 z-[70] flex w-full max-w-xl flex-col border-l border-app shadow-2xl focus:outline-none max-sm:max-w-full"
        >
          <div className="flex items-start justify-between gap-4 border-b border-app px-5 py-4">
            <div>
              <RadixDialog.Title className="text-lg font-semibold tracking-[-0.02em] text-app">
                Decision Trace
              </RadixDialog.Title>
              <RadixDialog.Description className="mt-1 text-sm text-muted">
                Summary, blockers, gates, sources, tools, and provenance for this record.
              </RadixDialog.Description>
            </div>
            <RadixDialog.Close asChild>
              <button type="button" className="theme-icon-button h-11 w-11 shrink-0" aria-label="Close decision trace">
                <X size={16} />
              </button>
            </RadixDialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
            {trace ? <TraceContent trace={trace} /> : <LoadingSpinner message="Loading decision trace..." />}
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}

export function DecisionTraceDrawerSafeBody({ trace }: { trace: DecisionTraceModel | null }) {
  if (!trace) return <ErrorMessage message="Decision trace unavailable." />
  return <TraceContent trace={trace} />
}
