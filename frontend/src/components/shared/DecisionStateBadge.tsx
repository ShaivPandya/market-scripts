import { Activity, AlertTriangle, CheckCircle2, Clock, FileText, GitPullRequest, ShieldCheck, XCircle } from "lucide-react"

import { StatusBadge, type StatusTone } from "@/components/shared/StatusBadge"
import {
  decisionStateLabel,
  decisionStateTone,
  humanizeDecisionValue,
  policyTone,
  qualityTone,
} from "@/lib/decisionState"
import { cn } from "@/lib/utils"

const STATE_ICONS = {
  draft: FileText,
  analysis: Activity,
  recommendation: Activity,
  proposal: GitPullRequest,
  pending_approval: Clock,
  approved: ShieldCheck,
  rejected: XCircle,
  failed: AlertTriangle,
  applied: CheckCircle2,
  executed: ShieldCheck,
} as const

interface DecisionStateBadgeProps {
  state: string | null | undefined
  className?: string
}

export function DecisionStateBadge({ state, className }: DecisionStateBadgeProps) {
  const normalized = String(state || "draft")
  const Icon = STATE_ICONS[normalized as keyof typeof STATE_ICONS] ?? FileText
  const label = decisionStateLabel(normalized)
  const tone = decisionStateTone(normalized)
  const tooltip = decisionStateDescription(normalized)
  return (
    <span
      className={cn("theme-badge theme-tooltip", toneClass(tone), className)}
      aria-label={`Decision state: ${label}. ${tooltip}`}
      data-tooltip={tooltip}
      title={tooltip}
    >
      <Icon size={12} aria-hidden="true" />
      {label}
    </span>
  )
}

export function PolicyStateBadge({ state }: { state: string | null | undefined }) {
  const normalized = String(state || "missing")
  return (
    <StatusBadge tone={policyTone(normalized)} tooltip={policyStateDescription(normalized)}>
      Policy {humanizeDecisionValue(normalized)}
    </StatusBadge>
  )
}

export function QualityStateBadge({ state }: { state: string | null | undefined }) {
  const normalized = String(state || "missing")
  return (
    <StatusBadge tone={qualityTone(normalized)} tooltip={qualityStateDescription(normalized)}>
      Data {humanizeDecisionValue(normalized)}
    </StatusBadge>
  )
}

export function EffectScopeBadge({ scope }: { scope: string | null | undefined }) {
  const normalized = String(scope || "unknown")
  const label = normalized === "read_only"
    ? "Read-only"
    : normalized === "internal_state"
      ? "Internal State"
      : normalized === "external_execution"
        ? "External Execution"
        : humanizeDecisionValue(normalized)
  const tone: StatusTone = normalized === "external_execution" ? "error" : normalized === "internal_state" ? "warning" : "neutral"
  return <StatusBadge tone={tone} tooltip={effectScopeDescription(normalized)}>{label}</StatusBadge>
}

function decisionStateDescription(state: string): string {
  switch (state) {
    case "draft":
      return "This item is being assembled and has not produced a decision yet."
    case "analysis":
      return "This is analysis-only output. It does not change app state or execute trades."
    case "recommendation":
      return "This is a generated recommendation that may still need review or approval."
    case "proposal":
      return "This staged proposal is ready for review before any state change is applied."
    case "pending_approval":
      return "A human approval decision is required before the staged change can proceed."
    case "approved":
      return "A human approved this proposal, but the resulting state change may not be applied yet."
    case "rejected":
      return "A human rejected this proposal. It remains in the audit history."
    case "failed":
      return "The workflow or recommendation failed and needs investigation before it can be used."
    case "applied":
      return "The approved internal state change has been applied in the app."
    case "executed":
      return "The approved action was sent to an external execution system."
    default:
      return `Decision state: ${humanizeDecisionValue(state)}.`
  }
}

function policyStateDescription(state: string): string {
  switch (state) {
    case "pass":
      return "Policy checks passed. The item still may require human approval."
    case "warn":
      return "Policy checks found warnings. Review the warnings before approving."
    case "review_required":
      return "Policy checks require explicit human review before this can be approved."
    case "blocked":
      return "Policy checks blocked this item. It should not be approved without fixing the issue."
    case "error":
      return "Policy checks could not complete because an error occurred."
    case "missing":
      return "No policy gate result is stored for this item yet. Review it before trusting or approving it."
    default:
      return `Policy state: ${humanizeDecisionValue(state)}.`
  }
}

function qualityStateDescription(state: string): string {
  switch (state) {
    case "ok":
      return "The required data checks passed for this item."
    case "degraded":
      return "Some data is incomplete or lower quality. Review the evidence before relying on it."
    case "stale":
      return "The supporting data is older than expected and may need a refresh."
    case "failed":
      return "Critical data checks failed. Do not rely on this item until the data issue is fixed."
    case "missing":
      return "No data quality result is stored for this item yet."
    default:
      return `Data quality state: ${humanizeDecisionValue(state)}.`
  }
}

function effectScopeDescription(scope: string): string {
  switch (scope) {
    case "read_only":
      return "This item is informational and will not change app state."
    case "internal_state":
      return "Approving this can update internal app state, such as records, actions, or triggers."
    case "external_execution":
      return "Approving this can send an action outside the app, such as an execution request."
    default:
      return `Effect scope: ${humanizeDecisionValue(scope)}.`
  }
}

function toneClass(tone: StatusTone) {
  switch (tone) {
    case "info":
      return "theme-badge-info"
    case "success":
      return "theme-badge-success"
    case "warning":
      return "theme-badge-warning"
    case "error":
      return "theme-badge-error"
    case "neutral":
    default:
      return "theme-badge-neutral"
  }
}
