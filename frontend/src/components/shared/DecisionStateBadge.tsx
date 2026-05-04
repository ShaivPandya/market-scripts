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
  return (
    <span className={cn("theme-badge", toneClass(tone), className)} aria-label={`Decision state: ${label}`}>
      <Icon size={12} aria-hidden="true" />
      {label}
    </span>
  )
}

export function PolicyStateBadge({ state }: { state: string | null | undefined }) {
  return <StatusBadge tone={policyTone(state)}>Policy {humanizeDecisionValue(state || "missing")}</StatusBadge>
}

export function QualityStateBadge({ state }: { state: string | null | undefined }) {
  return <StatusBadge tone={qualityTone(state)}>Data {humanizeDecisionValue(state || "missing")}</StatusBadge>
}

export function EffectScopeBadge({ scope }: { scope: string | null | undefined }) {
  const label = scope === "read_only"
    ? "Read-only"
    : scope === "internal_state"
      ? "Internal State"
      : scope === "external_execution"
        ? "External Execution"
        : humanizeDecisionValue(scope || "unknown")
  const tone: StatusTone = scope === "external_execution" ? "error" : scope === "internal_state" ? "warning" : "neutral"
  return <StatusBadge tone={tone}>{label}</StatusBadge>
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
