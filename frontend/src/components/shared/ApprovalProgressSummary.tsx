import type { ApprovalRecord, ApprovalRequirement } from "@/lib/api"
import { cn } from "@/lib/utils"

function requirementLabel(requirement: ApprovalRequirement): string {
  const scope = [requirement.scope_type, requirement.scope_id].filter(Boolean).join(":")
  return scope ? `${requirement.label} (${scope})` : requirement.label
}

export function approvalRemainingCount(approval: ApprovalRecord): number {
  return Number(approval.approval_progress?.remaining_count ?? approval.remaining_approval_requirements?.length ?? 1)
}

export function approvalRecordedCount(approval: ApprovalRecord): number {
  return Number(approval.approval_progress?.recorded_count ?? 0)
}

export function approvalRequiredCount(approval: ApprovalRecord): number {
  return Number(approval.approval_progress?.total_required ?? Math.max(1, approval.approval_requirements?.length ?? 1))
}

export function approvalActionLabel(approval: ApprovalRecord): string {
  if (approval.can_retry_apply) return "Retry Apply"
  return approvalRemainingCount(approval) > 1 ? "Record Approval" : "Approve & Apply"
}

export function ApprovalProgressSummary({
  approval,
  compact = false,
}: {
  approval: ApprovalRecord
  compact?: boolean
}) {
  const progress = approval.approval_progress
  const recorded = approvalRecordedCount(approval)
  const required = approvalRequiredCount(approval)
  const remaining = progress?.remaining_requirements ?? approval.remaining_approval_requirements ?? []
  const completed = Boolean(progress?.completed)

  return (
    <div className={cn("text-xs text-muted", compact ? "mt-1" : "rounded-lg border border-app px-3 py-2")}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-app">{recorded}/{required} approvals recorded</span>
        {completed ? (
          <span className="theme-badge theme-badge-success">Complete</span>
        ) : (
          <span className="theme-badge theme-badge-warning">{approvalRemainingCount(approval)} remaining</span>
        )}
      </div>
      {!completed && remaining.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {remaining.map(requirement => (
            <span key={requirement.id} className="rounded border border-app px-1.5 py-0.5 text-[11px] text-subtle">
              {requirementLabel(requirement)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
