import type { ApprovalRecord } from "@/lib/api"

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
