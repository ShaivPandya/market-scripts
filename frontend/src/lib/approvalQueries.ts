import type { QueryClient } from "@tanstack/react-query"

import type { ApprovalRecord, ApprovalSummaryParams, ApprovalSummaryResponse } from "@/lib/api"

const APPROVAL_SUMMARY_QUERY_ROOT = ["approvals", "summary"] as const

const THESIS_FACING_ACTIONS = new Set([
  "change_thesis_status",
  "save_thesis_content",
  "save_evaluation",
  "create_catalyst",
  "update_catalyst_status",
  "create_kill_condition",
  "update_kill_condition_status",
  "create_thesis_claim",
  "update_thesis_claim",
  "create_recommendation",
  "create_research_note",
])

const PORTFOLIO_ACTIONS = new Set(["update_portfolio_positions", "update_hedge_positions"])

function normalizedApprovalSummaryParams(params: ApprovalSummaryParams = {}) {
  return {
    status: params.status ?? "pending",
    ticker: params.ticker ? params.ticker.trim().toUpperCase() : undefined,
    application_status: params.application_status,
    limit: params.limit ?? 5,
  }
}

export function approvalSummaryQueryKey(params: ApprovalSummaryParams = {}) {
  return [...APPROVAL_SUMMARY_QUERY_ROOT, normalizedApprovalSummaryParams(params)] as const
}

function approvalStatus(approval: ApprovalRecord): string {
  return String(approval.status || "pending")
}

function approvalApplicationStatus(approval: ApprovalRecord): string {
  return String(approval.application_status || "pending")
}

function hasRecommendationApproval(approval: ApprovalRecord | undefined): boolean {
  return approval?.proposed_change?.recommendation_id != null
}

function approvalMatchesSummary(summary: ApprovalSummaryResponse, approval: ApprovalRecord): boolean {
  if (summary.status && approvalStatus(approval) !== summary.status) return false
  if (summary.ticker && String(approval.ticker || "").toUpperCase() !== summary.ticker) return false
  if (summary.application_status && approvalApplicationStatus(approval) !== summary.application_status) return false
  return true
}

export function patchResolvedApprovalSummaries(
  queryClient: QueryClient,
  resolvedApproval: ApprovalRecord,
  previousApproval?: ApprovalRecord,
) {
  const approval = previousApproval ?? resolvedApproval
  queryClient.setQueriesData<ApprovalSummaryResponse>(
    { queryKey: APPROVAL_SUMMARY_QUERY_ROOT },
    current => {
      if (!current) return current
      const id = approval.id
      const beforeMatches = approvalMatchesSummary(current, approval)
      const removedItem = current.items.find(item => item.id === id)
      const items = current.items.filter(item => item.id !== id)
      const decrementPending = current.status === "pending" && beforeMatches
      const count = decrementPending ? Math.max(0, current.count - 1) : current.count
      const recommendationApprovalCount =
        decrementPending && hasRecommendationApproval(removedItem ?? approval)
          ? Math.max(0, current.recommendation_approval_count - 1)
          : current.recommendation_approval_count

      return {
        ...current,
        count,
        items,
        recommendation_approval_count: recommendationApprovalCount,
        has_more: count > items.length,
      }
    },
  )
}

export function invalidateApprovalSummaries(queryClient: QueryClient) {
  return queryClient.invalidateQueries({ queryKey: APPROVAL_SUMMARY_QUERY_ROOT })
}

export function invalidateAfterApprovalResolution(queryClient: QueryClient, approval?: ApprovalRecord) {
  void invalidateApprovalSummaries(queryClient)
  void queryClient.invalidateQueries({ queryKey: ["workspace"] })
  const ticker = approval?.ticker ? String(approval.ticker).toUpperCase() : null
  if (ticker) void queryClient.invalidateQueries({ queryKey: ["dossier", ticker] })
  const actionId = String(approval?.action_id || "")
  if (THESIS_FACING_ACTIONS.has(actionId)) void queryClient.invalidateQueries({ queryKey: ["thesis"] })
  if (PORTFOLIO_ACTIONS.has(actionId)) void queryClient.invalidateQueries({ queryKey: ["portfolio", "all_timeframes"] })
}

export function shouldRefetchApprovalSummariesAfterError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)
  return /already|conflict|not found|no pending|404|409/i.test(message)
}
