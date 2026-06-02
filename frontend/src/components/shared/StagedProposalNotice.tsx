import type { ReactNode } from "react"
import { Link } from "react-router-dom"

import { DecisionStateBadge, EffectScopeBadge } from "@/components/shared/DecisionStateBadge"
import type { StagedMutationResponse } from "@/lib/api"
import { cn } from "@/lib/utils"

export type StagedProposal = {
  approval_id?: string | number | null
  decision_state?: StagedMutationResponse["decision_state"]
  effect_scope?: StagedMutationResponse["effect_scope"]
  review_route?: string | null
}

interface StagedProposalNoticeProps {
  proposal: StagedProposal | null | undefined
  children?: ReactNode
  className?: string
  showReviewLink?: boolean
  reviewLabel?: string
}

// eslint-disable-next-line react-refresh/only-export-components
export function formatApprovalDisplayLabel(approvalId: string | number | null | undefined, label = "Proposal"): string {
  const value = String(approvalId ?? "").trim()
  if (!value) return label
  return /^\d+$/.test(value) ? `${label} #${value}` : label
}

export function StagedProposalNotice({
  proposal,
  children,
  className,
  showReviewLink = false,
  reviewLabel = "Review",
}: StagedProposalNoticeProps) {
  if (!proposal) return null

  return (
    <div className={cn("rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <DecisionStateBadge state={proposal.decision_state ?? "pending_approval"} />
        <EffectScopeBadge scope={proposal.effect_scope ?? "internal_state"} />
        <span>
          {formatApprovalDisplayLabel(proposal.approval_id)}{" "}
          {children ?? "staged. Approval is required before app state changes."}
        </span>
        {showReviewLink && (
          <Link
            to={proposal.review_route ?? "/workspace"}
            className="font-semibold underline underline-offset-2"
          >
            {reviewLabel}
          </Link>
        )}
      </div>
    </div>
  )
}
