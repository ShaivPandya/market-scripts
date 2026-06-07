import { useEffect, useState } from "react"
import { CheckCircle2, PencilLine, ThumbsDown, ThumbsUp } from "lucide-react"
import type { AgentMessage } from "@/hooks/agentChatShared"
import {
  fetchAgentResponseFeedback,
  submitAgentResponseFeedback,
  type AgentFeedbackDecision,
  type AgentFeedbackFailureTag,
  type AgentResponseFeedbackRecord,
} from "@/lib/api"
import { AgentResponseFeedbackDialog } from "./AgentResponseFeedbackDialog"

interface AgentFeedbackControlsProps {
  message: AgentMessage
  sessionId: string | null
}

function decisionLabel(decision: AgentFeedbackDecision): string {
  switch (decision) {
    case "approve":
      return "Approved"
    case "reject":
      return "Rejected"
    case "correct":
      return "Corrected"
  }
}

export function AgentFeedbackControls({ message, sessionId }: AgentFeedbackControlsProps) {
  const [feedback, setFeedback] = useState<AgentResponseFeedbackRecord | null>(message.feedback ?? null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [dialogDecision, setDialogDecision] = useState<AgentFeedbackDecision>("approve")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [disclosure, setDisclosure] = useState<string | null>(null)

  const canSubmit = Boolean(sessionId && message.clientTurnId && !message.isStreaming)

  useEffect(() => {
    if (!canSubmit || feedback) return
    let cancelled = false
    void fetchAgentResponseFeedback({
      session_id: sessionId ?? undefined,
      client_turn_id: message.clientTurnId,
    })
      .then(data => {
        if (cancelled) return
        const existing = data.feedback?.[0]
        if (existing) setFeedback(existing)
      })
      .catch(() => {
        // Best-effort hydration; controls remain available for first submission.
      })
    return () => {
      cancelled = true
    }
  }, [canSubmit, feedback, message.clientTurnId, sessionId])

  const openDialog = (decision: AgentFeedbackDecision) => {
    setDialogDecision(decision)
    setError(null)
    setDialogOpen(true)
  }

  const handleSubmit = async (payload: {
    decision: AgentFeedbackDecision
    corrected_response?: string
    failure_tags: AgentFeedbackFailureTag[]
    notes?: string
    eligible_for_training: boolean
  }) => {
    if (!sessionId || !message.clientTurnId) return
    setSubmitting(true)
    setError(null)
    try {
      const response = await submitAgentResponseFeedback({
        session_id: sessionId,
        client_turn_id: message.clientTurnId,
        decision: payload.decision,
        corrected_response: payload.corrected_response,
        failure_tags: payload.failure_tags,
        notes: payload.notes,
        eligible_for_training: payload.eligible_for_training,
      })
      setFeedback(response.feedback)
      setDisclosure(response.disclosure)
      setDialogOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save feedback")
    } finally {
      setSubmitting(false)
    }
  }

  if (!canSubmit) return null

  return (
    <div className="mt-2 border-t border-app pt-2">
      {feedback ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <CheckCircle2 size={12} className="text-positive" aria-hidden="true" />
          <span>
            {decisionLabel(feedback.decision)}
            {feedback.training_eligible ? " · eligible for governed training" : ""}
          </span>
          <button
            type="button"
            onClick={() => openDialog(feedback.decision)}
            className="theme-button-secondary px-2 py-1 text-xs"
          >
            Update
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted">Review this response:</span>
          <button
            type="button"
            onClick={() => openDialog("approve")}
            className="inline-flex items-center gap-1 rounded-md border border-app px-2 py-1 text-xs text-muted hover:text-app"
          >
            <ThumbsUp size={12} aria-hidden="true" />
            Approve
          </button>
          <button
            type="button"
            onClick={() => openDialog("reject")}
            className="inline-flex items-center gap-1 rounded-md border border-app px-2 py-1 text-xs text-muted hover:text-app"
          >
            <ThumbsDown size={12} aria-hidden="true" />
            Reject
          </button>
          <button
            type="button"
            onClick={() => openDialog("correct")}
            className="inline-flex items-center gap-1 rounded-md border border-app px-2 py-1 text-xs text-muted hover:text-app"
          >
            <PencilLine size={12} aria-hidden="true" />
            Correct
          </button>
        </div>
      )}

      <p className="mt-2 text-[11px] leading-5 text-subtle">
        {disclosure
          ?? "Human-reviewed feedback is stored with this trajectory and model version for evaluation and optional governed training use."}
      </p>

      <AgentResponseFeedbackDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        decision={dialogDecision}
        initialContent={message.content}
        submitting={submitting}
        error={error}
        onSubmit={handleSubmit}
      />
    </div>
  )
}
