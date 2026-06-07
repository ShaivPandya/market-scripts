import { useState } from "react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton } from "@/components/shared/FormControls"
import type { AgentFeedbackDecision, AgentFeedbackFailureTag } from "@/lib/api"

const FAILURE_TAGS: { id: AgentFeedbackFailureTag; label: string }[] = [
  { id: "routing", label: "Routing" },
  { id: "tools", label: "Tools" },
  { id: "source_quality", label: "Source quality" },
  { id: "synthesis", label: "Synthesis" },
  { id: "calibration", label: "Calibration" },
  { id: "policy_boundary", label: "Policy boundary" },
]

interface AgentResponseFeedbackDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  decision: AgentFeedbackDecision
  initialContent?: string
  submitting?: boolean
  error?: string | null
  onSubmit: (payload: {
    decision: AgentFeedbackDecision
    corrected_response?: string
    failure_tags: AgentFeedbackFailureTag[]
    notes?: string
    eligible_for_training: boolean
  }) => void
}

export function AgentResponseFeedbackDialog({
  open,
  onOpenChange,
  decision,
  initialContent = "",
  submitting = false,
  error = null,
  onSubmit,
}: AgentResponseFeedbackDialogProps) {
  const [activeDecision, setActiveDecision] = useState<AgentFeedbackDecision>(decision)
  const [correctedResponse, setCorrectedResponse] = useState(initialContent)
  const [notes, setNotes] = useState("")
  const [failureTags, setFailureTags] = useState<AgentFeedbackFailureTag[]>([])
  const [eligibleForTraining, setEligibleForTraining] = useState(false)

  const toggleTag = (tag: AgentFeedbackFailureTag) => {
    setFailureTags(current => (current.includes(tag) ? current.filter(item => item !== tag) : [...current, tag]))
  }

  const title =
    activeDecision === "approve"
      ? "Approve Stan response"
      : activeDecision === "reject"
        ? "Reject Stan response"
        : "Correct Stan response"

  const description =
    "Human-reviewed feedback is stored with this trajectory and model version. "
    + "It may be used for evaluation review and, only when you opt in below, governed training datasets."

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={title} description={description} maxWidth="max-w-2xl">
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted">Decision</label>
          <div className="flex flex-wrap gap-2">
            {(["approve", "reject", "correct"] as const).map(option => (
              <button
                key={option}
                type="button"
                onClick={() => setActiveDecision(option)}
                className={`rounded-lg border px-3 py-1.5 text-sm capitalize ${
                  activeDecision === option
                    ? "border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                    : "border-app text-muted hover:text-app"
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        {activeDecision === "correct" && (
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted">Corrected response</label>
            <textarea
              value={correctedResponse}
              onChange={event => setCorrectedResponse(event.target.value)}
              rows={5}
              className="w-full rounded-lg border border-app bg-transparent px-3 py-2 text-sm text-app"
              placeholder="Provide the corrected assistant response."
            />
          </div>
        )}

        {(activeDecision === "reject" || activeDecision === "correct") && (
          <>
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted">Failure categories</label>
              <div className="flex flex-wrap gap-2">
                {FAILURE_TAGS.map(tag => (
                  <button
                    key={tag.id}
                    type="button"
                    onClick={() => toggleTag(tag.id)}
                    className={`rounded-lg border px-2.5 py-1 text-xs ${
                      failureTags.includes(tag.id)
                        ? "border-amber-500 bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
                        : "border-app text-muted hover:text-app"
                    }`}
                  >
                    {tag.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-muted">Review note</label>
              <textarea
                value={notes}
                onChange={event => setNotes(event.target.value)}
                rows={3}
                className="w-full rounded-lg border border-app bg-transparent px-3 py-2 text-sm text-app"
                placeholder="Explain what went wrong or how the answer should improve."
              />
            </div>
          </>
        )}

        <label className="flex items-start gap-2 rounded-lg border border-app bg-[hsl(var(--muted-2))] px-3 py-2 text-xs leading-5 text-muted">
          <input
            type="checkbox"
            checked={eligibleForTraining}
            onChange={event => setEligibleForTraining(event.target.checked)}
            className="mt-0.5"
          />
          <span>
            Allow this human-reviewed label to enter governed training datasets.
            {activeDecision === "approve"
              ? " Approvals with this checked can also promote the trajectory for sanitized export."
              : " Reject and correct labels never auto-promote trajectories."}
          </span>
        </label>

        {error && (
          <p className="text-sm text-negative" role="alert">
            {error}
          </p>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            disabled={submitting}
            className="theme-button-secondary px-4 py-2 text-sm"
          >
            Cancel
          </button>
          <ActionButton
            onClick={() =>
              onSubmit({
                decision: activeDecision,
                corrected_response: activeDecision === "correct" ? correctedResponse.trim() : undefined,
                failure_tags: failureTags,
                notes: notes.trim() || undefined,
                eligible_for_training: eligibleForTraining,
              })
            }
            disabled={submitting || (activeDecision === "correct" && !correctedResponse.trim())}
          >
            {submitting ? "Saving..." : "Save feedback"}
          </ActionButton>
        </div>
      </div>
    </Dialog>
  )
}
