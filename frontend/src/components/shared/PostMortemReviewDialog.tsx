import { useState } from "react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton } from "@/components/shared/ActionButton"
import { finalizeDecisionOutcome, type DecisionOutcomeRecord } from "@/lib/api"

interface PostMortemReviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  outcome: DecisionOutcomeRecord | null
  onFinalized?: () => void
}

export function PostMortemReviewDialog({
  open,
  onOpenChange,
  outcome,
  onFinalized,
}: PostMortemReviewDialogProps) {
  const [decision, setDecision] = useState<"confirm" | "correct" | "reject">("confirm")
  const [note, setNote] = useState("")
  const [correctedPostmortem, setCorrectedPostmortem] = useState("")
  const [lessonsLearned, setLessonsLearned] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const outcomeId = String(outcome?.decision_outcome_id || outcome?.object_uid || outcome?.id || "")

  const handleSubmit = async () => {
    if (!outcomeId) return
    setSubmitting(true)
    setError(null)
    try {
      await finalizeDecisionOutcome(outcomeId, {
        decision,
        note: note.trim() || undefined,
        corrected_postmortem: decision === "correct" ? correctedPostmortem.trim() : undefined,
        lessons_learned: lessonsLearned.trim() || undefined,
      })
      onFinalized?.()
      onOpenChange(false)
      setNote("")
      setCorrectedPostmortem("")
      setLessonsLearned("")
      setDecision("confirm")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to finalize post-mortem")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Review Post-Mortem"
      description="Confirm, correct, or reject the draft learning note. Finalized notes feed the decision learning loop."
      maxWidth="max-w-2xl"
    >
      {outcome && (
        <div className="space-y-4">
          <div className="rounded-lg border border-app bg-[hsl(var(--muted-2))] p-3 text-sm">
            <div className="flex flex-wrap gap-2 text-xs text-subtle">
              {outcome.ticker && <span>{outcome.ticker}</span>}
              {outcome.as_of && <span>As of {outcome.as_of}</span>}
              {outcome.process_label && <span>{outcome.process_label.replace(/_/g, " ")}</span>}
              {outcome.source_kind && <span>{outcome.source_kind.replace(/_/g, " ")}</span>}
            </div>
            <p className="mt-2 text-app whitespace-pre-wrap">{outcome.draft_postmortem || "No draft post-mortem text."}</p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-muted">Decision</label>
            <div className="flex flex-wrap gap-2">
              {(["confirm", "correct", "reject"] as const).map(option => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setDecision(option)}
                  className={`rounded-lg border px-3 py-1.5 text-sm capitalize ${
                    decision === option
                      ? "border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                      : "border-app text-muted hover:text-app"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          {decision === "correct" && (
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted">Corrected post-mortem</label>
              <textarea
                value={correctedPostmortem}
                onChange={event => setCorrectedPostmortem(event.target.value)}
                rows={4}
                className="w-full rounded-lg border border-app bg-transparent px-3 py-2 text-sm text-app"
                placeholder="Provide the corrected post-mortem narrative."
              />
            </div>
          )}

          {(decision === "correct" || decision === "reject") && (
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted">Review note</label>
              <textarea
                value={note}
                onChange={event => setNote(event.target.value)}
                rows={3}
                className="w-full rounded-lg border border-app bg-transparent px-3 py-2 text-sm text-app"
                placeholder="Explain the correction or rejection."
              />
            </div>
          )}

          <div className="space-y-2">
            <label className="text-xs font-medium text-muted">Lessons learned (optional)</label>
            <textarea
              value={lessonsLearned}
              onChange={event => setLessonsLearned(event.target.value)}
              rows={2}
              className="w-full rounded-lg border border-app bg-transparent px-3 py-2 text-sm text-app"
              placeholder="What should change in future process?"
            />
          </div>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
              {error}
            </div>
          )}

          <div className="flex flex-wrap justify-end gap-2">
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              disabled={submitting}
              className="rounded-lg border border-app px-3 py-2 text-sm font-medium text-muted hover:text-app disabled:opacity-50"
            >
              Cancel
            </button>
            <ActionButton onClick={handleSubmit} loading={submitting} loadingText="Saving..." className="w-auto px-4">
              Finalize
            </ActionButton>
          </div>
        </div>
      )}
    </Dialog>
  )
}
