import { ArrowUp, Pencil, Trash2, X } from "lucide-react"
import type { QueuedAgentMessage } from "@/hooks/agentChatShared"

interface AgentChatQueuePanelProps {
  queuedMessages: QueuedAgentMessage[]
  onSendNow: (id: string) => void
  onEdit: (id: string) => void
  onRemove: (id: string) => void
  onClear: () => void
}

export function AgentChatQueuePanel({
  queuedMessages,
  onSendNow,
  onEdit,
  onRemove,
  onClear,
}: AgentChatQueuePanelProps) {
  if (!queuedMessages.length) return null

  return (
    <div className="mx-4 mb-2 rounded-xl border border-app bg-card-muted px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-2 text-xs text-muted">
        <span>
          {queuedMessages.length} message{queuedMessages.length === 1 ? "" : "s"} queued
        </span>
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 transition-colors hover:bg-card hover:text-app"
        >
          <X size={12} aria-hidden="true" />
          Clear all
        </button>
      </div>
      <ul className="space-y-1.5">
        {queuedMessages.map(entry => (
          <li
            key={entry.id}
            className="flex items-start gap-2 rounded-lg border border-app/60 bg-card px-2 py-1.5 text-sm text-app"
          >
            <p className="min-w-0 flex-1 line-clamp-2">{entry.content}</p>
            <div className="flex shrink-0 items-center gap-0.5">
              <button
                type="button"
                onClick={() => onSendNow(entry.id)}
                className="rounded-md p-1 text-muted transition-colors hover:bg-card-muted hover:text-app"
                aria-label="Send queued message now and steer"
                title="Send now (interrupt current response and steer)"
              >
                <ArrowUp size={14} aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onEdit(entry.id)}
                className="rounded-md p-1 text-muted transition-colors hover:bg-card-muted hover:text-app"
                aria-label="Edit queued message"
                title="Edit"
              >
                <Pencil size={14} aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onRemove(entry.id)}
                className="rounded-md p-1 text-muted transition-colors hover:bg-card-muted hover:text-app"
                aria-label="Remove queued message"
                title="Remove"
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
