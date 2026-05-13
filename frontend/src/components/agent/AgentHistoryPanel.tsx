import { History, Loader2, Trash2 } from "lucide-react"
import type { SessionSummary } from "@/hooks/useAgentChat"

interface AgentHistoryPanelProps {
  history: SessionSummary[]
  loading: boolean
  onLoadSession: (sessionId: string) => void | Promise<void>
  onDeleteSession: (sessionId: string) => void | Promise<void>
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return ""
  try {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    if (days === 0) return "Today"
    if (days === 1) return "Yesterday"
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  } catch {
    return ""
  }
}

export function AgentHistoryPanel({
  history,
  loading,
  onLoadSession,
  onDeleteSession,
}: AgentHistoryPanelProps) {
  return (
    <div className="flex-1 overflow-y-auto bg-app px-4 py-4">
      <div className="mx-auto w-full max-w-[48rem]">
        {loading ? (
          <div className="flex h-44 flex-col items-center justify-center gap-3 text-sm text-muted">
            <Loader2 size={20} className="animate-spin" aria-hidden="true" />
            <span>Loading history...</span>
          </div>
        ) : history.length === 0 ? (
          <div className="flex h-44 flex-col items-center justify-center rounded-xl border border-app bg-card-muted text-center">
            <History size={24} className="mb-2 text-muted" aria-hidden="true" />
            <p className="text-sm font-medium text-app">No saved conversations</p>
            <p className="mt-1 text-xs text-muted">New sessions appear here after they are summarized.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-2">
            {history.map(session => (
              <article
                key={session.session_id}
                className="group flex items-start gap-3 rounded-xl border border-app bg-card p-3 shadow-sm transition-colors hover:bg-card-muted"
              >
                <button
                  type="button"
                  onClick={() => onLoadSession(session.session_id)}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold text-app">
                      {formatDate(session.ended_at ?? session.started_at)}
                    </span>
                    <span className="rounded-md border border-app bg-card-muted px-1.5 py-0.5 text-[10px] text-muted">
                      {session.message_count} msgs
                    </span>
                    {session.key_tickers && session.key_tickers.length > 0 && (
                      <span className="font-mono text-[10px] text-link">
                        {session.key_tickers.slice(0, 4).join(", ")}
                      </span>
                    )}
                  </div>
                  {session.summary ? (
                    <p className="line-clamp-2 text-xs leading-5 text-muted">{session.summary}</p>
                  ) : session.key_topics && session.key_topics.length > 0 ? (
                    <p className="line-clamp-2 text-xs leading-5 text-muted">{session.key_topics.join(" · ")}</p>
                  ) : (
                    <p className="text-xs text-subtle">Untitled conversation</p>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteSession(session.session_id)}
                  className="theme-icon-button h-8 w-8 shrink-0 opacity-80 transition-opacity hover:text-negative sm:opacity-0 sm:group-hover:opacity-100"
                  aria-label="Delete conversation"
                  title="Delete"
                >
                  <Trash2 size={13} aria-hidden="true" />
                </button>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
