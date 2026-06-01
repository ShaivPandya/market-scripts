import { Check, History, Loader2, Pencil, Trash2, X } from "lucide-react"
import { useRef, useState, type KeyboardEvent } from "react"
import type { SessionSummary } from "@/hooks/useAgentChat"

interface AgentHistoryPanelProps {
  history: SessionSummary[]
  loading: boolean
  onLoadSession: (sessionId: string) => void | Promise<void>
  onDeleteSession: (sessionId: string) => void | Promise<void>
  onRenameSession: (sessionId: string, title: string) => Promise<void>
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

function groupLabel(dateStr: string | null): string {
  if (!dateStr) return "Older"
  try {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    if (days <= 0) return "Today"
    if (days === 1) return "Yesterday"
    if (days < 7) return "Previous 7 Days"
    if (days < 30) return "Previous 30 Days"
    return "Older"
  } catch {
    return "Older"
  }
}

function displayTitle(session: SessionSummary): string {
  return session.title?.trim() || "Untitled conversation"
}

function sessionSecondaryText(session: SessionSummary): string {
  if (session.summary?.trim()) return session.summary.trim()
  if (session.key_topics?.length) return session.key_topics.join(" · ")
  return "No summary yet"
}

function groupedSessions(history: SessionSummary[]) {
  const groups: { label: string; sessions: SessionSummary[] }[] = []
  for (const session of history) {
    const label = groupLabel(session.ended_at ?? session.started_at)
    let group = groups.find(item => item.label === label)
    if (!group) {
      group = { label, sessions: [] }
      groups.push(group)
    }
    group.sessions.push(session)
  }
  return groups
}

interface HistorySessionRowProps {
  session: SessionSummary
  onLoadSession: (sessionId: string) => void | Promise<void>
  onDeleteSession: (sessionId: string) => void | Promise<void>
  onRenameSession: (sessionId: string, title: string) => Promise<void>
}

function HistorySessionRow({
  session,
  onLoadSession,
  onDeleteSession,
  onRenameSession,
}: HistorySessionRowProps) {
  const title = displayTitle(session)
  const running = Boolean(session.has_active_job)
  const [editing, setEditing] = useState(false)
  const [draftTitle, setDraftTitle] = useState(title)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function startEditing() {
    setDraftTitle(title)
    setError(null)
    setEditing(true)
    window.setTimeout(() => inputRef.current?.select(), 20)
  }

  function cancelEditing() {
    setEditing(false)
    setDraftTitle(title)
    setError(null)
  }

  async function saveTitle() {
    if (saving) return
    const next = draftTitle.replace(/\s+/g, " ").trim()
    if (!next) {
      setError("Title cannot be empty.")
      return
    }
    if (next.length > 80) {
      setError("Title must be 80 characters or fewer.")
      return
    }
    if (next === title) {
      cancelEditing()
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onRenameSession(session.session_id, next)
      setEditing(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename conversation.")
    } finally {
      setSaving(false)
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault()
      void saveTitle()
      return
    }
    if (event.key === "Escape") {
      event.preventDefault()
      cancelEditing()
    }
  }

  return (
    <article className="group rounded-xl border border-app bg-card p-3 shadow-sm transition-colors hover:bg-card-muted">
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={() => onLoadSession(session.session_id)}
          className="min-w-0 flex-1 text-left"
          disabled={editing}
        >
          {editing ? (
            <span className="block h-8" aria-hidden="true" />
          ) : (
            <>
              <span className="flex items-center gap-2 text-sm font-semibold text-app">
                <span className="truncate">{title}</span>
                {running && (
                  <span className="shrink-0 rounded-full bg-[hsl(var(--accent)/0.15)] px-2 py-0.5 text-[10px] font-medium text-[hsl(var(--accent))]">
                    Running
                  </span>
                )}
              </span>
              <span className="mt-1 block line-clamp-2 text-xs leading-5 text-muted">
                {sessionSecondaryText(session)}
              </span>
            </>
          )}
        </button>
        <div className="flex shrink-0 items-center gap-1">
          {!editing && (
            <button
              type="button"
              onClick={startEditing}
              className="theme-icon-button h-8 w-8 opacity-80 transition-opacity sm:opacity-0 sm:group-hover:opacity-100"
              aria-label={`Rename ${title}`}
              title="Rename"
            >
              <Pencil size={13} aria-hidden="true" />
            </button>
          )}
          <button
            type="button"
            onClick={() => onDeleteSession(session.session_id)}
            className="theme-icon-button h-8 w-8 opacity-80 transition-opacity hover:text-negative sm:opacity-0 sm:group-hover:opacity-100"
            aria-label={`Delete ${title}`}
            title="Delete"
          >
            <Trash2 size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {editing && (
        <div className="mt-1 flex min-w-0 items-center gap-1.5">
          <input
            ref={inputRef}
            value={draftTitle}
            onChange={event => {
              setDraftTitle(event.currentTarget.value)
              setError(null)
            }}
            onKeyDown={handleKeyDown}
            disabled={saving}
            aria-label="Conversation title"
            className="h-9 min-w-0 flex-1 rounded-md border border-app bg-app px-2 text-sm font-semibold text-app outline-none focus:border-[hsl(var(--accent))]"
          />
          <button
            type="button"
            onClick={() => void saveTitle()}
            disabled={saving}
            className="theme-icon-button h-9 w-9 shrink-0 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Save conversation title"
            title="Save"
          >
            <Check size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={cancelEditing}
            disabled={saving}
            className="theme-icon-button h-9 w-9 shrink-0 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Cancel conversation title edit"
            title="Cancel"
          >
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-muted">
        <span className="font-semibold text-app">{formatDate(session.ended_at ?? session.started_at)}</span>
        <span className="rounded-md border border-app bg-card-muted px-1.5 py-0.5">
          {session.message_count} msgs
        </span>
        {session.key_tickers && session.key_tickers.length > 0 && (
          <span className="font-mono text-link">
            {session.key_tickers.slice(0, 4).join(", ")}
          </span>
        )}
      </div>
      {error && <p className="mt-2 text-[11px] text-negative">{error}</p>}
    </article>
  )
}

export function AgentHistoryPanel({
  history,
  loading,
  onLoadSession,
  onDeleteSession,
  onRenameSession,
}: AgentHistoryPanelProps) {
  const groups = groupedSessions(history)

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
          <div className="space-y-5">
            {groups.map(group => (
              <section key={group.label} aria-labelledby={`agent-history-${group.label.replace(/\s+/g, "-").toLowerCase()}`}>
                <h3
                  id={`agent-history-${group.label.replace(/\s+/g, "-").toLowerCase()}`}
                  className="mb-2 px-1 text-xs font-semibold uppercase text-subtle"
                >
                  {group.label}
                </h3>
                <div className="grid grid-cols-1 gap-2">
                  {group.sessions.map(session => (
                    <HistorySessionRow
                      key={session.session_id}
                      session={session}
                      onLoadSession={onLoadSession}
                      onDeleteSession={onDeleteSession}
                      onRenameSession={onRenameSession}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
