import { ArrowLeft, Check, History, Maximize2, MessageCircle, Minimize2, PanelRight, Pencil, SlidersHorizontal, SquarePen, X } from "lucide-react"
import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import type { AgentPanel, AgentViewMode } from "./AgentChatTypes"

interface AgentChatHeaderProps {
  activePanel: AgentPanel
  viewMode: AgentViewMode
  isDesktop: boolean
  canClear: boolean
  sessionId: string | null
  sessionTitle: string | null
  onRenameTitle: (title: string) => Promise<void>
  onBack: () => void
  onShowHistory: () => void
  onShowPreferences: () => void
  onToggleViewMode: () => void
  onClearChat: () => void
  onClose: () => void
}

function panelTitle(panel: AgentPanel) {
  if (panel === "history") return "History"
  if (panel === "preferences") return "Preferences"
  return "Stan"
}

function panelSubtitle(panel: AgentPanel, viewMode: AgentViewMode) {
  if (panel === "history") return "Conversation continuity"
  if (panel === "preferences") return "Response behavior"
  return viewMode === "console" ? "Market intelligence console" : "Market intelligence"
}

export function AgentChatHeader({
  activePanel,
  viewMode,
  isDesktop,
  canClear,
  sessionId,
  sessionTitle,
  onRenameTitle,
  onBack,
  onShowHistory,
  onShowPreferences,
  onToggleViewMode,
  onClearChat,
  onClose,
}: AgentChatHeaderProps) {
  const inSubPanel = activePanel !== "chat"
  const title = activePanel === "chat" ? sessionTitle?.trim() || "Stan" : panelTitle(activePanel)
  const canRename = activePanel === "chat" && Boolean(sessionId)
  const [editing, setEditing] = useState(false)
  const [draftTitle, setDraftTitle] = useState(title)
  const [renameError, setRenameError] = useState<string | null>(null)
  const [savingTitle, setSavingTitle] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!editing) setDraftTitle(title)
  }, [editing, title])

  useEffect(() => {
    if (editing) window.setTimeout(() => inputRef.current?.select(), 20)
  }, [editing])

  function cancelRename() {
    setEditing(false)
    setDraftTitle(title)
    setRenameError(null)
  }

  async function saveRename() {
    if (!canRename || savingTitle) return
    const next = draftTitle.replace(/\s+/g, " ").trim()
    if (!next) {
      setRenameError("Title cannot be empty.")
      return
    }
    if (next.length > 80) {
      setRenameError("Title must be 80 characters or fewer.")
      return
    }
    if (next === title) {
      cancelRename()
      return
    }
    setSavingTitle(true)
    setRenameError(null)
    try {
      await onRenameTitle(next)
      setEditing(false)
    } catch (err) {
      setRenameError(err instanceof Error ? err.message : "Failed to rename conversation.")
    } finally {
      setSavingTitle(false)
    }
  }

  function handleTitleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault()
      void saveRename()
      return
    }
    if (event.key === "Escape") {
      event.preventDefault()
      cancelRename()
    }
  }

  return (
    <header className="flex shrink-0 items-center justify-between gap-3 border-b border-app bg-card px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        {inSubPanel ? (
          <button
            type="button"
            onClick={onBack}
            className="theme-icon-button h-10 w-10 shrink-0"
            aria-label="Back to chat"
            title="Back to chat"
          >
            <ArrowLeft size={16} aria-hidden="true" />
          </button>
        ) : (
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-app bg-card-muted text-link">
            <MessageCircle size={17} aria-hidden="true" />
          </span>
        )}
        <div className="min-w-0">
          {editing ? (
            <div className="flex min-w-0 items-center gap-1.5">
              <input
                ref={inputRef}
                value={draftTitle}
                onChange={event => {
                  setDraftTitle(event.currentTarget.value)
                  setRenameError(null)
                }}
                onKeyDown={handleTitleKeyDown}
                disabled={savingTitle}
                aria-label="Conversation title"
                className="h-8 min-w-0 rounded-md border border-app bg-app px-2 text-sm font-semibold text-app outline-none focus:border-[hsl(var(--accent))]"
              />
              <button
                type="button"
                onClick={() => void saveRename()}
                disabled={savingTitle}
                className="theme-icon-button h-8 w-8 shrink-0 disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Save conversation title"
                title="Save"
              >
                <Check size={14} aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={cancelRename}
                disabled={savingTitle}
                className="theme-icon-button h-8 w-8 shrink-0 disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Cancel conversation title edit"
                title="Cancel"
              >
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          ) : (
            <div className="group/title flex min-w-0 items-center gap-1.5">
              <h2 className="truncate text-sm font-semibold text-app">{title}</h2>
              {canRename && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(true)
                    setRenameError(null)
                  }}
                  className="theme-icon-button h-7 w-7 shrink-0 opacity-70 transition-opacity hover:opacity-100 sm:opacity-0 sm:group-hover/title:opacity-100"
                  aria-label={`Rename ${title}`}
                  title="Rename"
                >
                  <Pencil size={12} aria-hidden="true" />
                </button>
              )}
            </div>
          )}
          <p className="truncate text-xs text-subtle">{panelSubtitle(activePanel, viewMode)}</p>
          {renameError && <p className="mt-1 truncate text-[11px] text-negative">{renameError}</p>}
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        {activePanel === "chat" && (
          <>
            <button
              type="button"
              onClick={onShowPreferences}
              className="theme-icon-button h-10 w-10"
              aria-label="Open response preferences"
              title="Response preferences"
            >
              <SlidersHorizontal size={15} aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={onShowHistory}
              className="theme-icon-button h-10 w-10"
              aria-label="Open conversation history"
              title="Conversation history"
            >
              <History size={15} aria-hidden="true" />
            </button>
            {isDesktop && (
              <button
                type="button"
                onClick={onToggleViewMode}
                className="theme-icon-button h-10 w-10"
                aria-label={viewMode === "console" ? "Use compact chat" : "Use console layout"}
                title={viewMode === "console" ? "Use compact chat" : "Use console layout"}
              >
                {viewMode === "console" ? <Minimize2 size={15} aria-hidden="true" /> : <Maximize2 size={15} aria-hidden="true" />}
              </button>
            )}
            <button
              type="button"
              onClick={onClearChat}
              disabled={!canClear}
              className="theme-icon-button h-10 w-10 disabled:cursor-not-allowed disabled:opacity-35"
              aria-label="Start new chat"
              title="New chat"
            >
              <SquarePen size={15} aria-hidden="true" />
            </button>
          </>
        )}
        {activePanel !== "chat" && isDesktop && (
          <button
            type="button"
            onClick={onToggleViewMode}
            className="theme-icon-button h-10 w-10"
            aria-label={viewMode === "console" ? "Use compact chat" : "Use console layout"}
            title={viewMode === "console" ? "Use compact chat" : "Use console layout"}
          >
            <PanelRight size={15} aria-hidden="true" />
          </button>
        )}
        <button
          type="button"
          onClick={onClose}
          className="theme-icon-button h-10 w-10"
          aria-label="Close Stan"
          title="Close"
        >
          <X size={17} aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}
