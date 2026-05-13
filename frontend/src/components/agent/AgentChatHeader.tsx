import { ArrowLeft, History, Maximize2, MessageCircle, Minimize2, PanelRight, SlidersHorizontal, SquarePen, X } from "lucide-react"
import type { AgentPanel, AgentViewMode } from "./AgentChatTypes"

interface AgentChatHeaderProps {
  activePanel: AgentPanel
  viewMode: AgentViewMode
  isDesktop: boolean
  canClear: boolean
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
  onBack,
  onShowHistory,
  onShowPreferences,
  onToggleViewMode,
  onClearChat,
  onClose,
}: AgentChatHeaderProps) {
  const inSubPanel = activePanel !== "chat"
  const title = panelTitle(activePanel)

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
          <h2 className="truncate text-sm font-semibold text-app">{title}</h2>
          <p className="truncate text-xs text-subtle">{panelSubtitle(activePanel, viewMode)}</p>
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
