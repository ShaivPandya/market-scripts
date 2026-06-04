import { ChevronDown, Send, Square, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import { resizeChatTextarea } from "./agentChatTextarea"
import type { KeyboardEvent, ReactNode, RefObject } from "react"

interface AgentChatComposerProps {
  input: string
  onInputChange: (value: string) => void
  onInputSelectionChange: (start: number, end: number) => void
  onSend: () => void
  onStop: () => void
  /** True while the current turn is in progress (stream, job poll, or active tools). */
  isBusy: boolean
  queuedCount?: number
  textareaRef: RefObject<HTMLTextAreaElement | null>
  compactWorkflowSlot?: ReactNode
  workflowsOpen?: boolean
  onToggleWorkflows?: () => void
}

export function AgentChatComposer({
  input,
  onInputChange,
  onInputSelectionChange,
  onSend,
  onStop,
  isBusy,
  queuedCount = 0,
  textareaRef,
  compactWorkflowSlot,
  workflowsOpen,
  onToggleWorkflows,
}: AgentChatComposerProps) {
  const hasInput = Boolean(input.trim())
  const showStop = isBusy && !hasInput

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      onSend()
    }
  }

  function rememberSelection(element: HTMLTextAreaElement) {
    onInputSelectionChange(element.selectionStart, element.selectionEnd)
  }

  const sendLabel = isBusy ? "Queue follow-up" : "Send message"
  const sendTitle = isBusy ? "Queue follow-up" : "Send message"

  return (
    <div className="safe-bottom shrink-0 border-t border-app bg-card px-4 py-3">
      {compactWorkflowSlot && onToggleWorkflows && (
        <div className="mb-3">
          <button
            type="button"
            onClick={onToggleWorkflows}
            className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-muted transition-colors hover:text-app"
            aria-expanded={workflowsOpen}
          >
            <Zap size={13} aria-hidden="true" />
            <span>Workflows</span>
            <ChevronDown size={12} className={cn("transition-transform", workflowsOpen && "rotate-180")} aria-hidden="true" />
          </button>
          {workflowsOpen && compactWorkflowSlot}
        </div>
      )}

      {isBusy && (
        <div className="mb-2 flex items-center justify-between gap-3 rounded-lg border border-app bg-card-muted px-3 py-2 text-xs text-muted" aria-live="polite">
          <span>
            Generating response
            {queuedCount > 0 ? ` · ${queuedCount} queued` : " · Enter queues follow-up"}
          </span>
          <span className="h-2 w-2 rounded-full bg-[hsl(var(--accent))] animate-pulse" aria-hidden="true" />
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={event => {
            onInputChange(event.currentTarget.value)
            rememberSelection(event.currentTarget)
          }}
          onKeyDown={handleKeyDown}
          onKeyUp={event => rememberSelection(event.currentTarget)}
          onMouseUp={event => rememberSelection(event.currentTarget)}
          onSelect={event => rememberSelection(event.currentTarget)}
          onBlur={event => rememberSelection(event.currentTarget)}
          onInput={event => resizeChatTextarea(event.currentTarget)}
          placeholder="Ask about markets, portfolio, macro..."
          aria-label="Message Stan"
          rows={1}
          className="theme-input min-h-[44px] min-w-0 max-h-[120px] flex-1 resize-none overflow-x-hidden rounded-xl text-sm leading-5"
          style={{ height: "44px", overflowX: "hidden", overflowY: "hidden" }}
        />
        {showStop ? (
          <button
            type="button"
            onClick={onStop}
            className="theme-button-destructive flex h-11 w-11 flex-none items-center justify-center rounded-full"
            aria-label="Stop generating"
            title="Stop generating"
          >
            <Square size={14} aria-hidden="true" />
          </button>
        ) : (
          <button
            type="button"
            onClick={onSend}
            disabled={!hasInput}
            className="theme-button-primary flex h-11 w-11 flex-none items-center justify-center rounded-full text-[hsl(var(--accent-foreground))] disabled:cursor-not-allowed disabled:opacity-40"
            aria-label={sendLabel}
            title={sendTitle}
          >
            <Send size={14} aria-hidden="true" />
          </button>
        )}
      </div>
    </div>
  )
}
