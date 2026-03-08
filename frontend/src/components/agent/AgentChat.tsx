import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import { X, Trash2, Send, Square, MessageCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAgentChat } from "@/hooks/useAgentChat"
import { AgentMessage } from "./AgentMessage"

// ---------------------------------------------------------------------------
// Quick prompts shown when chat is empty
// ---------------------------------------------------------------------------

const QUICK_PROMPTS = [
  "What's the current market risk environment?",
  "Summarize my portfolio's performance",
  "How is global liquidity affecting risk assets?",
  "What does positioning data say about crowded trades?",
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface AgentChatProps {
  open: boolean
  onClose: () => void
}

export function AgentChat({ open, onClose }: AgentChatProps) {
  const { messages, isStreaming, error, sendMessage, stopStreaming, clearChat } = useAgentChat()
  const [input, setInput] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Focus textarea when drawer opens
  useEffect(() => {
    if (open) {
      setTimeout(() => textareaRef.current?.focus(), 300)
    }
  }, [open])

  function handleSend() {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setInput("")
    sendMessage(trimmed)
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleQuickPrompt(prompt: string) {
    sendMessage(prompt)
  }

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/30 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Drawer panel */}
      <div
        className={cn(
          "fixed top-0 right-0 z-50 h-full w-full sm:w-[420px] bg-app border-l border-app",
          "flex flex-col shadow-2xl",
          "transition-transform duration-300 ease-in-out",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-app bg-card">
          <div className="flex items-center gap-2">
            <MessageCircle size={16} className="text-blue-500" />
            <span className="text-sm font-semibold text-app">AI Agent</span>
          </div>
          <div className="flex items-center gap-1">
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
                title="Clear chat"
              >
                <Trash2 size={14} />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          {messages.length === 0 && !isStreaming ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <MessageCircle size={32} className="text-muted mb-3" />
              <p className="text-sm font-medium text-app mb-1">AI Agent</p>
              <p className="text-xs text-muted mb-6 max-w-[280px]">
                Ask questions about your portfolio, market conditions, or macro environment.
                The agent can fetch live data from all your dashboards.
              </p>
              <div className="flex flex-col gap-2 w-full max-w-[300px]">
                {QUICK_PROMPTS.map(prompt => (
                  <button
                    key={prompt}
                    onClick={() => handleQuickPrompt(prompt)}
                    className="text-left text-xs px-3 py-2 rounded-lg border border-app bg-card hover:bg-muted-surface transition-colors text-muted hover:text-app"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map(msg => (
                <AgentMessage key={msg.id} message={msg} />
              ))}
              {error && (
                <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-400">
                  {error}
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input area */}
        <div className="border-t border-app px-4 py-3 bg-card">
          <div className="flex items-end gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about markets, portfolio, macro..."
              rows={1}
              className={cn(
                "flex-1 resize-none rounded-lg border border-app bg-app px-3 py-2 text-sm text-app",
                "placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500",
                "max-h-[120px]",
              )}
              style={{ minHeight: "38px" }}
              onInput={e => {
                const el = e.currentTarget
                el.style.height = "auto"
                el.style.height = Math.min(el.scrollHeight, 120) + "px"
              }}
              disabled={isStreaming}
            />
            {isStreaming ? (
              <button
                onClick={stopStreaming}
                className="flex-none flex items-center justify-center h-[38px] w-[38px] rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors"
                title="Stop generating"
              >
                <Square size={14} />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="flex-none flex items-center justify-center h-[38px] w-[38px] rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                title="Send message"
              >
                <Send size={14} />
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
