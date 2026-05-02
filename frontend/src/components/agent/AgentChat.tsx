import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import { X, Trash2, Send, Square, MessageCircle, Maximize2, Minimize2, History, ArrowLeft, Zap, ChevronDown, SquarePen, SlidersHorizontal } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAgentChat, fetchSessionHistory, deleteSession, type AgentPreferenceLevel, type AgentResponsePreferences, type SessionSummary } from "@/hooks/useAgentChat"
import { AgentMessage } from "./AgentMessage"
import type { ScreenContext } from "@/contexts/ScreenContext"

// ---------------------------------------------------------------------------
// Quick prompts shown when chat is empty
// ---------------------------------------------------------------------------

const QUICK_PROMPTS = [
  "What's the current market risk environment?",
  "Summarize my portfolio's performance",
  "How is global liquidity affecting risk assets?",
  "What does positioning data say about crowded trades?",
  "Given current positioning data, macro liquidity, and my portfolio's sector tilts, what are my top 3 risks?",
]

const PREFERENCES_STORAGE_KEY = "agent-response-preferences"

const DEFAULT_RESPONSE_PREFERENCES: AgentResponsePreferences = {
  personality: "pragmatic",
  warmth: "less",
  enthusiasm: "less",
  headers_lists: "less",
  emoji: "less",
  fast_answers: true,
  custom_instructions: "",
}

const LEVEL_OPTIONS: { value: AgentPreferenceLevel; label: string }[] = [
  { value: "less", label: "Less" },
  { value: "balanced", label: "Balanced" },
  { value: "more", label: "More" },
]

function loadResponsePreferences(): AgentResponsePreferences {
  try {
    const raw = localStorage.getItem(PREFERENCES_STORAGE_KEY)
    if (!raw) return DEFAULT_RESPONSE_PREFERENCES
    const parsed = JSON.parse(raw) as Partial<AgentResponsePreferences>
    return {
      ...DEFAULT_RESPONSE_PREFERENCES,
      ...parsed,
    }
  } catch {
    return DEFAULT_RESPONSE_PREFERENCES
  }
}

function normalizePreferences(prefs: AgentResponsePreferences): AgentResponsePreferences {
  return {
    ...prefs,
    custom_instructions: prefs.custom_instructions?.trim() || "",
  }
}

// ---------------------------------------------------------------------------
// Workflow definitions (mirrors backend AVAILABLE_WORKFLOWS)
// ---------------------------------------------------------------------------

interface WorkflowDef {
  name: string
  label: string
  description: string
  requiresTicker: boolean
}

const WORKFLOWS: WorkflowDef[] = [
  { name: "morning_brief", label: "Morning Brief", description: "Macro + portfolio + signals overview", requiresTicker: false },
  { name: "thesis_review", label: "Thesis Review", description: "Deep review of a position's thesis", requiresTicker: true },
  { name: "pre_earnings", label: "Pre-Earnings Prep", description: "Earnings briefing with risk scenarios", requiresTicker: true },
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface AgentChatProps {
  open: boolean
  onClose: () => void
  screenContext?: ScreenContext
}

export function AgentChat({ open, onClose, screenContext }: AgentChatProps) {
  const { messages, isStreaming, error, sendMessage, stopStreaming, clearChat, loadSession } = useAgentChat()
  const [input, setInput] = useState("")
  const [isWide, setIsWide] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [showPreferences, setShowPreferences] = useState(false)
  const [history, setHistory] = useState<SessionSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [workflowTicker, setWorkflowTicker] = useState("")
  const [showWorkflows, setShowWorkflows] = useState(false)
  const [preferences, setPreferences] = useState<AgentResponsePreferences>(loadResponsePreferences)
  const [draftPreferences, setDraftPreferences] = useState<AgentResponsePreferences>(preferences)
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
    sendMessage(trimmed, screenContext, preferences)
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleQuickPrompt(prompt: string) {
    sendMessage(prompt, screenContext, preferences)
  }

  function handleWorkflow(wf: WorkflowDef) {
    const ticker = workflowTicker.trim().toUpperCase()
    if (wf.requiresTicker && !ticker) return
    const cmd = wf.requiresTicker
      ? `/workflow:${wf.name}:${ticker}`
      : `/workflow:${wf.name}`
    sendMessage(cmd, screenContext, preferences)
    setWorkflowTicker("")
    setShowWorkflows(false)
  }

  async function handleShowHistory() {
    setShowHistory(true)
    setShowPreferences(false)
    setHistoryLoading(true)
    const sessions = await fetchSessionHistory(30)
    setHistory(sessions)
    setHistoryLoading(false)
  }

  function handleShowPreferences() {
    setDraftPreferences(preferences)
    setShowHistory(false)
    setShowPreferences(true)
  }

  function handleSavePreferences() {
    const next = normalizePreferences(draftPreferences)
    setPreferences(next)
    localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(next))
    setShowPreferences(false)
  }

  type LevelPreferenceKey = "warmth" | "enthusiasm" | "headers_lists" | "emoji"

  function renderLevelSelect(label: string, key: LevelPreferenceKey) {
    return (
      <label className="flex items-center justify-between gap-3 py-2">
        <span className="text-sm text-app">{label}</span>
        <select
          value={draftPreferences[key]}
          onChange={e => setDraftPreferences(prev => ({ ...prev, [key]: e.target.value as AgentPreferenceLevel }))}
          className="w-28 rounded-md border border-app bg-app px-2 py-1.5 text-sm text-app focus:outline-none focus:ring-2 focus:ring-blue-500/40"
        >
          {LEVEL_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </label>
    )
  }

  async function handleLoadSession(sessionId: string) {
    await loadSession(sessionId)
    setShowHistory(false)
  }

  async function handleDeleteSession(sessionId: string) {
    const ok = await deleteSession(sessionId)
    if (ok) {
      setHistory(prev => prev.filter(s => s.session_id !== sessionId))
    }
  }

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return ""
    try {
      const d = new Date(dateStr)
      const now = new Date()
      const diff = now.getTime() - d.getTime()
      const days = Math.floor(diff / (1000 * 60 * 60 * 24))
      if (days === 0) return "Today"
      if (days === 1) return "Yesterday"
      if (days < 7) return `${days}d ago`
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
    } catch {
      return ""
    }
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
          "fixed top-0 right-0 z-50 h-full w-full bg-app border-l border-app",
          "flex flex-col shadow-2xl",
          "transition-[width,transform] duration-300 ease-in-out",
          isWide
            ? "sm:w-[min(680px,100vw)] md:w-[calc(100vw-14rem)]"
            : "sm:w-[500px]",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-app bg-card">
          <div className="flex items-center gap-2">
            {showHistory ? (
              <button
                onClick={() => { setShowHistory(false); setShowPreferences(false) }}
                className="p-1 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
                title="Back to chat"
              >
                <ArrowLeft size={16} />
              </button>
            ) : showPreferences ? (
              <button
                onClick={() => setShowPreferences(false)}
                className="p-1 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
                title="Back to chat"
              >
                <ArrowLeft size={16} />
              </button>
            ) : (
              <MessageCircle size={16} className="text-blue-500" />
            )}
            <span className="text-sm font-semibold text-app">
              {showHistory ? "History" : showPreferences ? "Preferences" : "Stan"}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {!showHistory && !showPreferences && (
              <button
                onClick={handleShowPreferences}
                className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
                title="Response preferences"
              >
                <SlidersHorizontal size={14} />
              </button>
            )}
            {!showHistory && !showPreferences && (
              <button
                onClick={handleShowHistory}
                className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
                title="Conversation history"
              >
                <History size={14} />
              </button>
            )}
            <button
              onClick={() => setIsWide(v => !v)}
              className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors"
              title={isWide ? "Restore default width" : "Widen chat"}
              aria-label={isWide ? "Restore default width" : "Widen chat"}
            >
              {isWide ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
            {!showHistory && !showPreferences && (
              <button
                onClick={clearChat}
                disabled={messages.length === 0}
                className="p-1.5 rounded-lg text-muted hover:text-app hover:bg-muted-surface transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="New chat"
              >
                <SquarePen size={14} />
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

        {showHistory ? (
          /* ---- History panel ---- */
          <div className="flex-1 overflow-y-auto px-4 py-4">
            {historyLoading ? (
              <div className="flex items-center justify-center h-32">
                <span className="text-xs text-muted">Loading history...</span>
              </div>
            ) : history.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 text-center">
                <History size={24} className="text-muted mb-2" />
                <p className="text-xs text-muted">No past conversations yet.</p>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {history.map(session => (
                  <div
                    key={session.session_id}
                    className="group flex items-start gap-3 rounded-lg border border-app bg-card p-3 hover:bg-muted-surface transition-colors cursor-pointer"
                    onClick={() => handleLoadSession(session.session_id)}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-app">
                          {formatDate(session.ended_at ?? session.started_at)}
                        </span>
                        <span className="text-[10px] text-muted">
                          {session.message_count} msgs
                        </span>
                        {session.key_tickers && session.key_tickers.length > 0 && (
                          <span className="text-[10px] text-blue-500 font-mono">
                            {session.key_tickers.slice(0, 4).join(", ")}
                          </span>
                        )}
                      </div>
                      {session.summary ? (
                        <p className="text-xs text-muted line-clamp-2">
                          {session.summary}
                        </p>
                      ) : session.key_topics && session.key_topics.length > 0 ? (
                        <p className="text-xs text-muted line-clamp-2">
                          {session.key_topics.join(" · ")}
                        </p>
                      ) : null}
                    </div>
                    <button
                      onClick={e => {
                        e.stopPropagation()
                        handleDeleteSession(session.session_id)
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-muted hover:text-red-500 transition-all"
                      title="Delete"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : showPreferences ? (
          <div className="flex-1 overflow-y-auto px-4 py-4">
            <div className="space-y-5">
              <section>
                <label className="flex items-center justify-between gap-3">
                  <span>
                    <span className="block text-sm font-medium text-app">Personality</span>
                    <span className="block text-xs text-muted">Default tone for Stan responses</span>
                  </span>
                  <select
                    value={draftPreferences.personality}
                    onChange={e => setDraftPreferences(prev => ({
                      ...prev,
                      personality: e.target.value as AgentResponsePreferences["personality"],
                    }))}
                    className="w-32 rounded-md border border-app bg-app px-2 py-1.5 text-sm text-app focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                  >
                    <option value="pragmatic">Pragmatic</option>
                    <option value="friendly">Friendly</option>
                  </select>
                </label>
              </section>

              <section className="divide-y divide-app">
                {renderLevelSelect("Warm", "warmth")}
                {renderLevelSelect("Enthusiastic", "enthusiasm")}
                {renderLevelSelect("Headers & Lists", "headers_lists")}
                {renderLevelSelect("Emoji", "emoji")}
              </section>

              <section>
                <label className="flex items-center justify-between gap-3">
                  <span>
                    <span className="block text-sm font-medium text-app">Fast Answers</span>
                    <span className="block text-xs text-muted">Prefer direct answers for simple questions</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={draftPreferences.fast_answers}
                    onChange={e => setDraftPreferences(prev => ({ ...prev, fast_answers: e.target.checked }))}
                    className="h-4 w-4 accent-blue-600"
                  />
                </label>
              </section>

              <section>
                <label className="block text-sm font-medium text-app mb-2">Custom Instructions</label>
                <textarea
                  value={draftPreferences.custom_instructions ?? ""}
                  onChange={e => setDraftPreferences(prev => ({ ...prev, custom_instructions: e.target.value }))}
                  placeholder="End responses after answering. Do not ask follow-up questions."
                  rows={6}
                  maxLength={2000}
                  className="w-full resize-none rounded-lg border border-app bg-app px-3 py-2 text-sm text-app placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                />
              </section>

              <div className="flex justify-end">
                <button
                  onClick={handleSavePreferences}
                  className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto px-4 py-4">
              {messages.length === 0 && !isStreaming ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <MessageCircle size={32} className="text-muted mb-3" />
                  <p className="text-sm font-medium text-app mb-1">Stan</p>
                  <p className="text-xs text-muted mb-6 max-w-[280px]">
                    Ask questions about your portfolio, market conditions, or macro environment.
                    Stan can fetch live data from all your dashboards.
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

            {/* Workflow launcher + Input area */}
            <div className="border-t border-app px-4 py-3 bg-card">
              {/* Workflow buttons */}
              {!isStreaming && (
                <div className="mb-2">
                  <button
                    onClick={() => setShowWorkflows(v => !v)}
                    className="flex items-center gap-1.5 text-[11px] text-muted hover:text-app transition-colors mb-1.5"
                  >
                    <Zap size={12} />
                    <span>Workflows</span>
                    <ChevronDown size={10} className={cn("transition-transform", showWorkflows && "rotate-180")} />
                  </button>
                  {showWorkflows && (
                    <div className="flex flex-wrap gap-1.5 mb-2">
                      {WORKFLOWS.map(wf => (
                        <div key={wf.name} className="flex items-center gap-1">
                          {wf.requiresTicker && (
                            <input
                              type="text"
                              value={workflowTicker}
                              onChange={e => setWorkflowTicker(e.target.value.toUpperCase())}
                              placeholder="TICKER"
                              className="w-[60px] px-1.5 py-1 text-[11px] rounded border border-app bg-app text-app placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-blue-500/40 font-mono"
                              onKeyDown={e => {
                                if (e.key === "Enter") {
                                  e.preventDefault()
                                  handleWorkflow(wf)
                                }
                              }}
                            />
                          )}
                          <button
                            onClick={() => handleWorkflow(wf)}
                            disabled={wf.requiresTicker && !workflowTicker.trim()}
                            className="px-2 py-1 text-[11px] rounded-md border border-app bg-card hover:bg-muted-surface text-muted hover:text-app transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                            title={wf.description}
                          >
                            {wf.label}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
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
          </>
        )}
      </div>
    </>
  )
}
