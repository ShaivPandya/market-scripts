import * as RadixDialog from "@radix-ui/react-dialog"
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { cn } from "@/lib/utils"
import {
  fetchAgentResponsePreferences,
  fetchAgentWorkflows,
  updateAgentResponsePreferences,
  type AgentResponsePreferences,
  type AgentWorkflow,
} from "@/lib/api"
import { deleteSession, fetchSessionHistory, renameSessionTitle, useAgentChat, type SessionSummary } from "@/hooks/useAgentChat"
import type { ScreenContext } from "@/contexts/ScreenContext"
import type { StanOpenDetail } from "@/lib/stanLauncher"
import { AgentChatComposer } from "./AgentChatComposer"
import { AgentChatQueuePanel } from "./AgentChatQueuePanel"
import { AgentChatHeader } from "./AgentChatHeader"
import { AgentContextPane } from "./AgentContextPane"
import { AgentHistoryPanel } from "./AgentHistoryPanel"
import { AgentMessageStream } from "./AgentMessageStream"
import { AgentPreferencesPanel } from "./AgentPreferencesPanel"
import { AgentWorkflowLauncher } from "./AgentWorkflowLauncher"
import { QUICK_PROMPT_GROUPS } from "./agentChatPrompts"
import { resizeChatTextarea } from "./agentChatTextarea"
import { useMediaQuery } from "./useMediaQuery"
import { useDecisionTrace } from "@/contexts/DecisionTraceContext"
import type { AgentPanel, AgentViewMode } from "./AgentChatTypes"

const PREFERENCES_STORAGE_KEY = "agent-response-preferences"
const RESPONSE_PREFERENCES_QUERY_KEY = ["agent-response-preferences"] as const

const DEFAULT_RESPONSE_PREFERENCES: AgentResponsePreferences = {
  personality: "pragmatic",
  warmth: "less",
  enthusiasm: "less",
  headers_lists: "less",
  emoji: "less",
  fast_answers: true,
  thinking_enabled: false,
  custom_instructions: "",
}

interface AgentChatProps {
  open: boolean
  onClose: () => void
  screenContext?: ScreenContext
  pendingCommand?: StanOpenDetail | null
  onPendingCommandConsumed?: () => void
}

interface WorkflowInvalidationTarget {
  workflowName: string
  ticker: string | null
}

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

function workflowTargetFromCommand(value: string): WorkflowInvalidationTarget | null {
  const match = value.trim().match(/^\/workflow:([A-Za-z0-9_]+)(?::([A-Za-z0-9._-]+))?(?:\s|$)/)
  if (!match) return null
  return {
    workflowName: match[1],
    ticker: match[2] ? match[2].toUpperCase() : null,
  }
}

function invalidateWorkflowRunQueries(
  queryClient: ReturnType<typeof useQueryClient>,
  target: WorkflowInvalidationTarget,
) {
  void queryClient.invalidateQueries({ queryKey: ["workspace"] })
  void queryClient.invalidateQueries({ queryKey: ["workflow-runs"] })
  if (target.ticker) {
    void queryClient.invalidateQueries({ queryKey: ["dossier", target.ticker] })
  }
}

export function AgentChat({
  open,
  onClose,
  screenContext,
  pendingCommand,
  onPendingCommandConsumed,
}: AgentChatProps) {
  const {
    messages,
    isStreaming,
    error,
    sessionId,
    sessionTitle,
    queuedMessages,
    sendMessage,
    stopStreaming,
    clearChat,
    loadSession,
    applySessionTitle,
    removeQueuedMessage,
    editQueuedMessage,
    clearQueuedMessages,
    sendQueuedMessageNow,
  } = useAgentChat()
  const { openDecisionTrace } = useDecisionTrace()
  const queryClient = useQueryClient()
  const isDesktop = useMediaQuery("(min-width: 1024px)")
  const [viewModeOverride, setViewModeOverride] = useState<AgentViewMode | null>(null)
  const viewMode: AgentViewMode = isDesktop ? viewModeOverride ?? "console" : "compact"
  const [activePanel, setActivePanel] = useState<AgentPanel>("chat")
  const [input, setInput] = useState("")
  const [history, setHistory] = useState<SessionSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [workflowTicker, setWorkflowTicker] = useState("")
  const [showCompactWorkflows, setShowCompactWorkflows] = useState(false)
  const [cachedPreferences] = useState<AgentResponsePreferences>(loadResponsePreferences)
  const [draftPreferences, setDraftPreferences] = useState<AgentResponsePreferences>(cachedPreferences)
  const [preferenceSaveError, setPreferenceSaveError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesScrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const inputValueRef = useRef(input)
  const inputSelectionRef = useRef<{ start: number; end: number } | null>(null)
  const wasOpenRef = useRef(open)
  const wasStreamingRef = useRef(isStreaming)
  const pendingWorkflowInvalidationRef = useRef<WorkflowInvalidationTarget | null>(null)
  const lastConsumedCommandRef = useRef<string | null>(null)

  const workflowsQuery = useQuery({
    queryKey: ["agent-workflows"],
    queryFn: fetchAgentWorkflows,
    enabled: open,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const preferencesQuery = useQuery({
    queryKey: RESPONSE_PREFERENCES_QUERY_KEY,
    queryFn: fetchAgentResponsePreferences,
    enabled: open,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const preferencesMutation = useMutation({
    mutationFn: updateAgentResponsePreferences,
    onSuccess: savedPreferences => {
      const next = normalizePreferences(savedPreferences)
      queryClient.setQueryData(RESPONSE_PREFERENCES_QUERY_KEY, next)
      setDraftPreferences(next)
      localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(next))
      setPreferenceSaveError(null)
      setActivePanel("chat")
    },
    onError: err => {
      setPreferenceSaveError(err instanceof Error ? err.message : "Failed to save preferences")
    },
  })

  const workflows = workflowsQuery.data ?? []
  const activePreferences = preferencesQuery.data
    ? normalizePreferences(preferencesQuery.data)
    : cachedPreferences

  const scrollMessagesToBottom = useCallback((behavior: ScrollBehavior) => {
    const container = messagesScrollRef.current
    if (container) {
      container.scrollTo({ top: container.scrollHeight, behavior })
      return
    }
    messagesEndRef.current?.scrollIntoView({ behavior, block: "end" })
  }, [])

  const focusComposerTextarea = useCallback(() => {
    const textarea = textareaRef.current
    if (!textarea) return

    const inputValue = inputValueRef.current
    const selection = inputSelectionRef.current ?? { start: inputValue.length, end: inputValue.length }
    textarea.focus()
    const start = Math.min(selection.start, inputValue.length)
    const end = Math.min(selection.end, inputValue.length)
    textarea.setSelectionRange(start, end)
  }, [])

  useLayoutEffect(() => {
    const wasOpen = wasOpenRef.current
    wasOpenRef.current = open

    if (!open || activePanel !== "chat") return

    const behavior: ScrollBehavior = isStreaming || !wasOpen ? "auto" : "smooth"
    scrollMessagesToBottom(behavior)

    const animationFrame = window.requestAnimationFrame(() => {
      scrollMessagesToBottom(behavior)
    })
    return () => window.cancelAnimationFrame(animationFrame)
  }, [messages, isStreaming, activePanel, open, scrollMessagesToBottom])

  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && pendingWorkflowInvalidationRef.current) {
      invalidateWorkflowRunQueries(queryClient, pendingWorkflowInvalidationRef.current)
      pendingWorkflowInvalidationRef.current = null
    }
    wasStreamingRef.current = isStreaming
  }, [isStreaming, queryClient])

  useEffect(() => {
    if (open && activePanel === "chat") {
      window.setTimeout(focusComposerTextarea, 160)
    }
  }, [open, activePanel, focusComposerTextarea])

  useEffect(() => {
    if (textareaRef.current) resizeChatTextarea(textareaRef.current)
  }, [input, open, viewMode])

  useEffect(() => {
    inputValueRef.current = input
  }, [input])

  useEffect(() => {
    if (preferencesQuery.data) {
      localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(normalizePreferences(preferencesQuery.data)))
    }
  }, [preferencesQuery.data])

  useEffect(() => {
    if (!pendingCommand) {
      lastConsumedCommandRef.current = null
    }
  }, [pendingCommand])

  useEffect(() => {
    if (!open || !pendingCommand?.command?.trim()) return
    const command = pendingCommand.command.trim()
    if (lastConsumedCommandRef.current === command) return
    lastConsumedCommandRef.current = command
    pendingWorkflowInvalidationRef.current = workflowTargetFromCommand(command)
    window.setTimeout(() => setActivePanel("chat"), 0)
    sendMessage(command, screenContext, activePreferences, {
      durable: pendingCommand.durable ?? true,
      mode: isStreaming ? "enqueue" : undefined,
    })
    onPendingCommandConsumed?.()
  }, [
    open,
    pendingCommand,
    isStreaming,
    screenContext,
    activePreferences,
    sendMessage,
    onPendingCommandConsumed,
  ])

  function updateDraftPreferences(updater: (prev: AgentResponsePreferences) => AgentResponsePreferences) {
    setDraftPreferences(updater)
    setPreferenceSaveError(null)
  }

  function handleSend() {
    const trimmed = input.trim()
    if (!trimmed) return
    pendingWorkflowInvalidationRef.current = workflowTargetFromCommand(trimmed)
    inputValueRef.current = ""
    inputSelectionRef.current = { start: 0, end: 0 }
    setInput("")
    setActivePanel("chat")
    sendMessage(trimmed, screenContext, activePreferences)
  }

  function handleQuickPrompt(prompt: string) {
    pendingWorkflowInvalidationRef.current = null
    setActivePanel("chat")
    sendMessage(prompt, screenContext, activePreferences, isStreaming ? { mode: "enqueue" } : undefined)
  }

  function handleWorkflow(workflow: AgentWorkflow) {
    const ticker = workflowTicker.trim().toUpperCase()
    if (workflow.requiresTicker && !ticker) return
    const command = workflow.requiresTicker
      ? `/workflow:${workflow.name}:${ticker}`
      : `/workflow:${workflow.name}`
    pendingWorkflowInvalidationRef.current = { workflowName: workflow.name, ticker: ticker || null }
    setActivePanel("chat")
    sendMessage(command, screenContext, activePreferences, {
      durable: true,
      mode: isStreaming ? "enqueue" : undefined,
    })
    setWorkflowTicker("")
    setShowCompactWorkflows(false)
  }

  async function handleShowHistory() {
    setActivePanel("history")
    setHistoryLoading(true)
    const sessions = await fetchSessionHistory(30)
    setHistory(sessions)
    setHistoryLoading(false)
  }

  function handleShowPreferences() {
    setDraftPreferences(activePreferences)
    setPreferenceSaveError(null)
    setActivePanel("preferences")
  }

  function handleSavePreferences() {
    if (preferencesMutation.isPending) return
    preferencesMutation.mutate(normalizePreferences(draftPreferences))
  }

  async function handleLoadSession(sessionId: string) {
    await loadSession(sessionId)
    setActivePanel("chat")
  }

  async function handleDeleteSession(sessionId: string) {
    const ok = await deleteSession(sessionId)
    if (ok) {
      setHistory(prev => prev.filter(session => session.session_id !== sessionId))
    }
  }

  async function handleRenameSession(targetSessionId: string, title: string) {
    const previous = history.find(session => session.session_id === targetSessionId)
    const previousActiveTitle = sessionId === targetSessionId ? sessionTitle : null
    const now = new Date().toISOString()

    setHistory(prev => prev.map(session =>
      session.session_id === targetSessionId
        ? { ...session, title, title_source: "manual", title_updated_at: now }
        : session,
    ))
    if (sessionId === targetSessionId) {
      applySessionTitle(targetSessionId, title, "manual")
    }

    try {
      const updated = await renameSessionTitle(targetSessionId, title)
      setHistory(prev => prev.map(session =>
        session.session_id === targetSessionId ? { ...session, ...updated } : session,
      ))
      if (sessionId === targetSessionId) {
        applySessionTitle(targetSessionId, updated.title ?? title, updated.title_source ?? "manual")
      }
    } catch (err) {
      if (previous) {
        setHistory(prev => prev.map(session =>
          session.session_id === targetSessionId ? previous : session,
        ))
      }
      if (sessionId === targetSessionId) {
        applySessionTitle(targetSessionId, previousActiveTitle, previous?.title_source ?? null)
      }
      throw err
    }
  }

  function handleClearChat() {
    clearChat()
    setActivePanel("chat")
  }

  function handleInputChange(value: string) {
    inputValueRef.current = value
    setInput(value)
  }

  function handleInputSelectionChange(start: number, end: number) {
    inputSelectionRef.current = { start, end }
  }

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen) onClose()
  }

  const compactWorkflowSlot = viewMode === "compact" ? (
    <AgentWorkflowLauncher
      workflows={workflows}
      isLoading={workflowsQuery.isLoading}
      isError={workflowsQuery.isError}
      isStreaming={isStreaming}
      workflowTicker={workflowTicker}
      onTickerChange={setWorkflowTicker}
      onWorkflow={handleWorkflow}
      variant="compact"
    />
  ) : null

  return (
    <RadixDialog.Root open={open} onOpenChange={handleOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--background-overlay))]/45 backdrop-blur-[2px]" />
        <RadixDialog.Content
          className={cn(
            "fixed z-50 flex flex-col overflow-hidden bg-app focus:outline-none",
            viewMode === "console"
              ? "bottom-[max(1rem,var(--safe-bottom))] left-[max(1rem,var(--safe-left))] right-[max(1rem,var(--safe-right))] top-[max(1rem,var(--safe-top))] rounded-[1.25rem] border border-app shadow-[var(--shadow-floating)]"
              : "right-0 top-0 h-[100dvh] w-full border-l border-app shadow-[var(--shadow-floating)] sm:w-[32rem]",
          )}
          onOpenAutoFocus={event => {
            event.preventDefault()
            if (activePanel === "chat") {
              window.setTimeout(focusComposerTextarea, 120)
            }
          }}
        >
          <RadixDialog.Title className="sr-only">Stan assistant</RadixDialog.Title>
          <RadixDialog.Description className="sr-only">
            Market intelligence assistant with chat, workflow controls, history, and preferences.
          </RadixDialog.Description>

          <AgentChatHeader
            activePanel={activePanel}
            viewMode={viewMode}
            isDesktop={isDesktop}
            canClear={messages.length > 0}
            sessionId={sessionId}
            sessionTitle={sessionTitle}
            onRenameTitle={title => {
              if (!sessionId) return Promise.resolve()
              return handleRenameSession(sessionId, title)
            }}
            onBack={() => setActivePanel("chat")}
            onShowHistory={handleShowHistory}
            onShowPreferences={handleShowPreferences}
            onToggleViewMode={() => setViewModeOverride(viewMode === "console" ? "compact" : "console")}
            onClearChat={handleClearChat}
            onClose={onClose}
          />

          <div className="flex min-h-0 flex-1">
            <section className="flex min-w-0 flex-1 flex-col">
              {activePanel === "history" ? (
                <AgentHistoryPanel
                  history={history}
                  loading={historyLoading}
                  onLoadSession={handleLoadSession}
                  onDeleteSession={handleDeleteSession}
                  onRenameSession={handleRenameSession}
                />
              ) : activePanel === "preferences" ? (
                <AgentPreferencesPanel
                  draftPreferences={draftPreferences}
                  onChange={updateDraftPreferences}
                  onSave={handleSavePreferences}
                  isSaving={preferencesMutation.isPending}
                  saveError={preferenceSaveError}
                  preferencesUnavailable={preferencesQuery.isError && !preferenceSaveError}
                />
              ) : (
                <>
                  <AgentMessageStream
                    messages={messages}
                    isStreaming={isStreaming}
                    error={error}
                    onPrompt={handleQuickPrompt}
                    onOpenTrace={message =>
                      openDecisionTrace({
                        kind: "agent_message",
                        record: {},
                        sessionId,
                        message,
                      })
                    }
                    scrollContainerRef={messagesScrollRef}
                    messagesEndRef={messagesEndRef}
                  />
                  <AgentChatQueuePanel
                    queuedMessages={queuedMessages}
                    onSendNow={id => { void sendQueuedMessageNow(id) }}
                    onEdit={id => {
                      const text = editQueuedMessage(id)
                      if (text != null) setInput(text)
                    }}
                    onRemove={removeQueuedMessage}
                    onClear={clearQueuedMessages}
                  />
                  <AgentChatComposer
                    input={input}
                    onInputChange={handleInputChange}
                    onInputSelectionChange={handleInputSelectionChange}
                    onSend={handleSend}
                    onStop={stopStreaming}
                    isStreaming={isStreaming}
                    queuedCount={queuedMessages.length}
                    textareaRef={textareaRef}
                    compactWorkflowSlot={compactWorkflowSlot}
                    workflowsOpen={showCompactWorkflows}
                    onToggleWorkflows={() => setShowCompactWorkflows(value => !value)}
                  />
                </>
              )}
            </section>

            {activePanel === "chat" && viewMode === "console" && (
              <AgentContextPane
                screenContext={screenContext}
                workflows={workflows}
                workflowsLoading={workflowsQuery.isLoading}
                workflowsError={workflowsQuery.isError}
                isStreaming={isStreaming}
                workflowTicker={workflowTicker}
                onTickerChange={setWorkflowTicker}
                onWorkflow={handleWorkflow}
                promptGroups={QUICK_PROMPT_GROUPS}
                onPrompt={handleQuickPrompt}
              />
            )}
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
