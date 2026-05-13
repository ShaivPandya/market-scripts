import * as RadixDialog from "@radix-ui/react-dialog"
import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { cn } from "@/lib/utils"
import {
  fetchAgentResponsePreferences,
  fetchAgentWorkflows,
  updateAgentResponsePreferences,
  type AgentResponsePreferences,
  type AgentWorkflow,
} from "@/lib/api"
import { deleteSession, fetchSessionHistory, useAgentChat, type SessionSummary } from "@/hooks/useAgentChat"
import type { ScreenContext } from "@/contexts/ScreenContext"
import { AgentChatComposer } from "./AgentChatComposer"
import { AgentChatHeader } from "./AgentChatHeader"
import { AgentContextPane } from "./AgentContextPane"
import { AgentHistoryPanel } from "./AgentHistoryPanel"
import { AgentMessageStream } from "./AgentMessageStream"
import { AgentPreferencesPanel } from "./AgentPreferencesPanel"
import { AgentWorkflowLauncher } from "./AgentWorkflowLauncher"
import { QUICK_PROMPT_GROUPS } from "./agentChatPrompts"
import { resizeChatTextarea } from "./agentChatTextarea"
import { useMediaQuery } from "./useMediaQuery"
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

export function AgentChat({ open, onClose, screenContext }: AgentChatProps) {
  const { messages, isStreaming, error, sendMessage, stopStreaming, clearChat, loadSession } = useAgentChat()
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const wasStreamingRef = useRef(isStreaming)
  const pendingWorkflowInvalidationRef = useRef<WorkflowInvalidationTarget | null>(null)

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth" })
  }, [messages, isStreaming, activePanel])

  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && pendingWorkflowInvalidationRef.current) {
      invalidateWorkflowRunQueries(queryClient, pendingWorkflowInvalidationRef.current)
      pendingWorkflowInvalidationRef.current = null
    }
    wasStreamingRef.current = isStreaming
  }, [isStreaming, queryClient])

  useEffect(() => {
    if (open && activePanel === "chat") {
      window.setTimeout(() => textareaRef.current?.focus(), 160)
    }
  }, [open, activePanel])

  useEffect(() => {
    if (textareaRef.current) resizeChatTextarea(textareaRef.current)
  }, [input, open, viewMode])

  useEffect(() => {
    if (preferencesQuery.data) {
      localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(normalizePreferences(preferencesQuery.data)))
    }
  }, [preferencesQuery.data])

  function updateDraftPreferences(updater: (prev: AgentResponsePreferences) => AgentResponsePreferences) {
    setDraftPreferences(updater)
    setPreferenceSaveError(null)
  }

  function handleSend() {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    pendingWorkflowInvalidationRef.current = workflowTargetFromCommand(trimmed)
    setInput("")
    setActivePanel("chat")
    sendMessage(trimmed, screenContext, activePreferences)
  }

  function handleQuickPrompt(prompt: string) {
    if (isStreaming) return
    pendingWorkflowInvalidationRef.current = null
    setActivePanel("chat")
    sendMessage(prompt, screenContext, activePreferences)
  }

  function handleWorkflow(workflow: AgentWorkflow) {
    if (isStreaming) return
    const ticker = workflowTicker.trim().toUpperCase()
    if (workflow.requiresTicker && !ticker) return
    const command = workflow.requiresTicker
      ? `/workflow:${workflow.name}:${ticker}`
      : `/workflow:${workflow.name}`
    pendingWorkflowInvalidationRef.current = { workflowName: workflow.name, ticker: ticker || null }
    setActivePanel("chat")
    sendMessage(command, screenContext, activePreferences, { durable: true })
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

  function handleClearChat() {
    clearChat()
    setActivePanel("chat")
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
              window.setTimeout(() => textareaRef.current?.focus(), 120)
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
                    messagesEndRef={messagesEndRef}
                  />
                  <AgentChatComposer
                    input={input}
                    onInputChange={setInput}
                    onSend={handleSend}
                    onStop={stopStreaming}
                    isStreaming={isStreaming}
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
