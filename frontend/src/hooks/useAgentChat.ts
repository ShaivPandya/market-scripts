import { useCallback, useEffect, useRef, useState } from "react"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ToolCall {
  name: string
  id: string
  status: "pending" | "ok" | "error"
}

export interface AgentMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  toolCalls?: ToolCall[]
  isStreaming?: boolean
}

interface AgentChatState {
  messages: AgentMessage[]
  isStreaming: boolean
  error: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STORAGE_KEY = "agent-chat"
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/+$/, "")

function loadState(): AgentChatState {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as { messages?: AgentMessage[] }
      if (Array.isArray(parsed.messages)) {
        return {
          messages: parsed.messages.map(m => ({ ...m, isStreaming: false })),
          isStreaming: false,
          error: null,
        }
      }
    }
  } catch {
    /* ignore */
  }
  return { messages: [], isStreaming: false, error: null }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAgentChat() {
  const [state, setState] = useState<AgentChatState>(loadState)
  const abortRef = useRef<AbortController | null>(null)

  // Persist messages to sessionStorage
  useEffect(() => {
    const toSave = state.messages.filter(m => !m.isStreaming || m.content.length > 0)
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ messages: toSave }))
  }, [state.messages])

  // ------ sendMessage ------
  const sendMessage = useCallback(async (content: string) => {
    const userMsg: AgentMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: Date.now(),
    }

    const assistantMsg: AgentMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      toolCalls: [],
      isStreaming: true,
    }

    setState(prev => ({
      messages: [...prev.messages, userMsg, assistantMsg],
      isStreaming: true,
      error: null,
    }))

    const assistantId = assistantMsg.id

    // Build the message history for the API (exclude the empty streaming msg)
    const apiMessages = [...state.messages, userMsg]
      .filter(m => m.content.length > 0)
      .map(m => ({ role: m.role, content: m.content }))

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch(`${BASE_URL}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ messages: apiMessages }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errText = await response.text().catch(() => "Request failed")
        throw new Error(`${response.status}: ${errText}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split("\n\n")
        buffer = events.pop()! // keep incomplete chunk

        for (const raw of events) {
          if (!raw.trim()) continue
          const typeMatch = raw.match(/^event: (.+)$/m)
          const dataMatch = raw.match(/^data: (.+)$/m)
          if (!typeMatch || !dataMatch) continue

          const eventType = typeMatch[1]
          let data: Record<string, unknown>
          try {
            data = JSON.parse(dataMatch[1])
          } catch {
            continue
          }

          switch (eventType) {
            case "delta":
              setState(prev => ({
                ...prev,
                messages: prev.messages.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + (data.text as string) }
                    : m,
                ),
              }))
              break

            case "tool_call":
              setState(prev => ({
                ...prev,
                messages: prev.messages.map(m =>
                  m.id === assistantId
                    ? {
                        ...m,
                        toolCalls: [
                          ...(m.toolCalls ?? []),
                          {
                            name: data.name as string,
                            id: data.id as string,
                            status: "pending" as const,
                          },
                        ],
                      }
                    : m,
                ),
              }))
              break

            case "tool_result":
              setState(prev => ({
                ...prev,
                messages: prev.messages.map(m =>
                  m.id === assistantId
                    ? {
                        ...m,
                        toolCalls: m.toolCalls?.map(tc =>
                          tc.id === data.id
                            ? { ...tc, status: (data.status as "ok" | "error") }
                            : tc,
                        ),
                      }
                    : m,
                ),
              }))
              break

            case "done":
              setState(prev => ({
                ...prev,
                isStreaming: false,
                messages: prev.messages.map(m =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m,
                ),
              }))
              break

            case "error":
              setState(prev => ({
                ...prev,
                error: (data.message as string) || "An error occurred",
                isStreaming: false,
                messages: prev.messages.map(m =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m,
                ),
              }))
              break
          }
        }
      }

      // If stream ended without a done event, finalize
      setState(prev => {
        if (!prev.isStreaming) return prev
        return {
          ...prev,
          isStreaming: false,
          messages: prev.messages.map(m =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        }
      })
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setState(prev => ({
          ...prev,
          isStreaming: false,
          messages: prev.messages.map(m =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        }))
        return
      }
      setState(prev => ({
        ...prev,
        error: String(err),
        isStreaming: false,
        messages: prev.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false } : m,
        ),
      }))
    }
  }, [state.messages])

  // ------ stopStreaming ------
  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // ------ clearChat ------
  const clearChat = useCallback(() => {
    abortRef.current?.abort()
    setState({ messages: [], isStreaming: false, error: null })
  }, [])

  return {
    messages: state.messages,
    isStreaming: state.isStreaming,
    error: state.error,
    sendMessage,
    stopStreaming,
    clearChat,
  }
}
