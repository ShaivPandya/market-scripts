import { useCallback, useEffect, useRef, useState } from "react"
import type { ScreenContext } from "@/contexts/ScreenContext"

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

export interface SessionSummary {
  session_id: string
  started_at: string | null
  ended_at: string | null
  message_count: number
  key_tickers: string[] | null
  key_topics: string[] | null
  summary: string | null
}

export type AgentPreferenceLevel = "less" | "balanced" | "more"
export type AgentPersonality = "friendly" | "pragmatic"

export interface AgentResponsePreferences {
  personality: AgentPersonality
  warmth: AgentPreferenceLevel
  enthusiasm: AgentPreferenceLevel
  headers_lists: AgentPreferenceLevel
  emoji: AgentPreferenceLevel
  fast_answers: boolean
  thinking_enabled: boolean
  custom_instructions?: string | null
}

interface AgentChatState {
  messages: AgentMessage[]
  isStreaming: boolean
  error: string | null
  sessionId: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STORAGE_KEY = "agent-chat"
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/+$/, "")

function schemaHeaders(method: string, url: string): Record<string, string> {
  const parsed = new URL(url, window.location.origin)
  return {
    "X-Request-Schema-Name": `${method.toLowerCase()}:${parsed.pathname}`,
    "X-Request-Schema-Version": "1",
  }
}

function truncateText(value: string, maxLen: number): string {
  return value.length <= maxLen ? value : `${value.slice(0, maxLen - 1)}…`
}

function decodeHtmlEntities(value: string): string {
  const textarea = document.createElement("textarea")
  textarea.innerHTML = value
  return textarea.value
}

function extractJsonError(data: unknown): string | null {
  if (data == null) return null
  if (typeof data === "string") return data.trim() || null
  if (typeof data !== "object") return null

  const rec = data as Record<string, unknown>
  for (const key of ["detail", "message", "error"]) {
    const value = rec[key]
    if (typeof value === "string" && value.trim()) return value.trim()
  }
  return null
}

function formatChatHttpError(response: Response, body: string): string {
  const statusPrefix = `${response.status}: `
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? ""
  const trimmed = body.trim()

  if (contentType.includes("json") || trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const message = extractJsonError(JSON.parse(trimmed))
      if (message) return statusPrefix + truncateText(message, 500)
    } catch {
      // Fall through to text/html handling.
    }
  }

  if (/^<!doctype html/i.test(trimmed) || /<html[\s>]/i.test(trimmed)) {
    const title = trimmed.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]
    const titleText = title
      ? decodeHtmlEntities(title.replace(/\s+/g, " ").trim())
      : "Upstream returned an HTML error page"
    if (response.status === 502) {
      return "502: Bad gateway from the API proxy/origin before the agent stream completed."
    }
    return statusPrefix + truncateText(titleText, 300)
  }

  return statusPrefix + truncateText(trimmed || response.statusText || "Request failed", 500)
}

function loadState(): AgentChatState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as { messages?: AgentMessage[]; sessionId?: string }
      if (Array.isArray(parsed.messages)) {
        return {
          messages: parsed.messages.map(m => ({ ...m, isStreaming: false })),
          isStreaming: false,
          error: null,
          sessionId: parsed.sessionId ?? null,
        }
      }
    }
  } catch {
    /* ignore */
  }
  return { messages: [], isStreaming: false, error: null, sessionId: null }
}

async function saveSessionToServer(messages: AgentMessage[], sessionId: string | null): Promise<string | null> {
  if (messages.length === 0) return null
  try {
    const body: Record<string, unknown> = {
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
      })),
    }
    if (sessionId) body.session_id = sessionId
    const resp = await fetch(`${BASE_URL}/memory/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...schemaHeaders("POST", `${BASE_URL}/memory/sessions`) },
      credentials: "include",
      body: JSON.stringify(body),
    })
    if (!resp.ok) return null
    const data = await resp.json()
    return data.session_id ?? null
  } catch {
    return null
  }
}

async function summarizeSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${BASE_URL}/memory/sessions/${sessionId}/summarize`, {
      method: "POST",
      headers: schemaHeaders("POST", `${BASE_URL}/memory/sessions/${sessionId}/summarize`),
      credentials: "include",
    })
  } catch {
    /* best-effort */
  }
}

export async function fetchSessionHistory(limit = 20): Promise<SessionSummary[]> {
  try {
    const resp = await fetch(`${BASE_URL}/memory/sessions?limit=${limit}`, {
      credentials: "include",
    })
    if (!resp.ok) return []
    return await resp.json()
  } catch {
    return []
  }
}

export async function fetchSession(sessionId: string): Promise<{ transcript: AgentMessage[] } | null> {
  try {
    const resp = await fetch(`${BASE_URL}/memory/sessions/${sessionId}`, {
      credentials: "include",
    })
    if (!resp.ok) return null
    const data = await resp.json()
    const transcript: AgentMessage[] = (data.transcript ?? []).map((m: Record<string, unknown>, i: number) => ({
      id: (m.id as string) ?? `restored-${i}`,
      role: m.role as "user" | "assistant",
      content: (m.content as string) ?? "",
      timestamp: (m.timestamp as number) ?? Date.now(),
      isStreaming: false,
    }))
    return { transcript }
  } catch {
    return null
  }
}

export async function deleteSession(sessionId: string): Promise<boolean> {
  try {
    const resp = await fetch(`${BASE_URL}/memory/sessions/${sessionId}`, {
      method: "DELETE",
      credentials: "include",
    })
    return resp.ok
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAgentChat() {
  const [state, setState] = useState<AgentChatState>(loadState)
  const abortRef = useRef<AbortController | null>(null)

  // Persist messages to localStorage
  useEffect(() => {
    const toSave = state.messages.filter(m => !m.isStreaming || m.content.length > 0)
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ messages: toSave, sessionId: state.sessionId }),
    )
  }, [state.messages, state.sessionId])

  // ------ sendMessage ------
  const sendMessage = useCallback(async (
    content: string,
    screenContext?: ScreenContext | null,
    responsePreferences?: AgentResponsePreferences | null,
  ) => {
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
      ...prev,
      messages: [...prev.messages, userMsg, assistantMsg],
      isStreaming: true,
      error: null,
    }))

    const assistantId = assistantMsg.id

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch(`${BASE_URL}/agent/chat/v2`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...schemaHeaders("POST", `${BASE_URL}/agent/chat/v2`) },
        credentials: "include",
        body: JSON.stringify({
          session_id: state.sessionId,
          message: content,
          ...(screenContext && {
            screen_context: {
              page_name: screenContext.pageName,
              route: screenContext.route,
              ticker: screenContext.ticker ?? null,
              metrics: screenContext.metrics ?? null,
              filters: screenContext.filters ?? null,
              summary: screenContext.summary ?? null,
              corresponding_tools: screenContext.correspondingTools ?? null,
            },
          }),
          ...(responsePreferences && {
            response_preferences: responsePreferences,
          }),
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errText = await response.text().catch(() => "Request failed")
        throw new Error(formatChatHttpError(response, errText))
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
                sessionId: (data.session_id as string) ?? prev.sessionId,
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
      const message = err instanceof Error ? err.message : String(err)
      setState(prev => ({
        ...prev,
        error: message,
        isStreaming: false,
        messages: prev.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false } : m,
        ),
      }))
    }
  }, [state.sessionId])

  // ------ stopStreaming ------
  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // ------ clearChat ------
  const clearChat = useCallback(() => {
    abortRef.current?.abort()
    // Summarize the ending session before clearing
    if (state.sessionId) {
      summarizeSession(state.sessionId)
    }
    setState({ messages: [], isStreaming: false, error: null, sessionId: null })
  }, [state.sessionId])

  // ------ loadSession ------
  const loadSession = useCallback(async (sessionId: string) => {
    const data = await fetchSession(sessionId)
    if (data && data.transcript.length > 0) {
      setState({
        messages: data.transcript,
        isStreaming: false,
        error: null,
        sessionId,
      })
    }
  }, [])

  return {
    messages: state.messages,
    isStreaming: state.isStreaming,
    error: state.error,
    sessionId: state.sessionId,
    sendMessage,
    stopStreaming,
    clearChat,
    loadSession,
  }
}
