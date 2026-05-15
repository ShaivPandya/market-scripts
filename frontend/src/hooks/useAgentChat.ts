import { useCallback, useEffect, useRef, useState } from "react"
import type { ScreenContext } from "@/contexts/ScreenContext"
import type { AgentResponsePreferences } from "@/lib/api"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ToolCall {
  name: string
  id: string
  status: "pending" | "running" | "ok" | "error" | "blocked" | "timeout" | "retrying" | "partial" | "cancelled"
  message?: string
  policyDecisionId?: string
  elapsedMs?: number
}

export interface EgressRecord {
  id: string
  decision: "allowed" | "allowed_with_warning" | "blocked" | string
  decisionReason?: string
  dataSensitivity?: string
  provider?: string
  model?: string
  policyDecisionId?: string
}

export interface AgentMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  clientTurnId?: string
  toolCalls?: ToolCall[]
  isStreaming?: boolean
  statusText?: string
  egressRecords?: EgressRecord[]
}

export interface SessionSummary {
  session_id: string
  started_at: string | null
  ended_at: string | null
  message_count: number
  key_tickers: string[] | null
  key_topics: string[] | null
  summary: string | null
  title: string | null
  title_source: string | null
  title_updated_at: string | null
}

interface AgentChatState {
  messages: AgentMessage[]
  isStreaming: boolean
  error: string | null
  sessionId: string | null
  sessionTitle: string | null
  sessionTitleSource: string | null
  activeJob: ActiveAgentJob | null
}

interface ActiveAgentJob {
  jobId: string
  assistantId: string
  afterSeq: number
  clientTurnId: string
}

interface AgentJobEvent {
  seq: number
  event_type:
    | "status"
    | "delta"
    | "phase"
    | "tool_call"
    | "tool_result"
    | "tool_progress"
    | "policy_failure"
    | "budget_update"
    | "egress_recorded"
    | "blocked"
    | "timeout"
    | "cancelled"
    | "error"
    | "done"
  payload: Record<string, unknown>
}

interface AgentJobResponse {
  job_id: string
  status: "queued" | "running" | "done" | "error" | "cancelled"
  session_id?: string | null
  timeout_s?: number
  error?: string
  result?: unknown
  events?: AgentJobEvent[]
  next_seq?: number
}

interface AgentStreamEvent {
  event_type: AgentJobEvent["event_type"] | "ping" | "handoff"
  payload: Record<string, unknown>
}

interface AgentSendOptions {
  durable?: boolean
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STORAGE_KEY = "agent-chat-current"
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/+$/, "")

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

function truncateTitle(value: string, maxLen: number): string {
  if (value.length <= maxLen) return value
  const candidate = value.slice(0, maxLen).trimEnd()
  const wordBoundary = candidate.lastIndexOf(" ")
  return (wordBoundary >= 40 ? candidate.slice(0, wordBoundary) : candidate).replace(/[-:,.!?\s]+$/, "")
}

export function deriveSessionTitleFromText(value: string): string | null {
  let text = value.replace(/\s+/g, " ").trim()
  if (!text) return null
  const workflowMatch = text.match(/^\/workflow:([A-Za-z0-9_]+)(?::([A-Za-z0-9._=-]+))?(?:\s+(.*))?$/)
  if (workflowMatch) {
    const workflow = workflowMatch[1].replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase())
    const ticker = workflowMatch[2]?.toUpperCase() ?? ""
    const trailing = workflowMatch[3]?.trim() ?? ""
    text = trailing || (ticker ? `${ticker} ${workflow}` : workflow)
  }
  text = text.replace(/\s+/g, " ").replace(/^[-:,.!?\s]+|[-:,.!?\s]+$/g, "")
  return text ? truncateTitle(text, 80) : null
}

function deriveSessionTitleFromMessages(messages: AgentMessage[]): string | null {
  const firstUser = messages.find(message => message.role === "user" && message.content.trim())
  return firstUser ? deriveSessionTitleFromText(firstUser.content) : null
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

function normalizeToolStatus(value: unknown): ToolCall["status"] {
  const allowed: ToolCall["status"][] = [
    "pending",
    "running",
    "ok",
    "error",
    "blocked",
    "timeout",
    "retrying",
    "partial",
    "cancelled",
  ]
  return typeof value === "string" && allowed.includes(value as ToolCall["status"])
    ? (value as ToolCall["status"])
    : "ok"
}

function normalizeToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index): ToolCall[] => {
    if (typeof item === "string" && item.trim()) {
      return [{ name: item, id: item, status: "ok" }]
    }
    if (!item || typeof item !== "object") return []
    const rec = item as Record<string, unknown>
    const name = typeof rec.name === "string" ? rec.name : typeof rec.tool === "string" ? rec.tool : null
    if (!name) return []
    const id = typeof rec.id === "string" ? rec.id : typeof rec.call_id === "string" ? rec.call_id : `${name}-${index}`
    return [{
      name,
      id,
      status: normalizeToolStatus(rec.status),
      message: typeof rec.message === "string" ? rec.message : undefined,
      policyDecisionId: typeof rec.policy_decision_id === "string" ? rec.policy_decision_id : undefined,
      elapsedMs: typeof rec.elapsed_ms === "number" ? rec.elapsed_ms : undefined,
    }]
  })
}

function mergeToolCalls(existing: ToolCall[] | undefined, incoming: ToolCall[]): ToolCall[] {
  if (!incoming.length) return existing ?? []
  const byId = new Map<string, ToolCall>()
  for (const call of existing ?? []) byId.set(call.id, call)
  for (const call of incoming) byId.set(call.id, { ...byId.get(call.id), ...call })
  return [...byId.values()]
}

function normalizeEgressRecord(value: Record<string, unknown>): EgressRecord | null {
  const policyDecisionId = typeof value.policy_decision_id === "string" ? value.policy_decision_id : undefined
  const decision = typeof value.decision === "string" ? value.decision : undefined
  if (!policyDecisionId || !decision) return null
  return {
    id: policyDecisionId,
    policyDecisionId,
    decision,
    decisionReason: typeof value.decision_reason === "string" ? value.decision_reason : undefined,
    dataSensitivity: typeof value.data_sensitivity === "string" ? value.data_sensitivity : undefined,
    provider: typeof value.provider === "string" ? value.provider : undefined,
    model: typeof value.model === "string" ? value.model : undefined,
  }
}

function mergeEgressRecords(existing: EgressRecord[] | undefined, incoming: EgressRecord): EgressRecord[] {
  const byId = new Map<string, EgressRecord>()
  for (const record of existing ?? []) byId.set(record.id, record)
  byId.set(incoming.id, { ...byId.get(incoming.id), ...incoming })
  return [...byId.values()].slice(-6)
}

function toolCallsFromDonePayload(data: Record<string, unknown>): ToolCall[] {
  const explicit = normalizeToolCalls(data.tool_calls)
  if (explicit.length) return explicit
  return normalizeToolCalls(data.tools_used)
}

function buildAgentRequestBody(
  sessionId: string | null,
  clientTurnId: string,
  content: string,
  screenContext?: ScreenContext | null,
  responsePreferences?: AgentResponsePreferences | null,
): Record<string, unknown> {
  return {
    session_id: sessionId,
    client_turn_id: clientTurnId,
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
  }
}

function parseSseFrame(frame: string): AgentStreamEvent | null {
  let eventType: string | null = null
  const dataLines: string[] = []
  for (const rawLine of frame.split(/\r?\n/)) {
    const line = rawLine.trimEnd()
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim()
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim())
    }
  }
  if (!eventType || dataLines.length === 0) return null
  try {
    const payload = JSON.parse(dataLines.join("\n"))
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null
    return { event_type: eventType as AgentStreamEvent["event_type"], payload: payload as Record<string, unknown> }
  } catch {
    return null
  }
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
      const parsed = JSON.parse(raw) as {
        messages?: AgentMessage[]
        sessionId?: string
        sessionTitle?: string | null
        sessionTitleSource?: string | null
        activeJob?: ActiveAgentJob | null
      }
      if (Array.isArray(parsed.messages)) {
        const activeJob = parsed.activeJob ?? null
        return {
          messages: parsed.messages.map(m => ({
            ...m,
            isStreaming: activeJob?.assistantId === m.id,
            statusText: activeJob?.assistantId === m.id ? (m.statusText || "Reconnecting...") : m.statusText,
          })),
          isStreaming: Boolean(activeJob),
          error: null,
          sessionId: parsed.sessionId ?? null,
          sessionTitle: parsed.sessionTitle ?? deriveSessionTitleFromMessages(parsed.messages),
          sessionTitleSource: parsed.sessionTitleSource ?? null,
          activeJob,
        }
      }
    }
  } catch {
    /* ignore */
  }
  return { messages: [], isStreaming: false, error: null, sessionId: null, sessionTitle: null, sessionTitleSource: null, activeJob: null }
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

export async function fetchSession(sessionId: string): Promise<{
  transcript: AgentMessage[]
  title: string | null
  title_source: string | null
  title_updated_at: string | null
} | null> {
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
      clientTurnId: typeof m.clientTurnId === "string"
        ? m.clientTurnId
        : typeof m.client_turn_id === "string"
          ? m.client_turn_id
          : undefined,
      toolCalls: normalizeToolCalls(m.toolCalls ?? m.tool_calls),
      isStreaming: false,
    }))
    return {
      transcript,
      title: typeof data.title === "string" ? data.title : null,
      title_source: typeof data.title_source === "string" ? data.title_source : null,
      title_updated_at: typeof data.title_updated_at === "string" ? data.title_updated_at : null,
    }
  } catch {
    return null
  }
}

export async function renameSessionTitle(sessionId: string, title: string): Promise<SessionSummary> {
  const url = `${BASE_URL}/memory/sessions/${encodeURIComponent(sessionId)}`
  const resp = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...schemaHeaders("PATCH", url) },
    credentials: "include",
    body: JSON.stringify({ title }),
  })
  return readJsonResponse<SessionSummary>(resp)
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

async function readJsonResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Request failed")
    throw new Error(formatChatHttpError(resp, errText))
  }
  return await resp.json() as T
}

async function startAgentJob(body: Record<string, unknown>, signal?: AbortSignal): Promise<AgentJobResponse> {
  const url = `${BASE_URL}/agent/chat/async`
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...schemaHeaders("POST", url) },
    credentials: "include",
    body: JSON.stringify(body),
    signal,
  })
  return readJsonResponse<AgentJobResponse>(resp)
}

async function startLiveAgentStream(
  body: Record<string, unknown>,
  onEvent: (event: AgentStreamEvent) => void,
  signal?: AbortSignal,
): Promise<{ handoff?: AgentJobResponse }> {
  const url = `${BASE_URL}/agent/chat`
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...schemaHeaders("POST", url) },
    credentials: "include",
    body: JSON.stringify(body),
    signal,
  })
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Request failed")
    throw new Error(formatChatHttpError(resp, errText))
  }
  if (!resp.body) {
    throw new Error("Agent stream did not include a response body.")
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  for (;;) {
    const { done, value } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: !done })
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? ""
      for (const frame of frames) {
        const event = parseSseFrame(frame)
        if (!event || event.event_type === "ping") continue
        if (event.event_type === "handoff") {
          return { handoff: event.payload as unknown as AgentJobResponse }
        }
        onEvent(event)
      }
    }
    if (done) break
  }

  const trailing = parseSseFrame(buffer)
  if (trailing && trailing.event_type !== "ping") {
    if (trailing.event_type === "handoff") {
      return { handoff: trailing.payload as unknown as AgentJobResponse }
    }
    onEvent(trailing)
  }
  return {}
}

async function fetchAgentJobEvents(jobId: string, afterSeq: number, signal?: AbortSignal): Promise<AgentJobResponse> {
  const params = new URLSearchParams({ after_seq: String(afterSeq), wait_ms: "10000" })
  const resp = await fetch(`${BASE_URL}/agent/chat/async/${encodeURIComponent(jobId)}/events?${params}`, {
    credentials: "include",
    signal,
  })
  return readJsonResponse<AgentJobResponse>(resp)
}

async function cancelAgentJob(jobId: string): Promise<void> {
  const url = `${BASE_URL}/agent/chat/async/${encodeURIComponent(jobId)}/cancel`
  const resp = await fetch(url, {
    method: "POST",
    headers: schemaHeaders("POST", url),
    credentials: "include",
  })
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "")
    throw new Error(formatChatHttpError(resp, errText))
  }
}

function nextSeqFrom(events: AgentJobEvent[] | undefined, fallback: number): number {
  if (!events?.length) return fallback
  return Math.max(fallback, ...events.map(event => Number(event.seq) || 0))
}

function statusTextForEventStatus(value: unknown): string {
  if (value === "starting") return "Starting..."
  if (value === "queued") return "Queued..."
  if (value === "cancelled") return "Cancelled."
  return "Running..."
}

function statusTextForPolledStatus(status: AgentJobResponse["status"], current?: string): string | undefined {
  if (status === "running") return "Running..."
  if (status === "queued") return current === "Starting..." ? current : "Queued..."
  return undefined
}

function statusTextForPhase(data: Record<string, unknown>): string | undefined {
  const phase = typeof data.phase === "string" ? data.phase : ""
  const label = typeof data.label === "string" && data.label.trim() ? data.label.trim() : null
  if (phase === "tool_running" && Array.isArray(data.tool_names) && data.tool_names.includes("get_portfolio")) {
    return "Reading portfolio..."
  }
  if (label) return label
  if (phase === "model_thinking") return "Thinking..."
  if (phase === "tool_running") return "Running tools..."
  if (phase === "model_writing") return "Writing answer..."
  if (phase === "finalizing") return "Finalizing..."
  return undefined
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAgentChat() {
  const [state, setState] = useState<AgentChatState>(loadState)
  const abortRef = useRef<AbortController | null>(null)
  const inFlightRef = useRef(false)
  const activeJobRef = useRef<ActiveAgentJob | null>(state.activeJob)
  const initialActiveJobRef = useRef<ActiveAgentJob | null>(state.activeJob)

  // Persist messages to localStorage. During streaming, debounce this work so
  // token updates do not synchronously serialize the whole transcript.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const toSave = state.messages.filter(
        m => !m.isStreaming || m.content.length > 0 || state.activeJob?.assistantId === m.id,
      )
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          messages: toSave,
          sessionId: state.sessionId,
          sessionTitle: state.sessionTitle,
          sessionTitleSource: state.sessionTitleSource,
          activeJob: state.activeJob,
        }),
      )
    }, state.isStreaming ? 1000 : 0)
    return () => window.clearTimeout(timer)
  }, [state.messages, state.sessionId, state.sessionTitle, state.sessionTitleSource, state.activeJob, state.isStreaming])

  const applyJobEvents = useCallback((assistantId: string, events: AgentJobEvent[], fallbackSessionId?: string | null) => {
    if (!events.length && !fallbackSessionId) return
    setState(prev => {
      let next = prev
      let sessionId = fallbackSessionId ?? prev.sessionId

      for (const event of events) {
        const data = event.payload ?? {}
        switch (event.event_type) {
          case "status": {
            const rawStatus = typeof data.status === "string" ? data.status : "running"
            const label = statusTextForEventStatus(rawStatus)
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId ? { ...m, statusText: label } : m,
              ),
            }
            break
          }
          case "delta":
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + (typeof data.text === "string" ? data.text : ""), statusText: undefined }
                  : m,
              ),
            }
            break
          case "phase": {
            const label = statusTextForPhase(data)
            if (!label) break
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId ? { ...m, statusText: label } : m,
              ),
            }
            break
          }
          case "tool_call":
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      statusText: undefined,
                      toolCalls: mergeToolCalls(
                        m.toolCalls,
                        normalizeToolCalls([{ name: data.name, id: data.id, status: "pending" }]),
                      ),
                    }
                  : m,
              ),
            }
            break
          case "tool_progress":
          case "policy_failure":
          case "blocked":
          case "timeout":
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? { ...m, statusText: undefined, toolCalls: mergeToolCalls(m.toolCalls, normalizeToolCalls([data])) }
                  : m,
              ),
            }
            break
          case "budget_update":
            break
          case "egress_recorded": {
            const record = normalizeEgressRecord(data)
            if (!record) break
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? { ...m, egressRecords: mergeEgressRecords(m.egressRecords, record) }
                  : m,
              ),
            }
            break
          }
          case "tool_result":
            next = {
              ...next,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? { ...m, statusText: undefined, toolCalls: mergeToolCalls(m.toolCalls, normalizeToolCalls([data])) }
                  : m,
              ),
            }
            break
          case "error":
            next = {
              ...next,
              error: (data.message as string) || "An error occurred",
              isStreaming: false,
              activeJob: null,
              messages: next.messages.map(m =>
                m.id === assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
              ),
            }
            activeJobRef.current = null
            break
          case "cancelled":
            next = {
              ...next,
              isStreaming: false,
              activeJob: null,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      isStreaming: false,
                      statusText: undefined,
                      toolCalls: (m.toolCalls ?? []).map(call =>
                        call.status === "pending" || call.status === "running"
                          ? { ...call, status: "cancelled" as const }
                          : call,
                      ),
                    }
                  : m,
              ),
            }
            activeJobRef.current = null
            break
          case "done":
            sessionId = (data.session_id as string) ?? sessionId
            next = {
              ...next,
              isStreaming: false,
              sessionId,
              sessionTitle: next.sessionTitle ?? deriveSessionTitleFromMessages(next.messages),
              sessionTitleSource: next.sessionTitleSource ?? "deterministic",
              activeJob: null,
              messages: next.messages.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      toolCalls: mergeToolCalls(m.toolCalls, toolCallsFromDonePayload(data)),
                      isStreaming: false,
                      statusText: undefined,
                    }
                  : m,
              ),
            }
            activeJobRef.current = null
            break
        }
      }

      if (fallbackSessionId && next.sessionId !== fallbackSessionId) {
        next = { ...next, sessionId: fallbackSessionId }
      }
      return next
    })
  }, [])

  const finishJobState = useCallback((assistantId: string, status: AgentJobResponse["status"], error?: string) => {
    setState(prev => ({
      ...prev,
      error: status === "error" ? (error || "Agent job failed") : prev.error,
      isStreaming: false,
      activeJob: null,
      messages: prev.messages.map(m =>
        m.id === assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
      ),
    }))
    activeJobRef.current = null
    inFlightRef.current = false
  }, [])

  const pollJob = useCallback(async (job: ActiveAgentJob, controller: AbortController) => {
    let afterSeq = job.afterSeq
    activeJobRef.current = job
    try {
      for (;;) {
        if (controller.signal.aborted) throw new DOMException("Polling cancelled", "AbortError")
        const response = await fetchAgentJobEvents(job.jobId, afterSeq, controller.signal)
        const events = response.events ?? []
        applyJobEvents(job.assistantId, events, response.session_id ?? null)
        afterSeq = response.next_seq ?? nextSeqFrom(events, afterSeq)
        if (!events.some(event => event.event_type === "status")) {
          setState(prev => ({
            ...prev,
            messages: prev.messages.map(m => {
              if (m.id !== job.assistantId) return m
              const statusText = statusTextForPolledStatus(response.status, m.statusText)
              return statusText ? { ...m, statusText } : m
            }),
          }))
        }
        if (events.some(event => event.event_type === "done" || event.event_type === "error")) {
          inFlightRef.current = false
          return
        }

        if (response.status === "done" || response.status === "error" || response.status === "cancelled") {
          if (!events.some(event => event.event_type === "done" || event.event_type === "error")) {
            finishJobState(job.assistantId, response.status, response.error)
          } else {
            inFlightRef.current = false
          }
          return
        }

        const nextJob = { ...job, afterSeq }
        activeJobRef.current = nextJob
        setState(prev => ({ ...prev, activeJob: nextJob, isStreaming: true }))
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return
      const message = err instanceof Error ? err.message : String(err)
      setState(prev => ({
        ...prev,
        error: message,
        isStreaming: false,
        activeJob: null,
        messages: prev.messages.map(m =>
          m.id === job.assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
        ),
      }))
      activeJobRef.current = null
    } finally {
      inFlightRef.current = false
    }
  }, [applyJobEvents, finishJobState])

  const beginDurableResponse = useCallback(async (
    assistantId: string,
    clientTurnId: string,
    started: AgentJobResponse,
    controller: AbortController,
  ) => {
    const events = started.events ?? []
    applyJobEvents(assistantId, events, started.session_id ?? null)
    const afterSeq = started.next_seq ?? nextSeqFrom(events, 0)

    if (events.some(event => event.event_type === "done" || event.event_type === "error")) {
      inFlightRef.current = false
      return
    }

    if (started.status === "done" || started.status === "error" || started.status === "cancelled") {
      finishJobState(assistantId, started.status, started.error)
      return
    }

    const activeJob: ActiveAgentJob = {
      jobId: started.job_id,
      assistantId,
      afterSeq,
      clientTurnId,
    }
    activeJobRef.current = activeJob
    setState(prev => ({
      ...prev,
      sessionId: started.session_id ?? prev.sessionId,
      activeJob,
      isStreaming: true,
    }))
    await pollJob(activeJob, controller)
  }, [applyJobEvents, finishJobState, pollJob])

  useEffect(() => {
    const activeJob = initialActiveJobRef.current
    if (!activeJob || inFlightRef.current) return
    const controller = new AbortController()
    abortRef.current = controller
    inFlightRef.current = true
    pollJob(activeJob, controller)
  }, [pollJob])

  // ------ sendMessage ------
  const sendMessage = useCallback(async (
    content: string,
    screenContext?: ScreenContext | null,
    responsePreferences?: AgentResponsePreferences | null,
    options?: AgentSendOptions,
  ) => {
    if (inFlightRef.current || activeJobRef.current) return
    inFlightRef.current = true
    const clientTurnId = crypto.randomUUID()
    const userMsg: AgentMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: Date.now(),
      clientTurnId,
    }

    const assistantMsg: AgentMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      clientTurnId,
      toolCalls: [],
      isStreaming: true,
      statusText: "Starting...",
    }

    const assistantId = assistantMsg.id
    const controller = new AbortController()
    abortRef.current = controller

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMsg, assistantMsg],
      isStreaming: true,
      error: null,
      sessionTitle: prev.sessionTitle ?? deriveSessionTitleFromText(content),
      sessionTitleSource: prev.sessionTitleSource ?? (prev.sessionTitle ? null : "deterministic"),
      activeJob: null,
    }))

    const body = buildAgentRequestBody(state.sessionId, clientTurnId, content, screenContext, responsePreferences)
    let sawAssistantDelta = false

    const handleAbort = () => {
      setState(prev => ({
        ...prev,
        isStreaming: false,
        activeJob: null,
        messages: prev.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
        ),
      }))
      activeJobRef.current = null
      inFlightRef.current = false
    }

    const handleError = (message: string) => {
      setState(prev => ({
        ...prev,
        error: message,
        isStreaming: false,
        activeJob: null,
        messages: prev.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
        ),
      }))
      activeJobRef.current = null
      inFlightRef.current = false
    }

    try {
      if (options?.durable) {
        const started = await startAgentJob(body, controller.signal)
        await beginDurableResponse(assistantId, clientTurnId, started, controller)
        return
      }

      let directSeq = 0
      const live = await startLiveAgentStream(body, event => {
        if (event.event_type === "delta" && typeof event.payload.text === "string" && event.payload.text) {
          sawAssistantDelta = true
        }
        const sessionId = typeof event.payload.session_id === "string" ? event.payload.session_id : null
        applyJobEvents(assistantId, [{
          seq: ++directSeq,
          event_type: event.event_type as AgentJobEvent["event_type"],
          payload: event.payload,
        }], sessionId)
      }, controller.signal)

      if (live.handoff) {
        await beginDurableResponse(assistantId, clientTurnId, live.handoff, controller)
        return
      }

      setState(prev => ({
        ...prev,
        isStreaming: false,
        activeJob: null,
        messages: prev.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false, statusText: undefined } : m,
        ),
      }))
      inFlightRef.current = false
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        handleAbort()
        return
      }

      if (!options?.durable && !sawAssistantDelta) {
        try {
          const started = await startAgentJob(body, controller.signal)
          await beginDurableResponse(assistantId, clientTurnId, started, controller)
          return
        } catch (fallbackErr) {
          if ((fallbackErr as Error).name === "AbortError") {
            handleAbort()
            return
          }
          const message = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr)
          handleError(message)
          return
        }
      }

      const message = err instanceof Error ? err.message : String(err)
      handleError(`Agent stream interrupted after the response started. ${message}`)
    }
  }, [applyJobEvents, beginDurableResponse, state.sessionId])

  // ------ stopStreaming ------
  const stopStreaming = useCallback(() => {
    const jobId = activeJobRef.current?.jobId
    abortRef.current?.abort()
    if (jobId) {
      cancelAgentJob(jobId).catch(() => undefined)
    }
    inFlightRef.current = false
    activeJobRef.current = null
    setState(prev => ({
      ...prev,
      isStreaming: false,
      activeJob: null,
      messages: prev.messages.map(m =>
        m.isStreaming
          ? {
              ...m,
              isStreaming: false,
              statusText: undefined,
              toolCalls: (m.toolCalls ?? []).map(call =>
                call.status === "pending" || call.status === "running"
                  ? { ...call, status: "cancelled" as const }
                  : call,
              ),
            }
          : m,
      ),
    }))
  }, [])

  // ------ clearChat ------
  const clearChat = useCallback(() => {
    abortRef.current?.abort()
    // Summarize the ending session before clearing
    if (state.sessionId) {
      summarizeSession(state.sessionId)
    }
    inFlightRef.current = false
    activeJobRef.current = null
    setState({
      messages: [],
      isStreaming: false,
      error: null,
      sessionId: null,
      sessionTitle: null,
      sessionTitleSource: null,
      activeJob: null,
    })
  }, [state.sessionId])

  // ------ loadSession ------
  const loadSession = useCallback(async (sessionId: string) => {
    const data = await fetchSession(sessionId)
    if (data && data.transcript.length > 0) {
      abortRef.current?.abort()
      inFlightRef.current = false
      activeJobRef.current = null
      setState({
        messages: data.transcript,
        isStreaming: false,
        error: null,
        sessionId,
        sessionTitle: data.title ?? deriveSessionTitleFromMessages(data.transcript),
        sessionTitleSource: data.title_source,
        activeJob: null,
      })
    }
  }, [])

  const applySessionTitle = useCallback((sessionId: string, title: string | null, source?: string | null) => {
    setState(prev => {
      if (prev.sessionId !== sessionId) return prev
      return {
        ...prev,
        sessionTitle: title,
        sessionTitleSource: source ?? prev.sessionTitleSource,
      }
    })
  }, [])

  return {
    messages: state.messages,
    isStreaming: state.isStreaming,
    error: state.error,
    sessionId: state.sessionId,
    sessionTitle: state.sessionTitle,
    sessionTitleSource: state.sessionTitleSource,
    sendMessage,
    stopStreaming,
    clearChat,
    loadSession,
    applySessionTitle,
  }
}
