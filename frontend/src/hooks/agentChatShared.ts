import type { ScreenContext } from "@/contexts/ScreenContext"
import type { AgentResponseFeedbackRecord, AgentResponsePreferences } from "@/lib/api"

import type { AgentTraceSnapshot } from "@/lib/decisionTrace"

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
  traceSnapshot?: AgentTraceSnapshot | null
  feedback?: AgentResponseFeedbackRecord | null
}

export interface ActiveAgentJob {
  jobId: string
  assistantId: string
  afterSeq: number
  clientTurnId: string
  sessionId: string
}

export type AgentMessageDelivery = "enqueue" | "immediate"

export interface AgentSendOptions {
  durable?: boolean
  mode?: AgentMessageDelivery
}

export interface QueuedAgentMessage {
  id: string
  content: string
  createdAt: number
  screenContext?: ScreenContext | null
  responsePreferences?: AgentResponsePreferences | null
  options?: AgentSendOptions
}

const ACTIVE_TOOL_STATUSES: ReadonlySet<ToolCall["status"]> = new Set([
  "pending",
  "running",
  "retrying",
])

/** True when an assistant turn is still in flight (streaming text or active tools). */
export function assistantTurnInProgress(messages: AgentMessage[]): boolean {
  for (const message of messages) {
    if (message.role !== "assistant") continue
    if (message.isStreaming || message.statusText) return true
    if (message.toolCalls?.some(tool => ACTIVE_TOOL_STATUSES.has(tool.status))) return true
  }
  return false
}
