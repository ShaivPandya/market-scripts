import type { ScreenContext } from "@/contexts/ScreenContext"
import type { AgentResponsePreferences } from "@/lib/api"

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
