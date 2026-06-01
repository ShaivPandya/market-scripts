import type { ActiveAgentJob, AgentMessage, QueuedAgentMessage } from "./agentChatShared"

export type { ActiveAgentJob, QueuedAgentMessage } from "./agentChatShared"

const JOBS_STORAGE_KEY = "agent-chat-jobs:v1"
const QUEUE_STORAGE_KEY = "agent-chat-queues:v1"
const SNAPSHOTS_STORAGE_KEY = "agent-chat-sessions:v1"

export interface SessionSnapshot {
  messages: AgentMessage[]
  sessionTitle: string | null
  sessionTitleSource: string | null
  error: string | null
}

function readJsonRecord<T>(key: string): Record<string, T> {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, T>)
      : {}
  } catch {
    return {}
  }
}

function writeJsonRecord<T>(key: string, value: Record<string, T>) {
  try {
    if (Object.keys(value).length === 0) {
      localStorage.removeItem(key)
      return
    }
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* ignore */
  }
}

export function readSessionSnapshots(): Record<string, SessionSnapshot> {
  return readJsonRecord<SessionSnapshot>(SNAPSHOTS_STORAGE_KEY)
}

export function writeSessionSnapshot(sessionId: string, snapshot: SessionSnapshot) {
  const all = readSessionSnapshots()
  all[sessionId] = snapshot
  writeJsonRecord(SNAPSHOTS_STORAGE_KEY, all)
}

export function readSessionSnapshot(sessionId: string): SessionSnapshot | null {
  return readSessionSnapshots()[sessionId] ?? null
}

export function readActiveJobs(): Record<string, ActiveAgentJob> {
  return readJsonRecord<ActiveAgentJob>(JOBS_STORAGE_KEY)
}

export function writeActiveJob(sessionId: string, job: ActiveAgentJob | null) {
  const all = readActiveJobs()
  if (job) all[sessionId] = job
  else delete all[sessionId]
  writeJsonRecord(JOBS_STORAGE_KEY, all)
}

export function readMessageQueues(): Record<string, QueuedAgentMessage[]> {
  return readJsonRecord<QueuedAgentMessage[]>(QUEUE_STORAGE_KEY)
}

export function writeMessageQueue(sessionId: string, queue: QueuedAgentMessage[]) {
  const all = readMessageQueues()
  if (queue.length === 0) delete all[sessionId]
  else all[sessionId] = queue
  writeJsonRecord(QUEUE_STORAGE_KEY, all)
}

export function readMessageQueue(sessionId: string): QueuedAgentMessage[] {
  return readMessageQueues()[sessionId] ?? []
}

export function combineQueuedPrompt(entries: QueuedAgentMessage[]): string {
  const parts = entries
    .map(entry => entry.content.trim())
    .filter(Boolean)
  return parts.join("\n\n")
}

export function shouldCombineQueueEntries(entries: QueuedAgentMessage[]): boolean {
  if (entries.length <= 1) return false
  return entries.every(entry => !entry.options?.durable && !entry.content.trim().startsWith("/workflow:"))
}

export interface ActiveAgentJobApiRow {
  job_id: string
  status: string
  client_turn_id?: string | null
}
