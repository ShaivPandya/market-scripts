import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { ToolCall, AgentMessage as AgentMessageType } from "@/hooks/useAgentChat"
import { Loader2, CheckCircle2, AlertCircle, Ban, Clock, GitPullRequest, RotateCw } from "lucide-react"
import { DecisionStateBadge, EffectScopeBadge } from "@/components/shared/DecisionStateBadge"

// ---------------------------------------------------------------------------
// Friendly tool labels
// ---------------------------------------------------------------------------

const TOOL_LABELS: Record<string, string> = {
  get_thesis: "Thesis",
  get_thesis_evaluations: "Thesis Evaluations",
  get_liquidity: "Liquidity",
  get_market_breadth: "Market Breadth",
  get_vix_term_structure: "VIX Term Structure",
  get_positioning: "Positioning",
  get_economic_growth: "Economic Growth",
  get_labor_market: "Labor Market",
  get_sector_metrics: "Sector Metrics",
  get_portfolio: "Portfolio",
  get_yield_curve: "Yield Curve",
  get_sentiment: "Sentiment",
  get_central_banks: "Central Banks",
  get_industry_monitor: "Industry Monitor",
  get_signal_aggregator: "Signal Aggregator",
  query_ontology: "Ontology Query",
  search_web: "Web Search",
  search_knowledge_base: "Knowledge Base",
  propose_thesis_status_change: "Thesis Status Proposal",
  propose_action_item: "Action Item Proposal",
  propose_portfolio_positions_update: "Portfolio Update Proposal",
  propose_hedge_positions_update: "Hedge Update Proposal",
  propose_thesis_content_update: "Thesis Content Proposal",
}

const PROPOSAL_TOOL_PREFIXES = ["propose_", "create_", "update_", "complete_", "dismiss_", "cancel_", "fire_"]

function toolEffectScope(name: string): "read_only" | "internal_state" {
  return PROPOSAL_TOOL_PREFIXES.some(prefix => name.startsWith(prefix)) ? "internal_state" : "read_only"
}

function toolStateLabel(status: ToolCall["status"]): string {
  switch (status) {
    case "pending":
      return "Pending"
    case "running":
      return "Running"
    case "ok":
      return "Complete"
    case "blocked":
      return "Blocked"
    case "timeout":
      return "Timed out"
    case "retrying":
      return "Retrying"
    case "partial":
      return "Partial"
    case "cancelled":
      return "Cancelled"
    case "error":
    default:
      return "Failed"
  }
}

const ARTIFACT_JSON_KEYS = [
  "evaluation_draft",
  "action_items",
  "watch_triggers",
]

function containsArtifactJsonKey(value: string): boolean {
  return ARTIFACT_JSON_KEYS.some(key => value.includes(`"${key}"`))
}

function stripTrailingArtifactJson(value: string): string {
  const trimmedStart = value.trimStart()
  if (trimmedStart.startsWith("{") && containsArtifactJsonKey(trimmedStart)) {
    return value.slice(0, value.length - trimmedStart.length).trimEnd()
  }

  const candidates = [...value.matchAll(/\n\s*\{/g)].reverse()
  for (const match of candidates) {
    const start = match.index ?? -1
    if (start < 0) continue
    const suffix = value.slice(start)
    if (!containsArtifactJsonKey(suffix)) continue
    return value.slice(0, start).trimEnd()
  }
  return value
}

function stripArtifactBlocks(value: string): string {
  const withoutClosedBlocks = value.replace(
    /(^|\n)```([^\n]*)\n([\s\S]*?)```[ \t]*(?=\n|$)/g,
    (match, prefix: string, info: string, body: string) => {
      const isArtifactBlock = info.trim().toLowerCase() === "artifacts" || containsArtifactJsonKey(body)
      return isArtifactBlock ? prefix : match
    },
  )

  const lastFence = withoutClosedBlocks.lastIndexOf("```")
  if (lastFence >= 0) {
    const trailingBlock = withoutClosedBlocks.slice(lastFence)
    const firstLine = trailingBlock.split("\n", 1)[0]?.replace(/^```/, "").trim().toLowerCase()
    if (firstLine === "artifacts" || containsArtifactJsonKey(trailingBlock)) {
      return withoutClosedBlocks.slice(0, lastFence).trimEnd()
    }
  }

  return stripTrailingArtifactJson(withoutClosedBlocks)
}

function ToolCallChip({ tc }: { tc: ToolCall }) {
  const label = TOOL_LABELS[tc.name] ?? tc.name
  const scope = toolEffectScope(tc.name)
  const isBusy = tc.status === "pending" || tc.status === "running" || tc.status === "retrying"
  const isFailure = tc.status === "error" || tc.status === "blocked" || tc.status === "timeout" || tc.status === "cancelled"
  const Icon = isBusy
    ? Loader2
    : tc.status === "ok"
      ? CheckCircle2
      : tc.status === "retrying"
        ? RotateCw
        : tc.status === "blocked"
          ? Ban
          : tc.status === "timeout"
            ? Clock
            : AlertCircle
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs mr-1.5 mb-1.5 ${
        isFailure
          ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
          : scope === "internal_state"
            ? "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200"
            : "bg-muted-surface border-app text-muted"
      }`}
      title={tc.message || `${label}: ${toolStateLabel(tc.status)}`}
    >
      {scope === "internal_state" && <GitPullRequest size={10} aria-hidden="true" />}
      <Icon size={10} className={isBusy ? "animate-spin text-blue-500" : ""} aria-hidden="true" />
      {label}
      <span className="opacity-75">· {toolStateLabel(tc.status)}</span>
      {tc.policyDecisionId && <span className="opacity-75">· policy {tc.policyDecisionId}</span>}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Message component
// ---------------------------------------------------------------------------

export function AgentMessage({ message }: { message: AgentMessageType }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[85%] rounded-2xl bg-blue-600 text-white px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  const displayContent = stripArtifactBlocks(message.content)

  // Assistant message
  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[85%] rounded-2xl bg-card border border-app px-4 py-2.5 text-sm text-app leading-relaxed">
        <div className="mb-2 flex flex-wrap gap-2">
          <DecisionStateBadge state="analysis" />
          <EffectScopeBadge scope="read_only" />
        </div>
        {/* Tool call indicators */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-wrap mb-2">
            {message.toolCalls.map(tc => (
              <ToolCallChip key={tc.id} tc={tc} />
            ))}
          </div>
        )}

        {/* Markdown content */}
        {displayContent && message.isStreaming && (
          <div className="whitespace-pre-wrap">
            {displayContent}
          </div>
        )}
        {displayContent && !message.isStreaming && (
          <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-1.5 prose-headings:mt-3 prose-headings:mb-1.5 prose-ul:my-1.5 prose-li:my-0.5 prose-table:my-2 prose-pre:my-2 prose-pre:bg-muted-surface prose-pre:border prose-pre:border-app">
            <Markdown remarkPlugins={[remarkGfm]}>{displayContent}</Markdown>
          </div>
        )}

        {!displayContent && message.statusText && (
          <span className="text-xs text-muted">{message.statusText}</span>
        )}

        {/* Streaming cursor */}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle rounded-sm" />
        )}
      </div>
    </div>
  )
}
