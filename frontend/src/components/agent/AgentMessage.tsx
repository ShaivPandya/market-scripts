import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { ToolCall, AgentMessage as AgentMessageType } from "@/hooks/useAgentChat"
import { Loader2, CheckCircle2, AlertCircle } from "lucide-react"

// ---------------------------------------------------------------------------
// Friendly tool labels
// ---------------------------------------------------------------------------

const TOOL_LABELS: Record<string, string> = {
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
  get_breakout: "Breakout Signals",
}

function ToolCallChip({ tc }: { tc: ToolCall }) {
  const label = TOOL_LABELS[tc.name] ?? tc.name
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-muted-surface border border-app px-2 py-0.5 text-xs text-muted mr-1.5 mb-1.5">
      {tc.status === "pending" && <Loader2 size={10} className="animate-spin text-blue-500" />}
      {tc.status === "ok" && <CheckCircle2 size={10} className="text-green-500" />}
      {tc.status === "error" && <AlertCircle size={10} className="text-red-500" />}
      {tc.status === "pending" ? `Fetching ${label}...` : label}
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

  // Assistant message
  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[85%] rounded-2xl bg-card border border-app px-4 py-2.5 text-sm text-app leading-relaxed">
        {/* Tool call indicators */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-wrap mb-2">
            {message.toolCalls.map(tc => (
              <ToolCallChip key={tc.id} tc={tc} />
            ))}
          </div>
        )}

        {/* Markdown content */}
        {message.content && (
          <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-1.5 prose-headings:mt-3 prose-headings:mb-1.5 prose-ul:my-1.5 prose-li:my-0.5 prose-table:my-2 prose-pre:my-2 prose-pre:bg-muted-surface prose-pre:border prose-pre:border-app">
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </div>
        )}

        {/* Streaming cursor */}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle rounded-sm" />
        )}
      </div>
    </div>
  )
}
