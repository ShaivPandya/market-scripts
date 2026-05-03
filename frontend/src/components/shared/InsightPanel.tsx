import type { ReactNode } from "react"
import { ChevronDown, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"
import { SurfaceCard } from "./SurfaceCard"

interface InsightPanelProps {
  open: boolean
  onToggle: () => void
  title?: string
  body?: ReactNode
  loading?: boolean
  loadingText?: string
  errorText?: string | null
}

export function InsightPanel({
  open,
  onToggle,
  title = "AI Overview",
  body,
  loading,
  loadingText,
  errorText,
}: InsightPanelProps) {
  if (!body && !loading && !errorText) return null

  return (
    <SurfaceCard className="overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 rounded-[inherit] bg-[hsl(var(--accent-muted))] px-4 py-3 text-left transition-colors hover:bg-[hsl(var(--selected))]"
      >
        <span className="flex items-center gap-2">
          <Sparkles size={15} className="text-link" aria-hidden="true" />
          <span className="text-sm font-semibold text-link">{title}</span>
        </span>
        <ChevronDown
          size={16}
          className={cn("text-link transition-transform duration-200", open && "rotate-180")}
          aria-hidden="true"
        />
      </button>

      {open ? (
        <div className="space-y-3 px-4 py-4">
          {loading ? (
            <div className="flex items-center gap-2 text-sm text-muted">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-app border-t-[hsl(var(--accent))]" />
              {loadingText ?? "Analyzing..."}
            </div>
          ) : null}
          {errorText ? <p className="body-copy text-negative">{errorText}</p> : null}
          {body ? <div className="body-copy whitespace-pre-wrap">{body}</div> : null}
        </div>
      ) : null}
    </SurfaceCard>
  )
}
