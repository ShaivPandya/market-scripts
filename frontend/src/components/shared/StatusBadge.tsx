import { AlertTriangle, CheckCircle2, Dot, Info, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"

export type StatusTone = "neutral" | "info" | "success" | "warning" | "error"

const ICONS = {
  neutral: Dot,
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  error: XCircle,
} satisfies Record<StatusTone, typeof Dot>

const TONE_CLASSES: Record<StatusTone, string> = {
  neutral: "theme-badge-neutral",
  info: "theme-badge-info",
  success: "theme-badge-success",
  warning: "theme-badge-warning",
  error: "theme-badge-error",
}

interface StatusBadgeProps {
  tone?: StatusTone
  children: React.ReactNode
  className?: string
  tooltip?: string
}

export function StatusBadge({ tone = "neutral", children, className, tooltip }: StatusBadgeProps) {
  const Icon = ICONS[tone]
  return (
    <span
      className={cn("theme-badge", TONE_CLASSES[tone], tooltip && "theme-tooltip", className)}
      data-tooltip={tooltip}
    >
      <Icon size={12} aria-hidden="true" />
      {children}
    </span>
  )
}
