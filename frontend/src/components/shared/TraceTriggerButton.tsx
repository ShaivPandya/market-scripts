import { GitBranch } from "lucide-react"

import { cn } from "@/lib/utils"

interface TraceTriggerButtonProps {
  onClick: () => void
  label?: string
  compact?: boolean
  className?: string
  disabled?: boolean
}

export function TraceTriggerButton({
  onClick,
  label = "Trace",
  compact = false,
  className,
  disabled = false,
}: TraceTriggerButtonProps) {
  if (compact) {
    return (
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className={cn("theme-icon-button h-8 w-8 shrink-0 disabled:opacity-50", className)}
        aria-label={label}
        title={label}
      >
        <GitBranch size={14} />
      </button>
    )
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border border-app px-2.5 py-1 text-xs font-medium text-muted hover:text-app disabled:opacity-50",
        className,
      )}
    >
      <GitBranch size={12} />
      {label}
    </button>
  )
}
