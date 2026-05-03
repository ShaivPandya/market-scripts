import { memo } from "react"
import { cn } from "@/lib/utils"
import { StatusBadge, type StatusTone } from "./StatusBadge"
import { SurfaceCard } from "./SurfaceCard"

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  signal?: "success" | "warning" | "error" | "info" | null
  signalLabel?: string
  className?: string
}

export const MetricCard = memo(function MetricCard({
  title,
  value,
  subtitle,
  signal,
  signalLabel,
  className,
}: MetricCardProps) {
  const signalToneMap: Record<NonNullable<MetricCardProps["signal"]>, StatusTone> = {
    success: "success",
    warning: "warning",
    error: "error",
    info: "info",
  }

  return (
    <SurfaceCard className={cn("p-4 sm:p-5", className)}>
      <p className="truncate text-sm font-medium text-muted">{title}</p>
      <p className="mt-2 text-[1.8rem] font-semibold tracking-[-0.03em] text-app">{value}</p>
      {subtitle && (
        <p className="mt-1 text-xs leading-5 text-subtle">{subtitle}</p>
      )}
      {signal && signalLabel && (
        <StatusBadge tone={signalToneMap[signal]} className="mt-3">
          {signalLabel}
        </StatusBadge>
      )}
    </SurfaceCard>
  )
})
