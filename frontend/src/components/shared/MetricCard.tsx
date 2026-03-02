import { cn } from "@/lib/utils"

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  signal?: "success" | "warning" | "error" | "info" | null
  signalLabel?: string
  className?: string
}

export function MetricCard({
  title,
  value,
  subtitle,
  signal,
  signalLabel,
  className,
}: MetricCardProps) {
  const signalColors = {
    success: "bg-green-50 text-green-800 border-green-200",
    warning: "bg-yellow-50 text-yellow-800 border-yellow-200",
    error: "bg-red-50 text-red-800 border-red-200",
    info: "bg-blue-50 text-blue-800 border-blue-200",
  }

  return (
    <div className={cn("rounded-xl border bg-white p-4 shadow-sm", className)}>
      <p className="text-sm text-gray-500 font-medium truncate">{title}</p>
      <p className="text-2xl font-bold mt-1 text-gray-900">{value}</p>
      {subtitle && (
        <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
      )}
      {signal && signalLabel && (
        <div
          className={cn(
            "mt-2 text-xs px-2 py-1 rounded-md border font-medium inline-block",
            signalColors[signal],
          )}
        >
          {signalLabel}
        </div>
      )}
    </div>
  )
}
