import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"

type NoticeTone = "info" | "success" | "warning" | "error"

const TONE_CONFIG = {
  info: { className: "theme-notice-info", Icon: Info },
  success: { className: "theme-notice-success", Icon: CheckCircle2 },
  warning: { className: "theme-notice-warning", Icon: AlertTriangle },
  error: { className: "theme-notice-error", Icon: XCircle },
} satisfies Record<NoticeTone, { className: string; Icon: typeof Info }>

interface NoticeProps {
  tone?: NoticeTone
  children: React.ReactNode
  className?: string
}

export function Notice({ tone = "info", children, className }: NoticeProps) {
  const { className: toneClassName, Icon } = TONE_CONFIG[tone]

  return (
    <div className={cn("theme-notice", toneClassName, className)}>
      <Icon size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}
