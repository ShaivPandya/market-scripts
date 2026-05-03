import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface SurfaceCardProps {
  children: ReactNode
  className?: string
  muted?: boolean
}

export function SurfaceCard({ children, className, muted = false }: SurfaceCardProps) {
  return (
    <section className={cn(muted ? "theme-surface-muted" : "theme-surface", className)}>
      {children}
    </section>
  )
}
