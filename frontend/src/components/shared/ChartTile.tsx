import type { ReactNode } from "react"
import { Link } from "react-router-dom"
import { SurfaceCard } from "./SurfaceCard"

interface ChartTileProps {
  title: string
  subtitle?: ReactNode
  href?: string
  meta?: ReactNode
  children: ReactNode
}

export function ChartTile({ title, subtitle, href, meta, children }: ChartTileProps) {
  const titleNode = href ? (
    <Link to={href} className="card-title transition-colors hover:text-link">
      {title}
    </Link>
  ) : (
    <h2 className="card-title">{title}</h2>
  )

  return (
    <SurfaceCard className="p-4 sm:p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          {titleNode}
          {subtitle ? <div className="mt-1 text-sm text-muted">{subtitle}</div> : null}
        </div>
        {meta ? <div className="shrink-0">{meta}</div> : null}
      </div>
      {children}
    </SurfaceCard>
  )
}
