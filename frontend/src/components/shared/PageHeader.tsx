import type { ReactNode } from "react"

interface PageHeaderProps {
  title: string
  subtitle?: string
  eyebrow?: string
  actions?: ReactNode
}

export function PageHeader({ title, subtitle, eyebrow, actions }: PageHeaderProps) {
  return (
    <header className="theme-page-header">
      <div>
        {eyebrow && <p className="theme-eyebrow">{eyebrow}</p>}
        <h1 className="theme-page-title">{title}</h1>
        {subtitle && <p className="theme-page-subtitle">{subtitle}</p>}
      </div>
      {actions ? <div className="theme-toolbar">{actions}</div> : null}
    </header>
  )
}
