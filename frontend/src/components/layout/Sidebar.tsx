/* eslint-disable react-refresh/only-export-components */
import { NavLink, useNavigate } from "react-router-dom"
import { Moon, Search, Sun } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuth } from "@/contexts/AuthContext"
import { useTheme } from "@/contexts/ThemeContext"

interface NavSection {
  label?: string
  pages: { label: string; path: string }[]
}

export const NAV_SECTIONS: NavSection[] = [
  {
    label: "Command Portfolio",
    pages: [
      { label: "Portfolio Dashboard", path: "/" },
      { label: "Portfolio Commander", path: "/workspace" },
      { label: "Investment Theses", path: "/theses" },
    ],
  },
  {
    label: "Review Decisions",
    pages: [
      { label: "Idea Watchlist", path: "/ideas" },
    ],
  },
  {
    label: "Pressure-Test Positions",
    pages: [
      { label: "Portfolio Analyzer", path: "/analyzer" },
      { label: "Portfolio Sizer", path: "/sizer" },
      { label: "Hedging Tool", path: "/hedging-tool" },
      { label: "Chart", path: "/chart" },
      { label: "Financials", path: "/financials" },
      { label: "DCF Model", path: "/dcf-model" },
      { label: "FX Model", path: "/fx-model" },
      { label: "Momentum", path: "/momentum" },
    ],
  },
  {
    label: "Scout Opportunities",
    pages: [
      { label: "Screeners", path: "/screeners" },
      { label: "Commodity Proxy Screener", path: "/commodity-research" },
    ],
  },
  {
    label: "Monitor Risks",
    pages: [
      { label: "Signal Aggregator", path: "/signal-aggregator" },
      { label: "Market Technicals", path: "/market-technicals" },
      { label: "News Digests", path: "/portfolio-news" },
      { label: "Sentiment", path: "/sentiment" },
      { label: "Positioning", path: "/positioning" },
      { label: "Central Bank Monitor", path: "/central-banks" },
      { label: "Industry Monitor", path: "/industry-monitor" },
      { label: "Sector Metrics", path: "/sector-metrics" },
    ],
  },
  {
    label: "Inspect Data & Provenance",
    pages: [
      { label: "Ontology Workbench", path: "/ontology" },
      { label: "Economic Growth", path: "/economic-growth" },
      { label: "Labor Market", path: "/labor-market" },
      { label: "Housing", path: "/housing" },
      { label: "Liquidity", path: "/liquidity" },
      { label: "Yield Curve", path: "/yield-curve" },
      { label: "Bond Dashboard", path: "/bond-dashboard" },
      { label: "Country Dashboard", path: "/country-dashboard" },
      { label: "Index Dashboard", path: "/index-dashboard" },
      { label: "FX Dashboard", path: "/fx-dashboard" },
      { label: "Commodity Dashboard", path: "/commodities" },
      { label: "Commodities Curve", path: "/commodities-curve" },
    ],
  },
  {
    label: "Administer",
    pages: [
      { label: "AI Settings", path: "/settings/ai" },
      { label: "Policy Matrix", path: "/settings/policy-matrix" },
    ],
  },
]

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onOpenSearch: () => void
}

export function getRouteLabel(pathname: string) {
  for (const section of NAV_SECTIONS) {
    const match = section.pages.find(page => page.path === pathname)
    if (match) return match.label
  }
  if (pathname.startsWith("/dossier/")) return "Position Dossier"
  if (pathname.startsWith("/ideas/") && pathname !== "/ideas") return "Idea Detail"
  return "Talisman"
}

export function Sidebar({ isOpen, onClose, onOpenSearch }: SidebarProps) {
  const { logout, mode } = useAuth()
  const { resolvedTheme, toggleTheme } = useTheme()
  const navigate = useNavigate()

  async function handleLogout() {
    if (mode === "cloudflare") {
      await logout()
      return
    }
    try {
      await logout()
    } finally {
      navigate("/login", { replace: true })
    }
  }

  return (
    <nav
      className={[
        "md:w-[17.5rem] md:shrink-0 md:static md:translate-x-0 md:h-screen md:sticky md:top-0",
        "fixed top-0 left-0 z-30 h-full w-[min(21rem,calc(100vw-1rem))]",
        "transition-transform duration-300",
        isOpen ? "translate-x-0" : "-translate-x-full",
        "theme-sidebar border-r border-strong flex flex-col",
      ].join(" ")}
      aria-label="Primary navigation"
    >
      <div className="border-b border-app px-4 pb-4 pt-[max(1rem,var(--safe-top))]">
        <p className="theme-eyebrow mb-2">Operating</p>
        <p className="text-lg font-semibold tracking-[-0.03em] text-app">Talisman</p>
        <button
          type="button"
          onClick={onOpenSearch}
          aria-label="Search workflows"
          aria-keyshortcuts="Meta+J"
          className="mt-4 flex h-11 w-full items-center gap-2 rounded-[var(--radius-md)] border border-app bg-input px-3 text-left text-sm text-muted transition-colors hover:border-strong hover:bg-hover hover:text-app"
        >
          <Search size={15} className="shrink-0 text-subtle" aria-hidden="true" />
          <span className="min-w-0 flex-1 truncate">Search workflows</span>
          <kbd className="shrink-0 rounded-md border border-app bg-card-muted px-1.5 py-0.5 text-[10px] font-medium leading-none text-subtle">
            Cmd J
          </kbd>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-4">
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {section.label && (
              <p className="mb-1 mt-4 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-subtle">
                {section.label}
              </p>
            )}
            {section.pages.map(page => (
              <NavLink
                key={page.path}
                to={page.path}
                end={page.path === "/"}
                onClick={onClose}
                className={({ isActive }) =>
                  cn(
                    "theme-sidebar-link mb-1 flex w-full truncate text-left text-sm",
                    isActive && "theme-sidebar-link-active font-medium",
                  )
                }
              >
                {page.label}
              </NavLink>
            ))}
            {si < NAV_SECTIONS.length - 1 && (
              <hr className="my-2 border-app" />
            )}
          </div>
        ))}
      </div>

      <div className="border-t border-app px-3 py-3 pb-[max(0.75rem,var(--safe-bottom))]">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleLogout}
            className="theme-button-base theme-button-secondary flex-1 justify-start px-4"
          >
            Sign out
          </button>
          <button
            type="button"
            role="switch"
            aria-checked={resolvedTheme === "dark"}
            aria-label="Toggle dark mode"
            onClick={toggleTheme}
            className="inline-flex h-11 shrink-0 items-center gap-2 rounded-full border border-app bg-card-muted px-3 transition-colors hover:bg-hover"
          >
            <Sun
              size={14}
              className={resolvedTheme === "dark" ? "text-subtle" : "text-link"}
              aria-hidden="true"
            />
            <span
              className="relative inline-flex h-5 w-9 rounded-full transition-colors duration-200"
              style={{
                backgroundColor: resolvedTheme === "dark"
                  ? "hsl(var(--accent))"
                  : "hsl(var(--border-strong))",
              }}
            >
              <span
                className={cn(
                  "mt-[2px] inline-block h-4 w-4 rounded-full bg-elevated shadow-sm transition-transform duration-200",
                  resolvedTheme === "dark" ? "translate-x-[18px]" : "translate-x-[2px]",
                )}
              />
            </span>
            <Moon
              size={14}
              className={resolvedTheme === "dark" ? "text-link" : "text-subtle"}
              aria-hidden="true"
            />
          </button>
        </div>
      </div>
    </nav>
  )
}
