import { NavLink, useNavigate } from "react-router-dom"
import { Moon, Sun } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuth } from "@/contexts/AuthContext"
import { useTheme } from "@/contexts/ThemeContext"

interface NavSection {
  label?: string
  pages: { label: string; path: string }[]
}

const NAV_SECTIONS: NavSection[] = [
  {
    label: "Core",
    pages: [
      { label: "Portfolio Dashboard", path: "/" },
      { label: "Workspace", path: "/workspace" },
      { label: "Investment Theses", path: "/theses" },
      { label: "Weekly Report", path: "/weekly-report" },
    ],
  },
  {
    label: "Labs",
    pages: [
      { label: "Portfolio Analyzer", path: "/analyzer" },
      { label: "Portfolio Sizer", path: "/sizer" },
      { label: "Hedging Tool", path: "/hedging-tool" },
      { label: "Chart", path: "/chart" },
      { label: "Screeners", path: "/screeners" },
      { label: "Financials", path: "/financials" },
      { label: "DCF Model", path: "/dcf-model" },
      { label: "FX Model", path: "/fx-model" },
      { label: "Momentum", path: "/momentum" },
    ],
  },
  {
    label: "Monitors",
    pages: [
      { label: "Signal Aggregator", path: "/signal-aggregator" },
      { label: "Ontology Workbench", path: "/ontology" },
      { label: "Market Technicals", path: "/market-technicals" },
      { label: "News Digests", path: "/portfolio-news" },
      { label: "Breakout", path: "/breakout" },
      { label: "Sentiment", path: "/sentiment" },
      { label: "Positioning", path: "/positioning" },
      { label: "Central Bank Monitor", path: "/central-banks" },
      { label: "Industry Monitor", path: "/industry-monitor" },
      { label: "Sector Metrics", path: "/sector-metrics" },
    ],
  },
  {
    label: "Macro",
    pages: [
      { label: "Economic Growth", path: "/economic-growth" },
      { label: "Labor Market", path: "/labor-market" },
      { label: "Housing", path: "/housing" },
      { label: "Liquidity", path: "/liquidity" },
      { label: "Yield Curve", path: "/yield-curve" },
      { label: "Bond Dashboard", path: "/bond-dashboard" },
      { label: "Country Dashboard", path: "/country-dashboard" },
    ],
  },
  {
    label: "Assets",
    pages: [
      { label: "Index Dashboard", path: "/index-dashboard" },
      { label: "FX Dashboard", path: "/fx-dashboard" },
      { label: "Commodity Dashboard", path: "/commodities" },
      { label: "Commodities Curve", path: "/commodities-curve" },
      { label: "Commodity Proxy Screener", path: "/commodity-research" },
    ],
  },
  {
    label: "Settings",
    pages: [
      { label: "AI Settings", path: "/settings/ai" },
    ],
  },
]

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
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
        "md:w-56 md:shrink-0 md:static md:translate-x-0 md:h-screen md:sticky md:top-0",
        "fixed top-0 left-0 h-full w-64 z-30",
        "transition-transform duration-300",
        isOpen ? "translate-x-0" : "-translate-x-full",
        "theme-sidebar border-r border-strong flex flex-col",
      ].join(" ")}
    >
      <div className="px-3 py-4 flex-1 overflow-y-auto">
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {section.label && (
              <p className="mt-3 mb-1 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
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
                    "theme-sidebar-link block w-full rounded-lg px-2.5 py-1.5 text-left text-sm mb-0.5 truncate",
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

      <div className="border-t border-app px-3 py-3">
        <div className="flex items-center gap-2">
          <button
            onClick={handleLogout}
            className="flex-1 rounded-lg px-2 py-1.5 text-left text-sm text-muted transition-colors hover:bg-[hsl(var(--muted-2))] hover:text-app"
          >
            Sign out
          </button>
          <button
            type="button"
            role="switch"
            aria-checked={resolvedTheme === "dark"}
            aria-label="Toggle dark mode"
            onClick={toggleTheme}
            className="inline-flex shrink-0 items-center gap-2 rounded-full border border-app bg-muted-surface px-2 py-1 transition-colors hover:bg-[hsl(var(--muted-2))]"
          >
            <Sun
              size={14}
              className={resolvedTheme === "dark" ? "text-subtle" : "text-amber-500"}
              aria-hidden="true"
            />
            <span
              className="relative inline-flex h-[18px] w-[32px] rounded-full transition-colors duration-200"
              style={{
                backgroundColor: resolvedTheme === "dark"
                  ? "hsl(var(--accent))"
                  : "hsl(var(--muted-3))",
              }}
            >
              <span
                className={cn(
                  "mt-[2px] inline-block h-[14px] w-[14px] rounded-full bg-card shadow-sm transition-transform duration-200",
                  resolvedTheme === "dark" ? "translate-x-[16px]" : "translate-x-[2px]",
                )}
              />
            </span>
            <Moon
              size={14}
              className={resolvedTheme === "dark" ? "text-blue-400" : "text-subtle"}
              aria-hidden="true"
            />
          </button>
        </div>
      </div>
    </nav>
  )
}
