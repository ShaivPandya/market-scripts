import { NavLink, useNavigate } from "react-router-dom"
import { cn } from "@/lib/utils"
import { useAuth } from "@/contexts/AuthContext"
import { useTheme } from "@/contexts/ThemeContext"
import { Toggle } from "@/components/shared/FormControls"

interface NavSection {
  pages: { label: string; path: string }[]
}

const NAV_SECTIONS: NavSection[] = [
  {
    pages: [
      { label: "💼 Portfolio Dashboard", path: "/portfolio" },
      { label: "📈 Portfolio Analyzer", path: "/analyzer" },
      { label: "🎯 Portfolio Sizer", path: "/sizer" },
      { label: "🛡️ Hedging Tool", path: "/hedging-tool" },
      { label: "🚀 Momentum", path: "/momentum" },
      { label: "📰 Portfolio News", path: "/portfolio-news" },
      { label: "📅 Weekly Report", path: "/weekly-report" },
    ],
  },
  {
    pages: [
      { label: "📐 Chart", path: "/chart" },
      { label: "🏅 Quality Screen", path: "/quality" },
      { label: "📉 Short Screen", path: "/short-screen" },
      { label: "📈 Fundamental Momentum", path: "/fundamental-momentum" },
      { label: "🧾 Financials", path: "/financials" },
    ],
  },
  {
    pages: [
      { label: "📊 Index Dashboard", path: "/index-dashboard" },
      { label: "💵 FX Dashboard", path: "/fx-dashboard" },
      { label: "🛢️ Commodity Dashboard", path: "/commodities" },
    ],
  },
  {
    pages: [
      { label: "🧭 Market Technicals", path: "/market-technicals" },
      { label: "🧩 Sector Metrics", path: "/sector-metrics" },
      { label: "📌 Positioning", path: "/positioning" },
      { label: "🌡️ Sentiment", path: "/sentiment" },
      { label: "🔔 Breakout", path: "/breakout" },
      { label: "💱 FX Model", path: "/fx-model" },
    ],
  },
  {
    pages: [
      { label: "📊 Economic Growth", path: "/economic-growth" },
      { label: "💧 Liquidity", path: "/liquidity" },
      { label: "〰️ Yield Curve", path: "/yield-curve" },
      { label: "📈 Commodities Curve", path: "/commodities-curve" },
      { label: "🌍 Country Dashboard", path: "/country-dashboard" },
    ],
  },
  {
    pages: [
      { label: "🏦 Central Bank Monitor", path: "/central-banks" },
      { label: "🏭 Industry Monitor", path: "/industry-monitor" },
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
        "border-r border-app bg-muted-surface flex flex-col",
      ].join(" ")}
    >
      <div className="px-3 py-4 flex-1 overflow-y-auto">
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-subtle">
          Navigation
        </p>
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {section.pages.map(page => (
              <NavLink
                key={page.path}
                to={page.path}
                onClick={onClose}
                className={({ isActive }) =>
                  cn(
                    "block w-full text-left px-2 py-1.5 rounded-lg text-sm mb-0.5 transition-colors truncate",
                    isActive
                      ? "bg-blue-50 text-blue-600 font-medium"
                      : "text-gray-700 hover:bg-gray-100",
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

      <div className="space-y-3 border-t border-app px-3 py-3">
        <div className="rounded-xl border border-app bg-card px-3 py-2.5">
          <Toggle
            label="Dark mode"
            checked={resolvedTheme === "dark"}
            onChange={toggleTheme}
            description="Swap surfaces, controls, and charts."
          />
        </div>
        <button
          onClick={handleLogout}
          className="w-full rounded-lg px-2 py-1.5 text-left text-sm text-muted transition-colors hover:bg-[hsl(var(--muted-2))] hover:text-app"
        >
          Sign out
        </button>
      </div>
    </nav>
  )
}
