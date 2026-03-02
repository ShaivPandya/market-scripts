import { NavLink, useNavigate } from "react-router-dom"
import { cn } from "@/lib/utils"
import { useAuth } from "@/contexts/AuthContext"

interface NavSection {
  pages: { label: string; path: string }[]
}

const NAV_SECTIONS: NavSection[] = [
  {
    pages: [
      { label: "💼 Portfolio Dashboard", path: "/portfolio" },
      { label: "📈 Portfolio Optimizer", path: "/optimizer" },
      { label: "🚀 Momentum", path: "/momentum" },
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
      { label: "🔔 Breakout", path: "/breakout" },
      { label: "💱 FX Model", path: "/fx-model" },
    ],
  },
  {
    pages: [
      { label: "📊 Economic Growth", path: "/economic-growth" },
      { label: "💧 Liquidity", path: "/liquidity" },
      { label: "📉 Yield Curve", path: "/yield-curve" },
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
        "border-r border-gray-200 bg-gray-50 flex flex-col",
      ].join(" ")}
    >
      <div className="px-3 py-4 flex-1 overflow-y-auto">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
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
                    "block w-full text-left px-2 py-1.5 rounded text-sm mb-0.5 transition-colors truncate",
                    isActive
                      ? "bg-blue-600 text-white font-medium"
                      : "text-gray-700 hover:bg-gray-200",
                  )
                }
              >
                {page.label}
              </NavLink>
            ))}
            {si < NAV_SECTIONS.length - 1 && (
              <hr className="my-2 border-gray-200" />
            )}
          </div>
        ))}
      </div>

      <div className="px-3 py-3 border-t border-gray-200">
        <button
          onClick={handleLogout}
          className="w-full text-left px-2 py-1.5 rounded text-sm text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors"
        >
          Sign out
        </button>
      </div>
    </nav>
  )
}
