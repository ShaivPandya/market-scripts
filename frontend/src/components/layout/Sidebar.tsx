import { NavLink } from "react-router-dom"
import { cn } from "@/lib/utils"

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
    ],
  },
  {
    pages: [
      { label: "📊 Index Dashboard", path: "/index-dashboard" },
      { label: "📉 FX Dashboard", path: "/fx-dashboard" },
      { label: "🛢️ Commodity Dashboard", path: "/commodities" },
    ],
  },
  {
    pages: [
      { label: "📈 Market Technicals", path: "/market-technicals" },
      { label: "🏛️ Sector Metrics", path: "/sector-metrics" },
      { label: "📌 Positioning", path: "/positioning" },
      { label: "🔔 Breakout", path: "/breakout" },
      { label: "💱 FX Model", path: "/fx-model" },
    ],
  },
  {
    pages: [
      { label: "📊 Economic Growth", path: "/economic-growth" },
      { label: "💧 Liquidity", path: "/liquidity" },
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

export function Sidebar() {
  return (
    <nav className="w-56 shrink-0 border-r border-gray-200 bg-gray-50 h-screen overflow-y-auto sticky top-0">
      <div className="px-3 py-4">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Navigation
        </p>
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {section.pages.map(page => (
              <NavLink
                key={page.path}
                to={page.path}
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
    </nav>
  )
}
