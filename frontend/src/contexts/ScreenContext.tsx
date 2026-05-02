import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { useLocation, useParams } from "react-router-dom"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ScreenContext {
  /** Human-readable page name */
  pageName: string
  /** Current route path */
  route: string
  /** Ticker if the page is ticker-specific (e.g. PositionDossier) */
  ticker?: string
  /** Key metrics visible on screen — formatted strings, not raw data */
  metrics?: Record<string, string>
  /** Active filters / parameters the user has applied */
  filters?: Record<string, string>
  /** Short summary of what the user is looking at */
  summary?: string
  /** Agent tools whose data overlaps with this page */
  correspondingTools?: string[]
}

// ---------------------------------------------------------------------------
// Route → page name + tool mapping (auto-baseline for all pages)
// ---------------------------------------------------------------------------

interface RouteEntry {
  pageName: string
  tools?: string[]
}

const ROUTE_PAGE_MAP: Record<string, RouteEntry> = {
  // Core
  "/": { pageName: "Portfolio Dashboard", tools: ["get_portfolio"] },
  "/workspace": { pageName: "Workspace" },
  "/theses": { pageName: "Investment Theses", tools: ["get_thesis"] },
  "/weekly-report": { pageName: "Weekly Report" },
  "/dossier": { pageName: "Position Dossier", tools: ["get_dossier"] },

  // Labs
  "/analyzer": { pageName: "Portfolio Analyzer" },
  "/sizer": { pageName: "Portfolio Sizer" },
  "/hedging-tool": { pageName: "Hedging Tool" },
  "/chart": { pageName: "Chart" },
  "/screeners": { pageName: "Screeners" },
  "/financials": { pageName: "Financials" },
  "/fx-model": { pageName: "FX Model" },
  "/momentum": { pageName: "Momentum" },

  // Monitors
  "/signal-aggregator": { pageName: "Signal Aggregator", tools: ["get_signal_aggregator"] },
  "/ontology": { pageName: "Ontology Workbench", tools: ["query_ontology"] },
  "/market-technicals": { pageName: "Market Technicals", tools: ["get_market_breadth", "get_vix_term_structure"] },
  "/portfolio-news": { pageName: "News Digests" },
  "/breakout": { pageName: "Breakout", tools: ["get_breakout"] },
  "/sentiment": { pageName: "Sentiment", tools: ["get_sentiment"] },
  "/positioning": { pageName: "Positioning", tools: ["get_positioning"] },
  "/central-banks": { pageName: "Central Bank Monitor", tools: ["get_central_banks"] },
  "/industry-monitor": { pageName: "Industry Monitor", tools: ["get_industry_monitor"] },
  "/sector-metrics": { pageName: "Sector Metrics", tools: ["get_sector_metrics"] },

  // Macro
  "/economic-growth": { pageName: "Economic Growth", tools: ["get_economic_growth"] },
  "/labor-market": { pageName: "Labor Market", tools: ["get_labor_market"] },
  "/housing": { pageName: "Housing", tools: ["get_housing"] },
  "/liquidity": { pageName: "Liquidity", tools: ["get_liquidity"] },
  "/yield-curve": { pageName: "Yield Curve", tools: ["get_yield_curve"] },
  "/bond-dashboard": { pageName: "Bond Dashboard", tools: ["get_bond_dashboard"] },
  "/country-dashboard": { pageName: "Country Dashboard" },

  // Assets
  "/index-dashboard": { pageName: "Index Dashboard" },
  "/fx-dashboard": { pageName: "FX Dashboard" },
  "/commodities": { pageName: "Commodity Dashboard" },
  "/commodities-curve": { pageName: "Commodities Curve" },
  "/commodity-research": { pageName: "Commodity Proxy Screener" },
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface ScreenContextValue {
  screenContext: ScreenContext | null
  setScreenContext: (ctx: ScreenContext | null) => void
}

const ScreenCtx = createContext<ScreenContextValue>({
  screenContext: null,
  setScreenContext: () => {},
})

export function ScreenContextProvider({ children }: { children: ReactNode }) {
  const [screenContext, setScreenContext] = useState<ScreenContext | null>(null)
  return (
    <ScreenCtx.Provider value={{ screenContext, setScreenContext }}>
      {children}
    </ScreenCtx.Provider>
  )
}

// ---------------------------------------------------------------------------
// Consumer hook — used by Layout / AgentChat to read context
// ---------------------------------------------------------------------------

export function useScreenContext() {
  return useContext(ScreenCtx)
}

// ---------------------------------------------------------------------------
// Auto-baseline hook — derives minimal context from the current route
// ---------------------------------------------------------------------------

export function useAutoScreenContext(): ScreenContext {
  const location = useLocation()
  const params = useParams()

  // Strip dynamic segments for lookup (e.g. /dossier/AAPL → /dossier)
  const basePath = location.pathname.replace(/\/[A-Z0-9.]+$/i, "") || "/"
  const entry = ROUTE_PAGE_MAP[location.pathname] ?? ROUTE_PAGE_MAP[basePath] ?? { pageName: location.pathname }

  return {
    pageName: entry.pageName,
    route: location.pathname,
    ticker: params.ticker,
    correspondingTools: entry.tools,
  }
}

// ---------------------------------------------------------------------------
// Producer hook — pages call this to register rich context
// ---------------------------------------------------------------------------

export function useRegisterScreenContext(ctx: Omit<ScreenContext, "route"> | null) {
  const location = useLocation()
  const { setScreenContext } = useContext(ScreenCtx)

  useEffect(() => {
    if (ctx) {
      setScreenContext({ ...ctx, route: location.pathname })
    } else {
      setScreenContext(null)
    }
    return () => setScreenContext(null)
  // Serialize ctx to avoid infinite loops from new object refs each render.
  // Pages should wrap their ctx in useMemo.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(ctx), location.pathname, setScreenContext])
}
