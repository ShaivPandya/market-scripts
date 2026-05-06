import { Component, lazy, Suspense, type ComponentType, type ErrorInfo, type ReactNode } from "react"
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { AuthProvider, useAuth } from "@/contexts/AuthContext"
import { ThemeProvider } from "@/contexts/ThemeContext"
import { ProtectedRoute } from "@/components/auth/ProtectedRoute"
import { Layout } from "@/components/layout/Layout"
import { isChunkLoadError, maybeReloadForChunkLoadError } from "@/lib/chunkRecovery"

const PAGE_LOAD_RETRY_DELAYS_MS = [300, 1_000, 2_000] as const

async function loadPageModule<T extends ComponentType>(
  loader: () => Promise<Record<string, T>>,
) {
  let lastError: unknown

  for (let attempt = 0; attempt <= PAGE_LOAD_RETRY_DELAYS_MS.length; attempt += 1) {
    if (attempt > 0) {
      await new Promise(resolve => setTimeout(resolve, PAGE_LOAD_RETRY_DELAYS_MS[attempt - 1]))
    }

    try {
      return await loader()
    } catch (error) {
      lastError = error
    }
  }

  throw lastError
}

function lazyPage<T extends ComponentType>(
  loader: () => Promise<Record<string, T>>,
  exportName: string,
) {
  return lazy(async () => {
    const mod = await loadPageModule(loader)
    const page = mod[exportName]
    if (!page) throw new Error(`Lazy page export "${exportName}" was not found.`)
    return { default: page }
  })
}

class RouteErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: unknown }> {
  state = { error: null as unknown }

  static getDerivedStateFromError(error: unknown) {
    return { error }
  }

  componentDidUpdate(prevProps: { resetKey: string }) {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null })
    }
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    if (maybeReloadForChunkLoadError(error, "route-lazy")) return
    console.error("Unhandled route render error", error, info)
  }

  render() {
    if (!this.state.error) return this.props.children

    const chunkLoadError = isChunkLoadError(this.state.error)

    return (
      <div className="flex min-h-screen items-center justify-center bg-app px-4 text-app">
        <section className="theme-surface max-w-md p-6 text-center">
          <h1 className="text-lg font-semibold text-app">
            {chunkLoadError ? "App Update Required" : "Page Failed To Load"}
          </h1>
          <p className="mt-2 text-sm leading-6 text-muted">
            {chunkLoadError
              ? "A deployed page module could not be loaded. Refresh to pull the current application bundle."
              : "This page hit an unexpected rendering error."}
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="theme-button-base theme-button-primary mt-5 min-h-10 px-4 text-sm"
          >
            Reload Application
          </button>
        </section>
      </div>
    )
  }
}

function RouteLoadBoundary({ children }: { children: ReactNode }) {
  const location = useLocation()
  return <RouteErrorBoundary resetKey={location.pathname}>{children}</RouteErrorBoundary>
}

const LoginPage = lazyPage(() => import("@/pages/LoginPage"), "LoginPage")
const PortfolioDashboard = lazyPage(() => import("@/pages/PortfolioDashboard"), "PortfolioDashboard")
const PortfolioAnalyzer = lazyPage(() => import("@/pages/PortfolioAnalyzer"), "PortfolioAnalyzer")
const HedgingTool = lazyPage(() => import("@/pages/HedgingTool"), "HedgingTool")
const PortfolioSizer = lazyPage(() => import("@/pages/PortfolioSizer"), "PortfolioSizer")
const Momentum = lazyPage(() => import("@/pages/Momentum"), "Momentum")
const ChartPage = lazyPage(() => import("@/pages/ChartPage"), "ChartPage")
const Screeners = lazyPage(() => import("@/pages/Screeners"), "Screeners")
const Financials = lazyPage(() => import("@/pages/Financials"), "Financials")
const IndexDashboard = lazyPage(() => import("@/pages/IndexDashboard"), "IndexDashboard")
const FXDashboard = lazyPage(() => import("@/pages/FXDashboard"), "FXDashboard")
const CommodityDashboard = lazyPage(() => import("@/pages/CommodityDashboard"), "CommodityDashboard")
const MarketTechnicals = lazyPage(() => import("@/pages/MarketTechnicals"), "MarketTechnicals")
const SectorMetrics = lazyPage(() => import("@/pages/SectorMetrics"), "SectorMetrics")
const Positioning = lazyPage(() => import("@/pages/Positioning"), "Positioning")
const Breakout = lazyPage(() => import("@/pages/Breakout"), "Breakout")
const FXModel = lazyPage(() => import("@/pages/FXModel"), "FXModel")
const EconomicGrowth = lazyPage(() => import("@/pages/EconomicGrowth"), "EconomicGrowth")
const LaborMarket = lazyPage(() => import("@/pages/LaborMarket"), "LaborMarket")
const Housing = lazyPage(() => import("@/pages/Housing"), "Housing")
const Liquidity = lazyPage(() => import("@/pages/Liquidity"), "Liquidity")
const CountryDashboard = lazyPage(() => import("@/pages/CountryDashboard"), "CountryDashboard")
const CentralBanks = lazyPage(() => import("@/pages/CentralBanks"), "CentralBanks")
const IndustryMonitor = lazyPage(() => import("@/pages/IndustryMonitor"), "IndustryMonitor")
const YieldCurve = lazyPage(() => import("@/pages/YieldCurve"), "YieldCurve")
const BondDashboard = lazyPage(() => import("@/pages/BondDashboard"), "BondDashboard")
const CommoditiesCurve = lazyPage(() => import("@/pages/CommoditiesCurve"), "CommoditiesCurve")
const CommodityResearch = lazyPage(() => import("@/pages/CommodityResearch"), "CommodityResearch")
const PortfolioNews = lazyPage(() => import("@/pages/PortfolioNews"), "PortfolioNews")
const WeeklyReport = lazyPage(() => import("@/pages/WeeklyReport"), "WeeklyReport")
const Sentiment = lazyPage(() => import("@/pages/Sentiment"), "Sentiment")
const SignalAggregator = lazyPage(() => import("@/pages/SignalAggregator"), "SignalAggregator")
const OntologyWorkbench = lazyPage(() => import("@/pages/OntologyWorkbench"), "OntologyWorkbench")
const ThesisManager = lazyPage(() => import("@/pages/ThesisManager"), "ThesisManager")
const Workspace = lazyPage(() => import("@/pages/Workspace"), "Workspace")
const PositionDossier = lazyPage(() => import("@/pages/PositionDossier"), "PositionDossier")
const IdeaWatchlist = lazyPage(() => import("@/pages/IdeaWatchlist"), "IdeaWatchlist")
const DCFModel = lazyPage(() => import("@/pages/DCFModel"), "DCFModel")
const AISettings = lazyPage(() => import("@/pages/AISettings"), "AISettings")

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 min — matches backend TTL cache
    },
  },
})

function LoginRoute() {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-app px-4 text-sm text-muted">
        Loading...
      </div>
    )
  }

  return isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
}

function RouteLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-app px-4 text-sm text-muted">
      Loading...
    </div>
  )
}

function AppRoutes() {
  return (
    <RouteLoadBoundary>
      <Suspense fallback={<RouteLoading />}>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<LoginRoute />} />

          {/* Protected — all existing routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route index element={<PortfolioDashboard />} />
              <Route path="/portfolio" element={<Navigate to="/" replace />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/dossier/:ticker" element={<PositionDossier />} />
              <Route path="/ideas" element={<IdeaWatchlist />} />
              <Route path="/theses" element={<ThesisManager />} />
              <Route path="/analyzer" element={<PortfolioAnalyzer />} />
              <Route path="/optimizer" element={<Navigate to="/analyzer" replace />} />
              <Route path="/hedging-tool" element={<HedgingTool />} />
              <Route path="/sizer" element={<PortfolioSizer />} />
              <Route path="/momentum" element={<Momentum />} />
              <Route path="/signal-aggregator" element={<SignalAggregator />} />
              <Route path="/ontology" element={<OntologyWorkbench />} />
              <Route path="/chart" element={<ChartPage />} />
              <Route path="/screeners" element={<Screeners />} />
              <Route path="/quality" element={<Navigate to="/screeners" replace />} />
              <Route path="/short-screen" element={<Navigate to="/screeners" replace />} />
              <Route path="/fundamental-momentum" element={<Navigate to="/screeners" replace />} />
              <Route path="/financials" element={<Financials />} />
              <Route path="/dcf-model" element={<DCFModel />} />
              <Route path="/index-dashboard" element={<IndexDashboard />} />
              <Route path="/fx-dashboard" element={<FXDashboard />} />
              <Route path="/commodities" element={<CommodityDashboard />} />
              <Route path="/market-technicals" element={<MarketTechnicals />} />
              <Route path="/sector-metrics" element={<SectorMetrics />} />
              <Route path="/positioning" element={<Positioning />} />
              <Route path="/sentiment" element={<Sentiment />} />
              <Route path="/breakout" element={<Breakout />} />
              <Route path="/fx-model" element={<FXModel />} />
              <Route path="/economic-growth" element={<EconomicGrowth />} />
              <Route path="/labor-market" element={<LaborMarket />} />
              <Route path="/housing" element={<Housing />} />
              <Route path="/liquidity" element={<Liquidity />} />
              <Route path="/country-dashboard" element={<CountryDashboard />} />
              <Route path="/central-banks" element={<CentralBanks />} />
              <Route path="/industry-monitor" element={<IndustryMonitor />} />
              <Route path="/yield-curve" element={<YieldCurve />} />
              <Route path="/bond-dashboard" element={<BondDashboard />} />
              <Route path="/commodities-curve" element={<CommoditiesCurve />} />
              <Route path="/commodity-research" element={<CommodityResearch />} />
              <Route path="/portfolio-news" element={<PortfolioNews />} />
              <Route path="/weekly-report" element={<WeeklyReport />} />
              <Route path="/settings/ai" element={<AISettings />} />
            </Route>
          </Route>
        </Routes>
      </Suspense>
    </RouteLoadBoundary>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <AuthProvider>
            <AppRoutes />
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
