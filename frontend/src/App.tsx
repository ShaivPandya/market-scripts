import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { AuthProvider, useAuth } from "@/contexts/AuthContext"
import { ThemeProvider } from "@/contexts/ThemeContext"
import { ProtectedRoute } from "@/components/auth/ProtectedRoute"
import { Layout } from "@/components/layout/Layout"

// Pages
import { LoginPage } from "@/pages/LoginPage"
import { PortfolioDashboard } from "@/pages/PortfolioDashboard"
import { PortfolioAnalyzer } from "@/pages/PortfolioAnalyzer"
import { HedgingTool } from "@/pages/HedgingTool"
import { PortfolioSizer } from "@/pages/PortfolioSizer"
import { Momentum } from "@/pages/Momentum"
import { ChartPage } from "@/pages/ChartPage"
import { QualityScreen } from "@/pages/QualityScreen"
import { ShortScreen } from "@/pages/ShortScreen"
import { FundamentalMomentum } from "@/pages/FundamentalMomentum"
import { Financials } from "@/pages/Financials"
import { IndexDashboard } from "@/pages/IndexDashboard"
import { FXDashboard } from "@/pages/FXDashboard"
import { CommodityDashboard } from "@/pages/CommodityDashboard"
import { MarketTechnicals } from "@/pages/MarketTechnicals"
import { SectorMetrics } from "@/pages/SectorMetrics"
import { Positioning } from "@/pages/Positioning"
import { Breakout } from "@/pages/Breakout"
import { FXModel } from "@/pages/FXModel"
import { EconomicGrowth } from "@/pages/EconomicGrowth"
import { LaborMarket } from "@/pages/LaborMarket"
import { Liquidity } from "@/pages/Liquidity"
import { CountryDashboard } from "@/pages/CountryDashboard"
import { CentralBanks } from "@/pages/CentralBanks"
import { IndustryMonitor } from "@/pages/IndustryMonitor"
import { YieldCurve } from "@/pages/YieldCurve"
import { CommoditiesCurve } from "@/pages/CommoditiesCurve"
import { PortfolioNews } from "@/pages/PortfolioNews"
import { WeeklyReport } from "@/pages/WeeklyReport"
import { Sentiment } from "@/pages/Sentiment"
import { SignalAggregator } from "@/pages/SignalAggregator"

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

  return isAuthenticated ? <Navigate to="/portfolio" replace /> : <LoginPage />
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<LoginRoute />} />

      {/* Protected — all existing routes */}
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/portfolio" replace />} />
          <Route path="/portfolio" element={<PortfolioDashboard />} />
          <Route path="/analyzer" element={<PortfolioAnalyzer />} />
          <Route path="/optimizer" element={<Navigate to="/analyzer" replace />} />
          <Route path="/hedging-tool" element={<HedgingTool />} />
          <Route path="/sizer" element={<PortfolioSizer />} />
          <Route path="/momentum" element={<Momentum />} />
          <Route path="/signal-aggregator" element={<SignalAggregator />} />
          <Route path="/chart" element={<ChartPage />} />
          <Route path="/quality" element={<QualityScreen />} />
          <Route path="/short-screen" element={<ShortScreen />} />
          <Route path="/fundamental-momentum" element={<FundamentalMomentum />} />
          <Route path="/financials" element={<Financials />} />
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
          <Route path="/liquidity" element={<Liquidity />} />
          <Route path="/country-dashboard" element={<CountryDashboard />} />
          <Route path="/central-banks" element={<CentralBanks />} />
          <Route path="/industry-monitor" element={<IndustryMonitor />} />
          <Route path="/yield-curve" element={<YieldCurve />} />
          <Route path="/commodities-curve" element={<CommoditiesCurve />} />
          <Route path="/portfolio-news" element={<PortfolioNews />} />
          <Route path="/weekly-report" element={<WeeklyReport />} />
        </Route>
      </Route>
    </Routes>
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
