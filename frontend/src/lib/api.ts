import axios from "axios"

import { getAuthMode } from "@/lib/authMode"

const client = axios.create({ baseURL: "/api", withCredentials: true })

client.interceptors.response.use(
  res => res,
  err => {
    if (
      getAuthMode() !== "cloudflare" &&
      err.response?.status === 401 &&
      !window.location.pathname.startsWith("/login")
    ) {
      window.location.href = "/login"
    }
    return Promise.reject(err)
  },
)

export const authApi = {
  login:  (password: string) => client.post("/auth/login", { password }).then(r => r.data),
  logout: ()                 => client.post("/auth/logout").then(r => r.data),
  me:     ()                 => client.get("/auth/me").then(r => r.data),
}

// ─── GET endpoints ───────────────────────────────────────────────────────────

export const fetchPortfolio = (timeframe: string) =>
  client.get(`/portfolio?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchMomentum = () =>
  client.get("/momentum").then(r => r.data)

export const fetchIndexDashboard = (timeframe: string) =>
  client.get(`/index-dashboard?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchFxDashboard = (timeframe: string) =>
  client.get(`/fx-dashboard?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchCommodities = (timeframe: string) =>
  client.get(`/commodities?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchMarketBreadth = () =>
  client.get("/market-breadth").then(r => r.data)

export const fetchTop50Breadth = () =>
  client.get("/top50-breadth").then(r => r.data)

export const fetchPriceVolumeSignals = () =>
  client.get("/price-volume-signals").then(r => r.data)

export const fetchVixTermStructure = () =>
  client.get("/vix-term-structure").then(r => r.data)

export const fetchEconomicGrowth = () =>
  client.get("/economic-growth").then(r => r.data)

export const fetchLiquidity = (skip_ecb: boolean) =>
  client.get(`/liquidity?skip_ecb=${skip_ecb}`).then(r => r.data)

export const fetchCountryDashboard = (metric: string) =>
  client.get(`/country-dashboard?metric=${encodeURIComponent(metric)}`).then(r => r.data)

export const fetchBreakout = () =>
  client.get("/breakout").then(r => r.data)

export const fetchCentralBanks = (refresh = false) =>
  client.get(`/central-banks?refresh=${refresh}`).then(r => r.data)

export const fetchSectorMetrics = () =>
  client.get("/sector-metrics").then(r => r.data)

export const fetchIndustryMonitor = (refresh = false) =>
  client.get(`/industry-monitor?refresh=${refresh}`).then(r => r.data)

export const fetchPositioningSummary = (params: Record<string, string>) =>
  client.get("/positioning/summary", { params }).then(r => r.data)

export const fetchPositioningTimeseries = (params: Record<string, string>) =>
  client.get("/positioning/timeseries", { params }).then(r => r.data)

export const fetchPositioningInstruments = () =>
  client.get("/positioning/instruments").then(r => r.data)

export const fetchFxModelPairs = () =>
  client.get("/fx-model/pairs").then(r => r.data)

// ─── POST endpoints ───────────────────────────────────────────────────────────

export const runChart = (body: { ticker: string; lookback: string }) =>
  client.post("/chart", body).then(r => r.data)

export const runPortfolioOptimizer = (body: { book: number; target_leverage: number }) =>
  client.post("/portfolio-optimizer", body).then(r => r.data)

export const runQualityScreen = (body: {
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}) => client.post("/quality-screen", body).then(r => r.data)

export const runShortScreen = (body: {
  pb_threshold: number
  loss_type: string
  check_issuance: boolean
}) => client.post("/short-screen", body).then(r => r.data)

export const runFundamentalMomentum = (body: {
  screen_type: string
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}) => client.post("/fundamental-momentum", body).then(r => r.data)

export const runFxModel = (body: {
  pair: string
  bootstrap: number
  skip_bis: boolean
  horizons: string
}) => client.post("/fx-model", body).then(r => r.data)
