import axios from "axios"

import { getAuthMode } from "@/lib/authMode"

const client = axios.create({
  baseURL: (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/+$/, ""),
  withCredentials: true,
  // Avoid "spinning forever" when the backend (or an upstream like Cloudflare) hangs.
  timeout: 60_000,
})

function _truncate(s: string, maxLen: number) {
  if (s.length <= maxLen) return s
  return s.slice(0, maxLen - 1) + "…"
}

function _extractDetail(data: unknown): unknown {
  if (data == null) return undefined
  if (typeof data === "string") return data
  if (typeof data !== "object") return undefined
  const rec = data as Record<string, unknown>
  if ("detail" in rec) return rec.detail
  if ("message" in rec) return rec.message
  if ("error" in rec) return rec.error
  return undefined
}

function formatApiError(err: unknown): string | null {
  if (!axios.isAxiosError(err)) return null

  const isTimeout =
    err.code === "ECONNABORTED" ||
    (typeof err.message === "string" && err.message.toLowerCase().includes("timeout"))

  const status = err.response?.status
  const data = err.response?.data
  const detail = _extractDetail(data)

  const prefix = status ? `${status}: ` : isTimeout ? "Timeout: " : ""

  if (typeof detail === "string" && detail.trim()) return prefix + _truncate(detail.trim(), 500)

  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>
    const msg = typeof d.message === "string" ? d.message.trim() : ""
    const failed = Array.isArray(d.failed) ? d.failed.filter(x => typeof x === "string") as string[] : []
    if (msg) {
      const extra = failed.length ? ` (failed: ${_truncate(failed.join(", "), 220)})` : ""
      return prefix + _truncate(msg + extra, 500)
    }
    try {
      return prefix + _truncate(JSON.stringify(detail), 500)
    } catch {
      // fall through
    }
  }

  if (typeof data === "string" && data.trim()) return prefix + _truncate(data.trim(), 500)
  if (err.message && err.message.trim()) return prefix + _truncate(err.message.trim(), 500)
  return status ? `${status}: Request failed` : "Request failed"
}

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
    const msg = formatApiError(err)
    if (msg) err.message = msg
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

export const fetchPortfolioAllTimeframes = () =>
  client.get("/portfolio?all_timeframes=true").then(r => r.data)

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

export const analyzeMarketTechnicals = (body: {
  market_breadth: Record<string, unknown>
  top50_breadth: Record<string, unknown>
  vix_term_structure: Record<string, unknown>
  price_volume_signals: Record<string, unknown>
}) => client.post("/market-technicals/analyze", body).then(r => r.data)

export const analyzeEconomicGrowth = (body: {
  commodities: Record<string, Record<string, number | null>>
  equities: Record<string, Record<string, number | null>>
  currencies: Record<string, Record<string, number | null>>
  equity_periods: string[]
  currency_periods: string[]
}) => client.post("/economic-growth/analyze", body).then(r => r.data)

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

export const fetchYieldCurve = (lookback_days = 90) =>
  client.get(`/yield-curve?lookback_days=${lookback_days}`).then(r => r.data)

export const fetchPositioningSummary = (params: Record<string, string>) =>
  client.get("/positioning/summary", { params }).then(r => r.data)

export const fetchPositioningTimeseries = (params: Record<string, string>) =>
  client.get("/positioning/timeseries", { params }).then(r => r.data)

export const fetchPositioningInstruments = () =>
  client.get("/positioning/instruments").then(r => r.data)

export const analyzePositioning = (body: { rows: Record<string, unknown>[] }) =>
  client.post("/positioning/analyze", body).then(r => r.data)

export const fetchFxModelPairs = () =>
  client.get("/fx-model/pairs").then(r => r.data)

// ─── POST endpoints ───────────────────────────────────────────────────────────

export const runChart = (body: { ticker: string; lookback: string }) =>
  client.post("/chart", body).then(r => r.data)

export const runPortfolioOptimizer = (body: { book: number; target_leverage: number }) =>
  client.post("/portfolio-optimizer", body, { timeout: 180_000 }).then(r => r.data)

type OptimizerJobStatus = "queued" | "running" | "done" | "error"
type OptimizerJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startPortfolioOptimizerJob = (body: { book: number; target_leverage: number }) =>
  client.post("/portfolio-optimizer/async", body, { timeout: 30_000 }).then(r => r.data as OptimizerJobResponse)

export const fetchPortfolioOptimizerJob = (job_id: string) =>
  client.get(`/portfolio-optimizer/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as OptimizerJobResponse)

export async function runPortfolioOptimizerAsync(body: { book: number; target_leverage: number }) {
  const started = await startPortfolioOptimizerJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result as any
  if (started.status === "error") throw new Error(started.error || "Optimizer failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  // Poll until completion; each request is short to avoid edge proxy timeouts.
  for (;;) {
    if (Date.now() > deadline) throw new Error("Timeout: Optimizer is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchPortfolioOptimizerJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result as any
      // cached:* jobs return done without result in the poll endpoint
      // (the initial response already carried the payload).
      return (started as any).result
    }
    if (job.status === "error") throw new Error(job.error || "Optimizer failed")
  }
}

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

export const runFinancials = (body: { ticker: string }) =>
  client.post("/financials", body).then(r => r.data)

export const runFxModel = (body: {
  pair: string
  bootstrap: number
  skip_bis: boolean
  horizons: string
}) => client.post("/fx-model", body).then(r => r.data)

export const clearCache = () => client.delete("/cache").then(r => r.data)
