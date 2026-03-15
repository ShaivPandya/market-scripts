import axios from "axios"

import { getAuthMode } from "@/lib/authMode"

const client = axios.create({
  baseURL: (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/+$/, ""),
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
  login: (password: string) => client.post("/auth/login", { password }).then(r => r.data),
  logout: () => client.post("/auth/logout").then(r => r.data),
  me: () => client.get("/auth/me").then(r => r.data),
}

// ─── GET endpoints ───────────────────────────────────────────────────────────

export const fetchPortfolio = (timeframe: string) =>
  client.get(`/portfolio?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchPortfolioAllTimeframes = () =>
  client.get("/portfolio?all_timeframes=true").then(r => r.data)

export interface PortfolioPosition {
  ticker: string
  asset: "equity" | "commodity" | "fx" | "bond"
  direction: "long" | "short"
  distressed: boolean
  conviction: number
  cost_basis: number | null
  shares: number | null
  role?: "position" | "hedge"
}

export const fetchPortfolioPositions = (includeHedges = false) =>
  client
    .get("/portfolio-positions", { params: includeHedges ? { include_hedges: true } : undefined })
    .then(r => r.data as { positions: PortfolioPosition[] })

export const savePortfolioPositions = (positions: PortfolioPosition[]) =>
  client.put("/portfolio-positions", { positions }).then(r => r.data)

export interface HedgePosition {
  ticker: string
  direction: "long" | "short"
  cost_basis: number | null
  shares: number | null
}

export const fetchHedgePositions = () =>
  client.get("/hedge-positions").then(r => r.data as { positions: HedgePosition[] })

export const saveHedgePositions = (positions: HedgePosition[]) =>
  client.put("/hedge-positions", { positions }).then(r => r.data)

export type ThesisStatus = "populated" | "empty" | "missing"

export const fetchThesisStatus = () =>
  client.get("/thesis/status").then(r => r.data as Record<string, ThesisStatus>)

export const fetchThesis = (ticker: string) =>
  client
    .get(`/thesis/${encodeURIComponent(ticker)}`)
    .then(r => r.data as { status: "ok"; ticker: string; content: string })

export const saveThesisContent = (ticker: string, content: string) =>
  client
    .put(`/thesis/${encodeURIComponent(ticker)}`, { content })
    .then(r => r.data as { status: "ok"; ticker: string; content: string })

export const uploadThesisPdf = (ticker: string, file: File) => {
  const formData = new FormData()
  formData.append("ticker", ticker)
  formData.append("file", file)
  return client
    .post("/thesis/generate", formData, { timeout: 120_000 })
    .then(r => r.data as { status: "ok"; ticker: string; content: string })
}

// --- Thesis metadata types ---

export type ThesisStatusValue = "active" | "under_review" | "invalidated"

export interface ThesisEvaluation {
  id: number
  ticker: string
  evaluated_at: string
  thesis_status: string
  technical_read: string
  fundamental_read: string
  action: string
  confidence: string
  key_developments: string[]
  earnings_note: string | null
  risk_flag: string | null
}

export interface ThesisMeta {
  ticker: string
  status: ThesisStatusValue
  created_at: string
  updated_at: string
  latest_evaluation?: ThesisEvaluation | null
}

export interface ThesisDetail {
  meta: ThesisMeta
  content: string | null
  status_history: Array<{
    id: number
    ticker: string
    old_status: string | null
    new_status: string
    reason: string | null
    changed_at: string
  }>
  evaluations: ThesisEvaluation[]
}

export const fetchThesisMeta = () =>
  client.get("/thesis/meta").then(r => r.data as ThesisMeta[])

export const fetchThesisDetail = (ticker: string) =>
  client
    .get(`/thesis/${encodeURIComponent(ticker)}/detail`)
    .then(r => r.data as ThesisDetail)

export const updateThesisStatus = (
  ticker: string,
  status: ThesisStatusValue,
  reason: string,
) =>
  client
    .put(`/thesis/${encodeURIComponent(ticker)}/status`, { status, reason })
    .then(r => r.data)

export const fetchMomentum = () =>
  client.get("/momentum").then(r => r.data)

export const fetchSignalAggregator = (params?: {
  lookback_weeks?: number
  positioning_instruments?: string
  include_raw_modules?: boolean
}) => client.get("/signal-aggregator", { params, timeout: 180_000 }).then(r => r.data)

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

type OntologyQueryBody = {
  query?: string
  intent?: "portfolio_risk_exposure" | "positions_in_deteriorating_macro" | "entity_context"
  filters?: {
    tickers?: string[]
    sectors?: string[]
    assets?: string[]
    max_results?: number
    min_risk_score?: number
  }
  timeframe?: "This Week" | "Daily" | "Weekly" | "Monthly"
  include_graph?: boolean
  run_id?: string
  refresh_snapshot?: boolean
}

export interface OntologyRunSummary {
  run_id: string
  as_of: string
  created_at: string
  required_modules_ok: boolean
}

export const fetchOntologyRuns = (limit = 100) =>
  client
    .get("/ontology/runs", { params: { limit } })
    .then(r => r.data as { runs: OntologyRunSummary[] })

export const queryOntology = (body: OntologyQueryBody) =>
  client.post("/ontology/query", body, { timeout: 180_000 }).then(r => r.data)

type OntologyJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startOntologyQueryJob = (body: OntologyQueryBody) =>
  client.post("/ontology/query/async", body, { timeout: 30_000 }).then(r => r.data as OntologyJobResponse)

export const fetchOntologyQueryJob = (job_id: string) =>
  client.get(`/ontology/query/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as OntologyJobResponse)

export async function runOntologyQueryAsync(body: OntologyQueryBody, signal?: AbortSignal) {
  const started = await startOntologyQueryJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Ontology query failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  for (; ;) {
    if (signal?.aborted) throw new DOMException("Query cancelled", "AbortError")
    if (Date.now() > deadline) throw new Error("Timeout: Ontology query is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchOntologyQueryJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Ontology query failed")
  }
}

export const fetchEconomicGrowth = () =>
  client.get("/economic-growth").then(r => r.data)

export const fetchWeeklyReport = () =>
  client.get("/weekly-report", { timeout: 180_000 }).then(r => r.data)

export const fetchWeeklyReportCached = () =>
  client
    .get("/weekly-report", { params: { cached_only: true }, timeout: 30_000 })
    .then(r => r.data)
    .catch(err => {
      if (axios.isAxiosError(err) && err.response?.status === 404) return null
      throw err
    })

export const generateWeeklyReport = () =>
  client.get("/weekly-report", { params: { refresh: true }, timeout: 180_000 }).then(r => r.data)

export const analyzeMarketTechnicals = (body: {
  market_breadth: Record<string, unknown>
  top50_breadth: Record<string, unknown>
  vix_term_structure: Record<string, unknown>
  price_volume_signals: Record<string, unknown>
}) => client.post("/market-technicals/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const analyzeEconomicGrowth = (body: {
  commodities: Record<string, Record<string, number | null>>
  equities: Record<string, Record<string, number | null>>
  currencies: Record<string, Record<string, number | null>>
  equity_periods: string[]
  currency_periods: string[]
}) => client.post("/economic-growth/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchLaborMarket = () =>
  client.get("/labor-market").then(r => r.data)

export const analyzeLaborMarket = (body: {
  latest: Record<string, { value: number | null; date: string | null; change: number | null }>
  series_labels: Record<string, string>
  series_units: Record<string, string>
  timestamp?: string | null
}) => client.post("/labor-market/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchHousing = () =>
  client.get("/housing").then(r => r.data)

export const analyzeHousing = (body: {
  latest: Record<string, { value: number | null; date: string | null; change: number | null }>
  series_labels: Record<string, string>
  series_units: Record<string, string>
  timestamp?: string | null
}) => client.post("/housing/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchLiquidity = () =>
  client.get("/liquidity").then(r => r.data)

export const analyzeLiquidity = (body: {
  composite_score?: number | null
  regime?: string | null
  latest_date?: string | null
  regional_scores?: Record<string, unknown>
  components?: Record<string, unknown>[]
  changes?: Record<string, Record<string, unknown>>
}) => client.post("/liquidity/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchCountryDashboard = (metric: string) =>
  client.get(`/country-dashboard?metric=${encodeURIComponent(metric)}`).then(r => r.data)

export const fetchBreakout = () =>
  client.get("/breakout").then(r => r.data)

export const fetchCentralBanks = (refresh = false) =>
  client.get(`/central-banks?refresh=${refresh}`).then(r => r.data)

export const fetchPortfolioNews = (refresh = false) =>
  client.get(`/portfolio-news?refresh=${refresh}`).then(r => r.data)

export const fetchSectorMetrics = () =>
  client.get("/sector-metrics").then(r => r.data)

export const analyzeSectorMetrics = (body: {
  rows: Record<string, unknown>[]
  timestamp?: string | null
}) => client.post("/sector-metrics/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchIndustryMonitor = (refresh = false) =>
  client.get(`/industry-monitor?refresh=${refresh}`).then(r => r.data)

export const fetchYieldCurve = (lookback_days = 90) =>
  client.get(`/yield-curve?lookback_days=${lookback_days}`).then(r => r.data)

export const fetchBondDashboard = () =>
  client.get("/bond-dashboard").then(r => r.data)

export const fetchCommoditiesCurve = (commodity = "CL", lookback_days = 30) =>
  client
    .get(`/commodities-curve?commodity=${encodeURIComponent(commodity)}&lookback_days=${lookback_days}`)
    .then(r => r.data)

export const fetchPositioningSummary = (params: Record<string, string>) =>
  client.get("/positioning/summary", { params }).then(r => r.data)

export const fetchPositioningTimeseries = (params: Record<string, string>) =>
  client.get("/positioning/timeseries", { params }).then(r => r.data)

export const fetchPositioningInstruments = () =>
  client.get("/positioning/instruments").then(r => r.data)

export const analyzePositioning = (body: { rows: Record<string, unknown>[] }) =>
  client.post("/positioning/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchFxModelPairs = () =>
  client.get("/fx-model/pairs").then(r => r.data)

export const fetchHedgingToolPrefill = () =>
  client.get("/hedging-tool/prefill").then(r => r.data)

export const fetchHedgingPortfolioWeights = (book?: number) =>
  client
    .get("/hedging-tool/portfolio-weights", { params: book ? { book } : undefined })
    .then(r => r.data as {
      positions: { ticker: string; weight: number }[]
      metadata: { ticker: string; direction: string; conviction: number; shares: number | null; cost_basis: number | null; weight: number }[]
      book: number
      source: string
    })

export const fetchHedgingRecommendations = (body: Record<string, unknown>) =>
  client.post("/hedging-tool/recommend", body, { timeout: 180_000 }).then(r => r.data as { analysis: string })

export const fetchSizerPrefill = () =>
  client.get("/portfolio-sizer/prefill").then(r => r.data)

// ─── POST endpoints ───────────────────────────────────────────────────────────

export const runChart = (body: { ticker: string; lookback: string }) =>
  client.post("/chart", body).then(r => r.data)

export const runPriceRatioChart = (body: {
  symbol_a: string
  symbol_b: string
  method?: string
  start_date?: string
  end_date?: string
}) => client.post("/chart/ratio", body).then(r => r.data)

type AnalyzerRequest = {
  book?: number
  target_leverage?: number
  beta_neutral?: boolean
}

export const runPortfolioAnalyzer = (body: AnalyzerRequest = {}) =>
  client.post("/portfolio-analyzer", body, { timeout: 180_000 }).then(r => r.data)

type AnalyzerJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startPortfolioAnalyzerJob = (body: AnalyzerRequest = {}) =>
  client.post("/portfolio-analyzer/async", body, { timeout: 30_000 }).then(r => r.data as AnalyzerJobResponse)

export const fetchPortfolioAnalyzerJob = (job_id: string) =>
  client.get(`/portfolio-analyzer/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as AnalyzerJobResponse)

export async function runPortfolioAnalyzerAsync(body: AnalyzerRequest = {}) {
  const started = await startPortfolioAnalyzerJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Portfolio analyzer failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  // Poll until completion; each request is short to avoid edge proxy timeouts.
  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Portfolio analyzer is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchPortfolioAnalyzerJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      // cached:* jobs return done without result in the poll endpoint
      // (the initial response already carried the payload).
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Portfolio analyzer failed")
  }
}

// Backward-compatible exports.
export const runPortfolioOptimizer = runPortfolioAnalyzer
export const startPortfolioOptimizerJob = startPortfolioAnalyzerJob
export const fetchPortfolioOptimizerJob = fetchPortfolioAnalyzerJob
export async function runPortfolioOptimizerAsync(body: AnalyzerRequest = {}) {
  return runPortfolioAnalyzerAsync(body)
}

export const runHedgingTool = (body: { book: number; positions: { ticker: string; weight: number }[] }) =>
  client.post("/hedging-tool", body, { timeout: 180_000 }).then(r => r.data)

type HedgingJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startHedgingToolJob = (body: { book: number; positions: { ticker: string; weight: number }[] }) =>
  client.post("/hedging-tool/async", body, { timeout: 30_000 }).then(r => r.data as HedgingJobResponse)

export const fetchHedgingToolJob = (job_id: string) =>
  client.get(`/hedging-tool/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as HedgingJobResponse)

export async function runHedgingToolAsync(body: { book: number; positions: { ticker: string; weight: number }[] }) {
  const started = await startHedgingToolJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Hedging tool failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Hedging tool is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchHedgingToolJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Hedging tool failed")
  }
}

type SizerJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startSizerJob = (body: {
  book: number
  target_leverage: number
  positions: { ticker: string; conviction: number }[]
}) =>
  client.post("/portfolio-sizer/async", body, { timeout: 30_000 }).then(r => r.data as SizerJobResponse)

export const fetchSizerJob = (job_id: string) =>
  client.get(`/portfolio-sizer/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as SizerJobResponse)

export async function runPortfolioSizerAsync(body: {
  book: number
  target_leverage: number
  positions: { ticker: string; conviction: number }[]
}) {
  const started = await startSizerJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Sizer failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Sizer is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchSizerJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Sizer failed")
  }
}

export const runQualityScreen = (body: {
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}) => {
  const controller = new AbortController()
  const timeoutMs = 90_000
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  return client
    .post("/quality-screen", body, { signal: controller.signal, timeout: timeoutMs })
    .then(r => r.data)
    .catch(err => {
      if (axios.isAxiosError(err) && err.code === "ERR_CANCELED") {
        throw new Error("Timeout: Quality screen exceeded 90s. Try a smaller universe or custom tickers.")
      }
      throw err
    })
    .finally(() => clearTimeout(timer))
}

export const runShortScreen = (body: {
  input_mode: string
  universe: string
  tickers: string
  pb_threshold: number | null
  loss_type: string | null
  check_issuance: boolean
  check_revenue: boolean
  max_revenue_growth: number
  check_eps: boolean
  max_eps_growth: number
  check_52w_positive: boolean
  check_min_drawdown: boolean
  min_drawdown_pct: number
  check_max_drawdown: boolean
  max_drawdown_pct: number
  check_3m_neg_momentum: boolean
  check_2m_neg_rel_momentum: boolean
  rel_momentum_benchmark: string
}) => client.post("/short-screen", body, { timeout: 600_000 }).then(r => r.data)

type FundamentalMomentumRequest = {
  screen_type: string
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}

type FundamentalMomentumResponse = {
  screen_type?: string
  eps?: { results_df?: Record<string, unknown>[]; [key: string]: unknown }
  rev?: { results_df?: Record<string, unknown>[]; [key: string]: unknown }
  [key: string]: unknown
}

type FundamentalMomentumJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: FundamentalMomentumResponse }

export const startFundamentalMomentumJob = (body: FundamentalMomentumRequest) =>
  client.post("/fundamental-momentum/async", body, { timeout: 30_000 }).then(r => r.data as FundamentalMomentumJobResponse)

export const fetchFundamentalMomentumJob = (job_id: string) =>
  client.get(`/fundamental-momentum/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as FundamentalMomentumJobResponse)

export async function runFundamentalMomentumAsync(body: FundamentalMomentumRequest): Promise<FundamentalMomentumResponse> {
  const started = await startFundamentalMomentumJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Fundamental momentum failed")

  let job_id = started.job_id
  const deadline = Date.now() + 300_000
  let restartedAfterUnknownJob = false

  for (; ;) {
    if (Date.now() > deadline) {
      throw new Error("Timeout: Fundamental momentum is taking too long. Try a smaller universe or custom tickers.")
    }

    await new Promise(r => setTimeout(r, 2000))
    let job: FundamentalMomentumJobResponse
    try {
      job = await fetchFundamentalMomentumJob(job_id)
    } catch (err) {
      const isUnknownJob =
        axios.isAxiosError(err) &&
        err.response?.status === 404 &&
        typeof err.message === "string" &&
        err.message.includes("Unknown job_id")

      if (isUnknownJob && !restartedAfterUnknownJob) {
        restartedAfterUnknownJob = true
        const restarted = await startFundamentalMomentumJob(body)
        if (restarted.status === "done" && "result" in restarted && restarted.result != null) return restarted.result
        if (restarted.status === "error") throw new Error(restarted.error || "Fundamental momentum failed")
        job_id = restarted.job_id
        continue
      }
      throw err
    }

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started && started.result != null ? started.result : {}
    }
    if (job.status === "error") throw new Error(job.error || "Fundamental momentum failed")
  }
}

// Keep existing import call-sites unchanged.
export const runFundamentalMomentum = runFundamentalMomentumAsync

export const runFinancials = (body: { ticker: string }) =>
  client.post("/financials", body).then(r => r.data)

export const runFxModel = (body: {
  pair: string
  bootstrap: number
  skip_bis: boolean
  horizons: string
}) => client.post("/fx-model", body).then(r => r.data)

export const fetchSentimentPutCall = (lookback_days = 180) =>
  client.get(`/sentiment/put-call?lookback_days=${lookback_days}`).then(r => r.data)

export const fetchSentimentSurveys = () =>
  client.get("/sentiment/surveys").then(r => r.data)

export const fetchSentimentVolatility = (lookback_days = 365) =>
  client.get(`/sentiment/volatility?lookback_days=${lookback_days}`).then(r => r.data)

export const analyzeSentiment = (body: {
  put_call: Record<string, unknown>
  surveys: Record<string, unknown>
  volatility: unknown[]
}) => client.post("/sentiment/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const clearCache = () => client.delete("/cache").then(r => r.data)

// ---------------------------------------------------------------------------
// Investing OS APIs
// ---------------------------------------------------------------------------

// Workspace
export const fetchWorkspace = () => client.get("/workspace").then(r => r.data)

// Dossier
export const fetchDossier = (ticker: string) =>
  client.get(`/dossier/${encodeURIComponent(ticker)}`).then(r => r.data)

// Approvals
export const fetchApprovals = (status?: string) =>
  client.get("/approvals", { params: status ? { status } : undefined }).then(r => r.data)
export const approveItem = (id: number, note?: string) =>
  client.post(`/approvals/${id}/approve`, note ? { note } : {}).then(r => r.data)
export const rejectItem = (id: number, note?: string) =>
  client.post(`/approvals/${id}/reject`, note ? { note } : {}).then(r => r.data)
export const bulkApprove = (ids: number[], note?: string) =>
  client.post("/approvals/bulk-approve", { ids, note }).then(r => r.data)
export const bulkReject = (ids: number[], note?: string) =>
  client.post("/approvals/bulk-reject", { ids, note }).then(r => r.data)

// Action Items
export const fetchActions = (params?: { status?: string; ticker?: string }) =>
  client.get("/actions", { params }).then(r => r.data)
export const createAction = (body: { description: string; action_type?: string; ticker?: string; urgency?: string }) =>
  client.post("/actions", body).then(r => r.data)
export const completeAction = (id: number, resolution_note?: string) =>
  client.put(`/actions/${id}/complete`, { resolution_note: resolution_note ?? "" }).then(r => r.data)
export const dismissAction = (id: number) =>
  client.put(`/actions/${id}/dismiss`).then(r => r.data)

// Watch Triggers
export const fetchTriggers = (params?: { status?: string; ticker?: string }) =>
  client.get("/triggers", { params }).then(r => r.data)
export const createTrigger = (body: { condition: string; trigger_type?: string; ticker?: string; expires_at?: string }) =>
  client.post("/triggers", body).then(r => r.data)
export const fireTrigger = (id: number) =>
  client.put(`/triggers/${id}/fire`).then(r => r.data)
export const cancelTrigger = (id: number) =>
  client.put(`/triggers/${id}/cancel`).then(r => r.data)

// Catalysts
export const fetchCatalysts = (ticker: string) =>
  client.get("/catalysts", { params: { ticker } }).then(r => r.data)
export const createCatalyst = (body: { ticker: string; description: string; category?: string; target_date?: string }) =>
  client.post("/catalysts", body).then(r => r.data)
export const updateCatalystStatus = (id: number, status: string, evidence?: string) =>
  client.put(`/catalysts/${id}/status`, { status, evidence }).then(r => r.data)

// Kill Conditions
export const fetchKillConditions = (ticker: string) =>
  client.get("/kill-conditions", { params: { ticker } }).then(r => r.data)
export const createKillCondition = (body: { ticker: string; condition: string; metric?: string; threshold?: string }) =>
  client.post("/kill-conditions", body).then(r => r.data)
export const updateKillConditionStatus = (id: number, status: string) =>
  client.put(`/kill-conditions/${id}/status`, { status }).then(r => r.data)

// Research Notes
export const fetchResearchNotes = (params?: { ticker?: string; limit?: number }) =>
  client.get("/research-notes", { params }).then(r => r.data)
export const createResearchNote = (body: { title: string; content: string; ticker?: string; note_type?: string }) =>
  client.post("/research-notes", body).then(r => r.data)

// Workflow Runs
export const fetchWorkflowRuns = (params?: { workflow_name?: string; ticker?: string; limit?: number }) =>
  client.get("/workflow-runs", { params }).then(r => r.data)
export const fetchWorkflowRun = (runId: string) =>
  client.get(`/workflow-runs/${runId}`).then(r => r.data)
