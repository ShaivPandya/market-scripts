import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { Info, Plus, X } from "lucide-react"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SegmentedControl, SliderInput, TextInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MetricCard } from "@/components/shared/MetricCard"
import { DecisionStateBadge, EffectScopeBadge, QualityStateBadge } from "@/components/shared/DecisionStateBadge"
import { colorPositiveNegative } from "@/lib/colors"
import { fetchPortfolioPositions, fetchSizerJob, fetchSizerPrefill, startSizerJob, type BetaHedgeMode } from "@/lib/api"
import { groupKey, normalizeGroupConviction, normalizeGroupName } from "@/lib/positionGroups"
import { cn } from "@/lib/utils"

type SizerTab = "Weights" | "Exposures" | "Constraints" | "Max Scaled"
type ExposureAssetClass = "equity" | "fx" | "commodity" | "bond"
type WeightsViewMode = "basic" | "advanced"
type PresetHedgeTicker = "SPY" | "IWM" | "QQQ"

interface SizerExposures {
  equity_gross?: number
  equity_net?: number
  fx_gross?: number
  fx_net?: number
  commodity_gross?: number
  commodity_net?: number
  bond_gross?: number
  bond_net?: number
  hedge_gross?: number
  total_gross?: number
  total_net?: number
  [key: string]: unknown
}

interface SizerConstraint {
  utilization?: number
  current?: number
  limit?: number
  [key: string]: unknown
}

interface SizerMaxScaled {
  scale_factor?: number
  vol_daily?: number
  binding_constraint?: string
  exposures?: SizerExposures
  weights_df?: Record<string, unknown>[]
  [key: string]: unknown
}

interface SizerResponse {
  vol_daily?: number
  gross_leverage?: number
  equity_net?: number
  beta_hedge_mode?: BetaHedgeMode
  net_beta_spy?: number
  net_beta_iwm?: number
  net_beta_qqq?: number
  post_hedge_beta_spy?: number
  post_hedge_beta_iwm?: number
  post_hedge_beta_qqq?: number
  beta_scope?: string
  beta_asset_classes?: string[]
  beta_tickers?: string[]
  hedge_tickers?: string[]
  selected_hedges?: string[]
  net_betas?: Record<string, number>
  post_hedge_betas?: Record<string, number>
  exposures?: SizerExposures
  constraints?: Record<string, SizerConstraint>
  weights_df?: Record<string, unknown>[]
  hedges_df?: Record<string, unknown>[]
  hedge_spy_weight?: number
  hedge_iwm_weight?: number
  hedge_qqq_weight?: number
  hedge_weights?: Record<string, number>
  hedge_direction_warning?: string | null
  hedge_direction_issues?: string[]
  max_scaled?: SizerMaxScaled
  [key: string]: unknown
}

interface SizerRow {
  id: string
  ticker: string
  direction: string
  conviction: number
  groupName: string | null
  groupConviction: number | null
}

const SIZER_STATE_KEY = ["portfolio-sizer", "state", "equity-beta-v5"] as const
const DEFAULT_BOOK_SIZE = 100_000
const DEFAULT_BETA_HEDGE_MODE: BetaHedgeMode = "spy_iwm"
const DEFAULT_HEDGE_TICKERS = ["SPY", "IWM"]
const MIN_BOOK_SIZE = 10_000
const MAX_BOOK_SIZE = 10_000_000
const SIZER_POLL_INTERVAL_MS = 2_000
const SIZER_TABS: SizerTab[] = ["Weights", "Exposures", "Constraints", "Max Scaled"]
const EXPOSURE_CLASSES: ExposureAssetClass[] = ["equity", "fx", "commodity", "bond"]
const GROSS_LIMITS: Record<ExposureAssetClass, number> = {
  equity: 4.0,
  fx: 2.0,
  commodity: 1.0,
  bond: 3.0,
}
const HEDGE_TICKER_PATTERN = /^[A-Z0-9^][A-Z0-9.^=_-]{0,31}$/
const HEDGE_TICKERS: PresetHedgeTicker[] = ["SPY", "IWM", "QQQ"]
const HEDGE_MODE_TO_TICKERS: Record<BetaHedgeMode, string[]> = {
  spy: ["SPY"],
  iwm: ["IWM"],
  qqq: ["QQQ"],
  spy_iwm: ["SPY", "IWM"],
  spy_qqq: ["SPY", "QQQ"],
  iwm_qqq: ["IWM", "QQQ"],
  spy_iwm_qqq: ["SPY", "IWM", "QQQ"],
}
const ALWAYS_HIDDEN_COLUMNS = ["direction_intended", "days_since_new_low"] as const
const BASIC_WEIGHT_COLUMN_ORDER = [
  "ticker",
  "asset",
  "price",
  "quantity",
  "weight",
  "direction",
  "group_name",
  "dollar_weight",
  "target_quantity",
  "group_raw_target",
] as const
const BASIC_WEIGHT_COLUMN_ALIASES: Partial<Record<typeof BASIC_WEIGHT_COLUMN_ORDER[number], string[]>> = {
  quantity: ["shares"],
}

const STATUS_CLASSES: Record<"healthy" | "moderate" | "near", string> = {
  healthy: "bg-emerald-50 text-emerald-700 border border-emerald-200",
  moderate: "bg-amber-50 text-amber-700 border border-amber-200",
  near: "bg-red-50 text-red-700 border border-red-200",
}

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
})

const priceFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const COLUMN_LABELS: Record<string, string> = {
  index: "#",
  ticker: "Ticker",
  asset: "Asset",
  direction: "Direction",
  contrarian: "Contrarian",
  conviction: "Conviction",
  group_name: "Group",
  group_conviction: "Group Conviction",
  group_raw_target: "Group Raw Target",
  group_member_share: "Group Share",
  drawdown_52w: "Drawdown 52W",
  stabilized_10d: "Stabilized",
  beta_spy: "Equity Beta SPY",
  beta_iwm: "Equity Beta IWM",
  beta_qqq: "Equity Beta QQQ",
  realized_vol: "Vol",
  weight: "Weight",
  dollar_weight: "Dollar",
  price: "Price",
  instrument_type: "Instrument",
  price_symbol: "Price Symbol",
  contract_multiplier: "Multiplier",
  quantity: "Quantity",
  target_quantity: "Target Qty",
  contracts: "Contracts",
  shares: "Quantity",
  type: "Type",
}
const TICKER_SOURCE_KEYS = ["ticker", "Ticker", "symbol", "Symbol"] as const

function betaColumnLabel(key: string) {
  if (!key.startsWith("beta_")) return null
  return `Equity Beta ${key.slice("beta_".length).toUpperCase()}`
}

function makeRow(ticker = "", direction = "", conviction = 3, groupName?: string | null, groupConviction?: number | null): SizerRow {
  return {
    id: `row-${Math.random().toString(36).slice(2, 10)}`,
    ticker,
    direction,
    conviction,
    groupName: normalizeGroupName(groupName),
    groupConviction: normalizeGroupName(groupName) ? normalizeGroupConviction(groupConviction) ?? conviction : null,
  }
}

function sizerGroupState(rows: SizerRow[]) {
  const groups = new Map<string, { key: string; name: string; conviction: number; direction: string; ids: string[]; tickers: string[] }>()
  const errors: string[] = []
  for (const row of rows) {
    const name = normalizeGroupName(row.groupName)
    const key = groupKey(name)
    if (!key || !name) continue
    const conviction = normalizeGroupConviction(row.groupConviction)
    if (conviction == null) {
      errors.push(`Group ${name} requires a conviction.`)
      continue
    }
    const existing = groups.get(key)
    if (!existing) {
      groups.set(key, { key, name, conviction, direction: row.direction, ids: [row.id], tickers: [row.ticker] })
      continue
    }
    existing.ids.push(row.id)
    existing.tickers.push(row.ticker)
    if (existing.conviction !== conviction) errors.push(`Group ${existing.name} has inconsistent convictions.`)
    if (existing.direction && row.direction && existing.direction !== row.direction) {
      errors.push(`Group ${existing.name} cannot mix ${existing.direction} and ${row.direction} positions.`)
    }
  }
  return { groups, errors: Array.from(new Set(errors)) }
}

function toRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value.filter((row): row is Record<string, unknown> => row != null && typeof row === "object")
}

function cleanTickerValue(value: unknown, allowNumeric = true): string | null {
  if (value == null) return null
  const ticker = String(value).trim().toUpperCase()
  if (!ticker) return null
  if (!allowNumeric && /^\d+$/.test(ticker)) return null
  return ticker
}

function rowTicker(row: Record<string, unknown>): string | null {
  for (const key of TICKER_SOURCE_KEYS) {
    const ticker = cleanTickerValue(row[key])
    if (ticker) return ticker
  }
  return cleanTickerValue(row.index, false)
}

function rowsWithTickerColumn(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  return rows.map(row => {
    const ticker = rowTicker(row)
    if (!ticker || cleanTickerValue(row.ticker) === ticker) return row
    return { ...row, ticker }
  })
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function firstNumber(...values: unknown[]) {
  for (const value of values) {
    const num = toNumber(value)
    if (num != null) return num
  }
  return null
}

function betaHedgeModeToTickers(mode: BetaHedgeMode): string[] {
  return HEDGE_MODE_TO_TICKERS[mode] ?? HEDGE_MODE_TO_TICKERS[DEFAULT_BETA_HEDGE_MODE]
}

function tickersToBetaHedgeMode(tickers: string[]): BetaHedgeMode {
  const selected = [...tickers].sort()
  const match = Object.entries(HEDGE_MODE_TO_TICKERS).find(([, modeTickers]) => {
    const normalized = [...modeTickers].sort()
    return normalized.length === selected.length && normalized.every((ticker, idx) => ticker === selected[idx])
  })
  if (match) return match[0] as BetaHedgeMode
  return DEFAULT_BETA_HEDGE_MODE
}

function normalizeHedgeTicker(value: string): string | null {
  const ticker = value.trim().toUpperCase()
  if (!ticker || !HEDGE_TICKER_PATTERN.test(ticker)) return null
  return ticker
}

function normalizeHedgeTickers(values: string[] | undefined | null): string[] {
  const normalized: string[] = []
  const seen = new Set<string>()
  for (const value of values ?? []) {
    const ticker = normalizeHedgeTicker(String(value ?? ""))
    if (ticker && !seen.has(ticker)) {
      seen.add(ticker)
      normalized.push(ticker)
    }
  }
  return normalized.length > 0 ? normalized : [...DEFAULT_HEDGE_TICKERS]
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function clampBookSize(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_BOOK_SIZE
  return Math.min(MAX_BOOK_SIZE, Math.max(MIN_BOOK_SIZE, Math.round(value)))
}

function isPercentColumn(key: string) {
  const normalized = key.toLowerCase()
  if (isCurrencyColumn(normalized)) return false
  return normalized.includes("weight") || normalized.includes("pct") || normalized.includes("percent")
}

function isIntegerColumn(key: string) {
  const normalized = key.toLowerCase()
  return (
    normalized === "shares" ||
    normalized === "quantity" ||
    normalized === "target_quantity" ||
    normalized === "contracts" ||
    normalized === "conviction" ||
    normalized === "group_conviction"
  )
}

function isCurrencyColumn(key: string) {
  const normalized = key.toLowerCase()
  return (
    normalized === "price" ||
    normalized.includes("usd") ||
    normalized.includes("dollar") ||
    normalized.includes("notional") ||
    normalized.includes("amount") ||
    normalized.includes("value") ||
    normalized.includes("book")
  )
}

function formatPercent(value: number) {
  const pct = Math.abs(value) <= 1 ? value * 100 : value
  return `${pct >= 0 ? "+" : ""}${numberFormatter.format(pct)}%`
}

function formatRatioPercent(value: number, signed = true, precision = 1) {
  const pct = value * 100
  const sign = signed && pct >= 0 ? "+" : ""
  return `${sign}${pct.toFixed(precision)}%`
}

function resolveBasicWeightColumns(rows: Record<string, unknown>[]) {
  const availableKeys = new Set(rows.flatMap(row => Object.keys(row)))
  return BASIC_WEIGHT_COLUMN_ORDER.flatMap(key => {
    if (availableKeys.has(key)) return [key]
    const alias = BASIC_WEIGHT_COLUMN_ALIASES[key]?.find(aliasKey => availableKeys.has(aliasKey))
    return alias ? [alias] : []
  })
}

function buildCols(rows: Record<string, unknown>[], hiddenKeys?: readonly string[], visibleKeys?: readonly string[]): ColumnDef[] {
  if (rows.length === 0) return []
  const hidden = new Set(hiddenKeys ?? [])
  const availableKeys = Array.from(new Set(rows.flatMap(row => Object.keys(row))))
    .filter(k => k !== "index" && k !== "Ticker" && k !== "symbol" && k !== "Symbol" && !hidden.has(k))
  const keys = visibleKeys
    ? visibleKeys.filter((key, index) => availableKeys.includes(key) && visibleKeys.indexOf(key) === index)
    : availableKeys
  const tickerIndex = keys.indexOf("ticker")
  if (tickerIndex > 0) {
    const [tickerKey] = keys.splice(tickerIndex, 1)
    if (tickerKey) keys.unshift(tickerKey)
  }

  return keys.map(k => ({
    key: k,
    header: COLUMN_LABELS[k] ?? betaColumnLabel(k) ?? k,
    colorFn: isPercentColumn(k) ? colorPositiveNegative : undefined,
    format: (v: unknown) => {
      if (typeof v !== "number") return String(v ?? "N/A")
      if (k === "weight") return formatRatioPercent(v, true, 2)
      if (isPercentColumn(k)) return formatPercent(v)
      if (k === "price") return priceFormatter.format(v)
      if (isCurrencyColumn(k)) return currencyFormatter.format(v)
      if (isIntegerColumn(k)) return Math.round(v).toLocaleString("en-US")
      return `${v >= 0 ? "+" : ""}${numberFormatter.format(v)}`
    },
  }))
}

const CONVICTION_LABELS: Record<number, string> = {
  1: "Very Low",
  2: "Low",
  3: "Medium",
  4: "High",
  5: "Very High",
}

export function PortfolioSizer() {
  const queryClient = useQueryClient()
  const cachedState = queryClient.getQueryData<{
    bookSize: number
    bookSizeInput: string
    targetLeverage: number
    betaHedgeMode?: BetaHedgeMode
    hedgeTickers?: string[]
    rows: SizerRow[]
    result: SizerResponse | null
    activeJobId: string | null
    errorMessage: string | null
  }>(SIZER_STATE_KEY)

  const hasCachedBookSize = cachedState?.bookSize != null
  const hasCachedRows = Boolean(cachedState?.rows && cachedState.rows.length > 0)
  const initialBookSize = clampBookSize(cachedState?.bookSize ?? DEFAULT_BOOK_SIZE)
  const initialHedgeTickers = normalizeHedgeTickers(cachedState?.hedgeTickers ?? betaHedgeModeToTickers(cachedState?.betaHedgeMode ?? DEFAULT_BETA_HEDGE_MODE))
  const [bookSize, setBookSize] = useState(initialBookSize)
  const [bookSizeInput, setBookSizeInput] = useState(cachedState?.bookSizeInput ?? String(initialBookSize))
  const [targetLeverage, setTargetLeverage] = useState(cachedState?.targetLeverage ?? 2.0)
  const [hedgeTickers, setHedgeTickers] = useState<string[]>(initialHedgeTickers)
  const [customHedgeTicker, setCustomHedgeTicker] = useState("")
  const [customHedgeError, setCustomHedgeError] = useState<string | null>(null)
  const [rows, setRows] = useState<SizerRow[]>(cachedState?.rows && cachedState.rows.length > 0 ? cachedState.rows : [])
  const [cachedResult, setCachedResult] = useState<SizerResponse | null>(cachedState?.result ?? null)
  const [activeJobId, setActiveJobId] = useState<string | null>(cachedState?.activeJobId ?? null)
  const [isRunning, setIsRunning] = useState(Boolean(cachedState?.activeJobId))
  const [errorMessage, setErrorMessage] = useState<string | null>(cachedState?.errorMessage ?? null)
  const [tab, setTab] = useState<SizerTab>("Weights")
  const [weightsViewMode, setWeightsViewMode] = useState<WeightsViewMode>("basic")
  const [showInfo, setShowInfo] = useState(false)
  const [currentHoldings, setCurrentHoldings] = useState<Record<string, number>>({})
  const runSeqRef = useRef(0)

  useEffect(() => {
    let canceled = false
    fetchSizerPrefill()
      .then(data => {
        if (canceled) return
        const configuredBookSize = toNumber(data.book_size)
        if (!hasCachedBookSize && configuredBookSize != null) {
          const nextBookSize = clampBookSize(configuredBookSize)
          setBookSize(nextBookSize)
          setBookSizeInput(String(nextBookSize))
        }

        if (!hasCachedRows) {
          const positions = data.positions ?? []
          const prefilled = positions
            .map(p => ({
              ticker: String(p?.ticker ?? "").trim().toUpperCase(),
              direction: String(p?.direction ?? "").trim().toLowerCase(),
              conviction: typeof p?.conviction === "number" ? p.conviction : 3,
              groupName: normalizeGroupName(p?.group_name),
              groupConviction: normalizeGroupConviction(p?.group_conviction),
            }))
            .filter(p => p.ticker.length > 0)
            .map(p => makeRow(p.ticker, p.direction, p.conviction, p.groupName, p.groupConviction))

          if (prefilled.length > 0) setRows(prefilled)
        }
      })
      .catch(() => { })

    return () => { canceled = true }
  }, [hasCachedBookSize, hasCachedRows])

  const refreshCurrentHoldings = useCallback(() => {
    fetchPortfolioPositions(true)
      .then(({ positions }) => {
        const map: Record<string, number> = {}
        for (const p of positions) {
          const quantity = p.quantity ?? p.shares
          if (quantity != null) map[p.ticker.toUpperCase()] = quantity
        }
        setCurrentHoldings(map)
      })
      .catch(() => { })
  }, [])

  const handleSizerResult = useCallback((result: unknown) => {
    setCachedResult((result as SizerResponse) ?? null)
    setActiveJobId(null)
    setIsRunning(false)
    setErrorMessage(null)
    refreshCurrentHoldings()
  }, [refreshCurrentHoldings])

  useEffect(() => {
    if (!activeJobId) return

    const jobId = activeJobId
    let canceled = false
    let timeoutId: ReturnType<typeof setTimeout> | undefined

    async function pollJob() {
      try {
        const job = await fetchSizerJob(jobId)
        if (canceled) return

        if (job.status === "done") {
          handleSizerResult("result" in job ? job.result : undefined)
          return
        }
        if (job.status === "error") {
          setActiveJobId(null)
          setIsRunning(false)
          setErrorMessage(job.error || "Sizer failed")
          return
        }

        setIsRunning(true)
        timeoutId = setTimeout(pollJob, SIZER_POLL_INTERVAL_MS)
      } catch (error) {
        if (canceled) return
        setActiveJobId(null)
        setIsRunning(false)
        setErrorMessage(error instanceof Error ? error.message : String(error))
      }
    }

    setIsRunning(true)
    setErrorMessage(null)
    pollJob()

    return () => {
      canceled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [activeJobId, handleSizerResult])

  useEffect(() => {
    queryClient.setQueryData(SIZER_STATE_KEY, {
      bookSize,
      bookSizeInput,
      targetLeverage,
      betaHedgeMode: tickersToBetaHedgeMode(hedgeTickers),
      hedgeTickers,
      rows,
      result: cachedResult,
      activeJobId,
      errorMessage,
    })
  }, [bookSize, bookSizeInput, targetLeverage, hedgeTickers, rows, cachedResult, activeJobId, errorMessage, queryClient])

  useEffect(() => {
    setBookSizeInput(String(bookSize))
  }, [bookSize])

  function updateConviction(id: string, conviction: number) {
    setRows(prev => prev.map(row => (row.id === id ? { ...row, conviction } : row)))
  }

  function updateGroupName(id: string, value: string) {
    setRows(prev => {
      const target = prev.find(row => row.id === id)
      const name = normalizeGroupName(value)
      if (!target || !name) {
        return prev.map(row => (row.id === id ? { ...row, groupName: null, groupConviction: null } : row))
      }
      const key = groupKey(name)
      const existing = prev.find(row => row.id !== id && groupKey(row.groupName) === key)
      const groupName = normalizeGroupName(existing?.groupName) ?? name
      const groupConviction = normalizeGroupConviction(existing?.groupConviction) ?? normalizeGroupConviction(target.groupConviction) ?? target.conviction
      return prev.map(row => (row.id === id ? { ...row, groupName, groupConviction } : row))
    })
  }

  function updateGroupConviction(group: string | null | undefined, conviction: number) {
    const key = groupKey(group)
    if (!key) return
    setRows(prev => prev.map(row => (groupKey(row.groupName) === key ? { ...row, groupConviction: conviction } : row)))
  }

  function toggleHedgeTicker(ticker: string) {
    const selected = hedgeTickers
    const isSelected = selected.includes(ticker)
    if (isSelected && selected.length === 1) return

    const next = isSelected
      ? selected.filter(t => t !== ticker)
      : [...selected, ticker]
    setHedgeTickers(normalizeHedgeTickers(next))
  }

  function removeHedgeTicker(ticker: string) {
    if (hedgeTickers.length === 1) return
    setHedgeTickers(prev => normalizeHedgeTickers(prev.filter(t => t !== ticker)))
  }

  function addCustomHedgeTicker() {
    const ticker = normalizeHedgeTicker(customHedgeTicker)
    if (!ticker) {
      setCustomHedgeError("Enter a valid ticker symbol.")
      return
    }
    setHedgeTickers(prev => normalizeHedgeTickers([...prev, ticker]))
    setCustomHedgeTicker("")
    setCustomHedgeError(null)
  }

  async function handleRun() {
    const parsedBook = Number(bookSizeInput)
    const effectiveBook = Number.isFinite(parsedBook) ? clampBookSize(parsedBook) : bookSize
    setBookSize(effectiveBook)
    setBookSizeInput(String(effectiveBook))

    const groupState = sizerGroupState(rows)
    if (groupState.errors.length > 0) {
      setErrorMessage(groupState.errors[0])
      return
    }

    const positions = rows
      .filter(r => r.ticker.trim().length > 0)
      .map(r => ({
        ticker: r.ticker.trim().toUpperCase(),
        conviction: r.conviction,
        group_name: normalizeGroupName(r.groupName),
        group_conviction: normalizeGroupName(r.groupName) ? normalizeGroupConviction(r.groupConviction) : null,
      }))

    if (positions.length === 0) return

    const runSeq = runSeqRef.current + 1
    runSeqRef.current = runSeq
    setIsRunning(true)
    setErrorMessage(null)

    try {
      const started = await startSizerJob({
        book: effectiveBook,
        target_leverage: targetLeverage,
        beta_hedge_mode: tickersToBetaHedgeMode(hedgeTickers),
        hedge_tickers: hedgeTickers,
        positions,
      })
      if (runSeq !== runSeqRef.current) return

      if (started.status === "done") {
        handleSizerResult("result" in started ? started.result : undefined)
        return
      }
      if (started.status === "error") {
        setActiveJobId(null)
        setIsRunning(false)
        setErrorMessage(started.error || "Sizer failed")
        return
      }

      setActiveJobId(started.job_id)
      setIsRunning(true)
    } catch (error) {
      if (runSeq !== runSeqRef.current) return
      setActiveJobId(null)
      setIsRunning(false)
      setErrorMessage(error instanceof Error ? error.message : String(error))
    }
  }

  const data = cachedResult
  const weightsRows = rowsWithTickerColumn(toRows(data?.weights_df))
  const weightsVisibleKeys = weightsViewMode === "basic" ? resolveBasicWeightColumns(weightsRows) : undefined
  const weightsHiddenKeys = weightsViewMode === "advanced" ? ALWAYS_HIDDEN_COLUMNS : undefined
  const hedgesRows = rowsWithTickerColumn(toRows(data?.hedges_df))
  const exposures = data?.exposures ?? {}
  const constraints = data?.constraints ?? {}
  const maxScaled = data?.max_scaled
  const maxScaledRows = rowsWithTickerColumn(toRows(maxScaled?.weights_df))
  const maxScaledExposures = maxScaled?.exposures ?? {}
  const selectedHedgeTickers = hedgeTickers
  const hedgeModeLabel = selectedHedgeTickers.join(" + ")

  const volDaily = firstNumber(data?.vol_daily)
  const grossLeverage = firstNumber(data?.gross_leverage)
  const hedgeGross = firstNumber(exposures.hedge_gross, 0) ?? 0
  const hedgeDirectionIssues = Array.from(new Set([
    ...(Array.isArray(data?.hedge_direction_issues) ? data.hedge_direction_issues.filter(v => typeof v === "string") : []),
  ]))
  const hedgeDirectionWarning = typeof data?.hedge_direction_warning === "string" && data.hedge_direction_warning.trim().length > 0
    ? data.hedge_direction_warning
    : hedgeDirectionIssues.length > 0
      ? "Potential hedge direction mismatch detected."
      : null
  const equityNet = firstNumber(exposures.equity_net, data?.equity_net)
  const betaMetricCards = Array.from(new Set([
    ...(Array.isArray(data?.selected_hedges) ? data.selected_hedges : []),
    ...(Array.isArray(data?.hedge_tickers) ? data.hedge_tickers : []),
    ...Object.keys(data?.net_betas ?? {}),
    ...Object.keys(data?.post_hedge_betas ?? {}),
  ])).flatMap(ticker => {
    const key = ticker.toLowerCase()
    const preHedge = firstNumber(data?.[`net_beta_${key}`], data?.net_betas?.[ticker])
    const postHedge = firstNumber(data?.[`post_hedge_beta_${key}`], data?.post_hedge_betas?.[ticker])
    return [
      ...(preHedge != null ? [{ key: `${ticker}-pre`, title: `Equity Beta ${ticker} (pre-hedge)`, value: preHedge.toFixed(3) }] : []),
      ...(postHedge != null ? [{ key: `${ticker}-post`, title: `Equity Beta ${ticker} (post-hedge)`, value: postHedge.toFixed(3) }] : []),
    ]
  })
  const showHeaderMetrics = [
    volDaily,
    grossLeverage,
    equityNet,
  ].some(v => v != null) || betaMetricCards.length > 0
  const groupState = sizerGroupState(rows)

  const sizingDeltas = useMemo(() => {
    if (!data) return []
    const allRows = [...weightsRows, ...hedgesRows]
    if (allRows.length === 0) return []

    const deltaList: {
      ticker: string
      type: string
      direction: string
      currentShares: number
      targetShares: number
      delta: number
      price: number
      notional: number
    }[] = []

    for (const row of allRows) {
      const ticker = String(row.ticker ?? "").trim().toUpperCase()
      if (!ticker) continue
      const targetShares = toNumber(row.target_quantity) ?? toNumber(row.contracts) ?? toNumber(row.quantity) ?? toNumber(row.shares) ?? 0
      const price = toNumber(row.price) ?? 0
      const multiplier = toNumber(row.contract_multiplier) ?? 1
      const currentShares = currentHoldings[ticker] ?? 0
      const delta = targetShares - currentShares
      const type = row.type === "hedge" ? "Hedge" : "Position"
      const direction = String(row.direction ?? "")

      deltaList.push({
        ticker,
        type,
        direction,
        currentShares: Math.round(currentShares),
        targetShares: Math.round(targetShares),
        delta: Math.round(delta),
        price,
        notional: Math.round(delta) * price * multiplier,
      })
    }

    return deltaList.sort((a, b) => Math.abs(b.notional) - Math.abs(a.notional))
  }, [data, weightsRows, hedgesRows, currentHoldings])

  const totalNotionalDelta = sizingDeltas.reduce((sum, t) => sum + Math.abs(t.notional), 0)
  const increaseNotional = sizingDeltas.filter(t => t.delta > 0).reduce((sum, t) => sum + t.notional, 0)
  const decreaseNotional = sizingDeltas.filter(t => t.delta < 0).reduce((sum, t) => sum + Math.abs(t.notional), 0)

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Sizer</h1>
          <button
            onClick={() => setShowInfo(v => !v)}
            className="text-gray-300 hover:text-gray-500 transition-colors"
            title="What is this?"
          >
            <Info size={16} />
          </button>
        </div>
        <p className="text-sm text-gray-400 mt-0.5">Conviction-based portfolio sizing with constraint optimization and equity beta hedging</p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <DecisionStateBadge state={String(data?.decision_state ?? "analysis")} />
          <EffectScopeBadge scope={String(data?.effect_scope ?? "read_only")} />
          <QualityStateBadge state={String(data?.quality_state ?? "ok")} />
          <span className="text-xs text-gray-500">Decision support only. Sizing deltas are not orders and are not sent to a broker.</span>
        </div>
        {showInfo && (
          <p className="text-xs text-gray-500 mt-2 max-w-xl leading-relaxed">
            Converts ticker convictions and optional group convictions into target weights, quantities, hedge legs, and
            constraint utilization using book size, gross leverage, realized volatility, and the selected equity beta
            basket. Outputs are decision support only; sizing deltas estimate trade impact but do not submit orders.
          </p>
        )}
      </div>

      <div className="rounded-xl border border-gray-200/80 bg-white p-5 mb-6 space-y-5">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-x-8 gap-y-5">
          <div className="space-y-1">
            <TextInput
              label="Book Size"
              type="number"
              value={bookSizeInput}
              onChange={setBookSizeInput}
              placeholder="100000"
            />
            <p className="text-xs text-gray-400">$10k - $10m · used for analysis run</p>
          </div>

          <SliderInput
            label="Target Gross Leverage"
            value={targetLeverage}
            onChange={setTargetLeverage}
            min={0.5}
            max={4.0}
            step={0.05}
            formatValue={v => `${v.toFixed(2)}x`}
            minLabel="0.5x"
            maxLabel="4.0x"
          />

          <div className="space-y-2">
            <div className="flex items-baseline justify-between text-sm text-muted">
              <span>Hedge Basket</span>
              <span className="text-sm font-semibold text-app">
                {hedgeModeLabel}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {HEDGE_TICKERS.map(ticker => {
                const selected = selectedHedgeTickers.includes(ticker)
                const disabled = selected && selectedHedgeTickers.length === 1

                return (
                  <button
                    key={ticker}
                    type="button"
                    onClick={() => toggleHedgeTicker(ticker)}
                    disabled={disabled}
                    aria-pressed={selected}
                    className={cn(
                      "h-10 rounded-lg border px-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-45",
                      selected
                        ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent-muted))] text-app shadow-sm"
                        : "border-app bg-card-muted text-muted hover:border-[hsl(var(--accent)/0.45)] hover:text-app",
                    )}
                  >
                    {ticker}
                  </button>
                )
              })}
            </div>
            <div className="flex flex-wrap gap-2">
              {selectedHedgeTickers.map(ticker => {
                const disabled = selectedHedgeTickers.length === 1
                return (
                  <span
                    key={ticker}
                    className="inline-flex h-8 items-center gap-1 rounded-lg border border-app bg-card-muted px-2 text-xs font-semibold text-app"
                  >
                    {ticker}
                    <button
                      type="button"
                      onClick={() => removeHedgeTicker(ticker)}
                      disabled={disabled}
                      title={disabled ? "At least one hedge ticker is required" : `Remove ${ticker}`}
                      className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted transition hover:bg-[hsl(var(--muted-3))] hover:text-app disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      <X size={12} />
                    </button>
                  </span>
                )
              })}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={customHedgeTicker}
                onChange={e => {
                  setCustomHedgeTicker(e.target.value)
                  setCustomHedgeError(null)
                }}
                onKeyDown={e => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    addCustomHedgeTicker()
                  }
                }}
                placeholder="SMH"
                className="theme-input h-10 flex-1 text-sm"
                aria-label="Custom hedge ticker"
              />
              <button
                type="button"
                onClick={addCustomHedgeTicker}
                title="Add custom hedge ticker"
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-app bg-card-muted text-muted transition hover:border-[hsl(var(--accent)/0.45)] hover:text-app"
              >
                <Plus size={16} />
              </button>
            </div>
            {customHedgeError && <p className="text-xs text-red-600">{customHedgeError}</p>}
            <p className="text-xs text-gray-400">
              Neutralizes and reports beta against the selected hedge tickers.
            </p>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-gray-700">Conviction Levels</p>
          </div>
          <p className="text-xs text-gray-400 mb-3">
            Set conviction (1–5) for each position. Higher conviction = larger allocation toward the position cap.
          </p>

          <div className="space-y-2">
            {rows.length === 0 && (
              <p className="text-sm text-gray-400">Loading portfolio tickers...</p>
            )}
            {rows.map((row, idx) => (
              <div key={row.id} className="grid gap-3 items-center" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                <div className="col-span-2">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Ticker</p>}
                  <span className="inline-flex w-full items-center justify-center rounded-lg border border-app bg-[hsl(var(--muted-2))] px-2 py-1.5 text-center text-sm font-mono text-app">
                    {row.ticker}
                  </span>
                </div>
                <div className="col-span-2">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Direction</p>}
                  <span
                    className="inline-flex w-full items-center justify-center rounded-lg border px-2 py-1.5 text-center text-xs font-medium"
                    style={
                      row.direction === "long"
                        ? {
                          backgroundColor: "hsl(var(--success-muted))",
                          color: "hsl(var(--success))",
                          borderColor: "hsl(var(--success) / 0.25)",
                        }
                        : row.direction === "short"
                          ? {
                            backgroundColor: "hsl(var(--destructive-muted))",
                            color: "hsl(var(--destructive))",
                            borderColor: "hsl(var(--destructive) / 0.25)",
                          }
                          : {
                            backgroundColor: "hsl(var(--background-card-muted))",
                            color: "hsl(var(--foreground-secondary))",
                            borderColor: "hsl(var(--border))",
                          }
                    }
                  >
                    {row.direction || "—"}
                  </span>
                </div>
                <div className="col-span-3">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Group</p>}
                  <input
                    type="text"
                    value={row.groupName ?? ""}
                    onChange={e => updateGroupName(row.id, e.target.value)}
                    placeholder="Optional"
                    className="theme-input w-full text-sm"
                  />
                </div>
                <div className="col-span-3">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Group Conv.</p>}
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={normalizeGroupConviction(row.groupConviction) ?? row.conviction}
                    onChange={e => updateGroupConviction(row.groupName, Number(e.target.value))}
                    className="hig-slider w-full cursor-pointer"
                    style={{ accentColor: "hsl(var(--accent))" }}
                    disabled={!normalizeGroupName(row.groupName)}
                  />
                  <span className="text-xs text-muted">
                    {normalizeGroupName(row.groupName)
                      ? `${normalizeGroupConviction(row.groupConviction) ?? row.conviction} — ${CONVICTION_LABELS[normalizeGroupConviction(row.groupConviction) ?? row.conviction] ?? ""}`
                      : "Ungrouped"}
                  </span>
                </div>
                <div className="col-span-4">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Conviction</p>}
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={row.conviction}
                    onChange={e => updateConviction(row.id, Number(e.target.value))}
                    className="hig-slider w-full cursor-pointer"
                    style={{ accentColor: "hsl(var(--accent))" }}
                  />
                </div>
                <div className="col-span-2">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Level</p>}
                  <span className="text-sm font-medium text-app">
                    {row.conviction} — {CONVICTION_LABELS[row.conviction] ?? ""}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {groupState.errors.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            {groupState.errors[0]}
          </div>
        )}

        <ActionButton onClick={handleRun} loading={isRunning} loadingText="Sizing portfolio..." disabled={groupState.errors.length > 0}>
          Size Portfolio
        </ActionButton>
      </div>

      {isRunning && <LoadingSpinner message="Running portfolio sizer..." />}
      {errorMessage && <ErrorMessage message={errorMessage} />}

      {data && !isRunning && (
        <div className="space-y-6">
          {hedgeDirectionWarning && (
            <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3">
              <p className="text-sm font-semibold text-amber-900">Hedge Direction Warning</p>
              <p className="mt-1 text-sm text-amber-800">{hedgeDirectionWarning}</p>
              {hedgeDirectionIssues.length > 0 && (
                <ul className="mt-2 list-disc pl-5 text-sm text-amber-800 space-y-1">
                  {hedgeDirectionIssues.map(issue => (
                    <li key={issue}>{issue}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {showHeaderMetrics && (
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
              {volDaily != null && <MetricCard title="Daily Volatility" value={`${(volDaily * 100).toFixed(2)}%`} />}
              {grossLeverage != null && <MetricCard title="Gross Leverage (incl. hedges)" value={`${grossLeverage.toFixed(2)}x`} />}
              {equityNet != null && <MetricCard title="Equity Net" value={formatRatioPercent(equityNet, true, 1)} />}
              {betaMetricCards.map(card => (
                <MetricCard key={card.key} title={card.title} value={card.value} />
              ))}
            </div>
          )}

          <div className="mb-2">
            <SegmentedControl
              options={SIZER_TABS.map(t => ({ value: t, label: t }))}
              value={tab}
              onChange={setTab}
            />
          </div>

          {tab === "Weights" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-gray-600">Table View</p>
                <SegmentedControl
                  options={[
                    { value: "basic", label: "Basic" },
                    { value: "advanced", label: "Advanced" },
                  ]}
                  value={weightsViewMode}
                  onChange={setWeightsViewMode}
                  size="sm"
                />
              </div>

              {weightsRows.length > 0 && (
                <DataTable
                  label="Portfolio Weights"
                  columns={buildCols(weightsRows, weightsHiddenKeys, weightsVisibleKeys)}
                  rows={weightsRows}
                />
              )}

              {hedgesRows.length > 0 && (
                <DataTable label="Computed Hedge Legs" columns={buildCols(hedgesRows)} rows={hedgesRows} />
              )}

              {weightsRows.length === 0 && hedgesRows.length === 0 && (
                <p className="text-gray-400 text-sm">No weights or hedge rows returned.</p>
              )}
            </div>
          )}

          {tab === "Exposures" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="rounded-xl border border-gray-200 p-4 space-y-3">
                  <h2 className="text-base font-semibold">Gross Exposures</h2>
                  {EXPOSURE_CLASSES.map(assetClass => {
                    const gross = firstNumber(exposures[`${assetClass}_gross`], 0) ?? 0
                    const maxLimit = GROSS_LIMITS[assetClass]
                    const utilization = clamp01(gross / maxLimit)

                    return (
                      <div key={assetClass} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium text-gray-700 capitalize">{assetClass}</span>
                          <span className="text-gray-500">
                            {formatRatioPercent(gross, false, 1)} / {formatRatioPercent(maxLimit, false, 0)}
                          </span>
                        </div>
                        <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                          <div
                            className="h-full bg-gray-800 transition-all"
                            style={{ width: `${(utilization * 100).toFixed(1)}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                  <div className="pt-2">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <MetricCard
                        title="Hedge Gross"
                        value={formatRatioPercent(hedgeGross, false, 1)}
                      />
                      <MetricCard
                        title="Total Gross"
                        value={formatRatioPercent(firstNumber(exposures.total_gross, data.gross_leverage, 0) ?? 0, false, 1)}
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-gray-200 p-4 space-y-3">
                  <h2 className="text-base font-semibold">Net Exposures</h2>
                  <div className="grid grid-cols-2 gap-3">
                    {EXPOSURE_CLASSES.map(assetClass => {
                      const net = firstNumber(exposures[`${assetClass}_net`], 0) ?? 0
                      return (
                        <MetricCard
                          key={assetClass}
                          title={assetClass[0].toUpperCase() + assetClass.slice(1)}
                          value={formatRatioPercent(net, true, 1)}
                        />
                      )
                    })}
                    <MetricCard
                      title="Total Net"
                      value={formatRatioPercent(firstNumber(exposures.total_net, 0) ?? 0, true, 1)}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {tab === "Constraints" && (
            <div className="space-y-3">
              {Object.entries(constraints).length > 0 ? (
                Object.entries(constraints).map(([name, constraint]) => {
                  const utilization = clamp01(firstNumber(constraint?.utilization, 0) ?? 0)
                  const current = firstNumber(constraint?.current)
                  const limit = firstNumber(constraint?.limit)
                  const tone = utilization > 0.9 ? "near" : utilization > 0.7 ? "moderate" : "healthy"
                  const status = tone === "near" ? "Near Limit" : tone === "moderate" ? "Moderate" : "Healthy"

                  return (
                    <div key={name} className="rounded-xl border border-gray-200 p-4">
                      <div className="flex items-center justify-between mb-3 gap-3">
                        <h2 className="text-sm font-semibold text-gray-800">{name}</h2>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${STATUS_CLASSES[tone]}`}>{status}</span>
                      </div>
                      <div className="h-2 rounded-full bg-gray-100 overflow-hidden mb-2">
                        <div
                          className={`h-full transition-all ${tone === "near" ? "bg-red-500" : tone === "moderate" ? "bg-amber-500" : "bg-emerald-500"}`}
                          style={{ width: `${(utilization * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <p className="text-xs text-gray-500">
                        Current: {current != null ? formatRatioPercent(current, true, 1) : "N/A"} / Limit:{" "}
                        {limit != null ? formatRatioPercent(limit, true, 1) : "N/A"}
                      </p>
                    </div>
                  )
                })
              ) : (
                <p className="text-gray-400 text-sm">No constraint utilization data available.</p>
              )}
            </div>
          )}

          {tab === "Max Scaled" && (
            <div className="space-y-6">
              {maxScaled ? (
                <>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <MetricCard
                      title="Scale Factor"
                      value={firstNumber(maxScaled.scale_factor) != null ? `${(firstNumber(maxScaled.scale_factor) ?? 0).toFixed(4)}x` : "N/A"}
                    />
                    <MetricCard
                      title="Daily Volatility"
                      value={
                        firstNumber(maxScaled.vol_daily) != null
                          ? `${((firstNumber(maxScaled.vol_daily) ?? 0) * 100).toFixed(2)}%`
                          : "N/A"
                      }
                    />
                    <MetricCard title="Binding Constraint" value={String(maxScaled.binding_constraint ?? "N/A")} />
                  </div>

                  <div>
                    <h2 className="text-base font-semibold mb-2">Max Scaled Exposures</h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Gross"
                        value={formatRatioPercent(firstNumber(maxScaledExposures.total_gross, 0) ?? 0, false, 1)}
                      />
                      <MetricCard
                        title="Hedge Gross"
                        value={formatRatioPercent(firstNumber(maxScaledExposures.hedge_gross, 0) ?? 0, false, 1)}
                      />
                      <MetricCard
                        title="Equity Net"
                        value={formatRatioPercent(firstNumber(maxScaledExposures.equity_net, 0) ?? 0, true, 1)}
                      />
                      <MetricCard
                        title="FX Gross"
                        value={formatRatioPercent(firstNumber(maxScaledExposures.fx_gross, 0) ?? 0, false, 1)}
                      />
                      <MetricCard
                        title="Commodity Gross"
                        value={formatRatioPercent(firstNumber(maxScaledExposures.commodity_gross, 0) ?? 0, false, 1)}
                      />
                    </div>
                  </div>

                  <div>
                    {maxScaledRows.length > 0
                      ? <DataTable label="Max Scaled Weights" columns={buildCols(maxScaledRows)} rows={maxScaledRows} />
                      : <p className="text-gray-400 text-sm">No max-scaled weights returned.</p>}
                  </div>
                </>
              ) : (
                <p className="text-gray-400 text-sm">No max scaled data available.</p>
              )}
            </div>
          )}

          {sizingDeltas.length > 0 && (
            <div className="space-y-4">
              <div>
                <h2 className="text-base font-semibold text-gray-900">Sizing Delta Summary</h2>
                <p className="mt-1 text-xs text-gray-500">
                  Analysis only. These deltas compare current quantity with computed target quantity; they are not executable orders.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <MetricCard title="Increase Notional" value={currencyFormatter.format(increaseNotional)} />
                <MetricCard title="Decrease Notional" value={currencyFormatter.format(decreaseNotional)} />
                <MetricCard title="Total Absolute Delta" value={currencyFormatter.format(totalNotionalDelta)} />
              </div>

              <div className="rounded-xl border border-gray-200 overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead>
                    <tr className="border-b border-gray-100 bg-gray-50/60">
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Ticker</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Type</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Direction</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Current</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Target</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Quantity Delta</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Sizing Direction</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Price</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Notional Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sizingDeltas.map(t => {
                      const direction =
                        t.delta > 0 ? "Increase" : t.delta < 0 ? "Decrease" : "No change"
                      const directionColor =
                        direction === "Increase"
                          ? "text-emerald-700 bg-emerald-50"
                          : direction === "Decrease"
                            ? "text-red-700 bg-red-50"
                            : "text-gray-500 bg-gray-50"

                      return (
                        <tr key={t.ticker} className="border-b border-gray-50 last:border-0">
                          <td className="px-3 py-2 font-mono font-medium text-gray-900">{t.ticker}</td>
                          <td className="px-3 py-2 text-gray-500">{t.type}</td>
                          <td className="px-3 py-2 text-gray-500 capitalize">{t.direction}</td>
                          <td className="px-3 py-2 text-right font-mono text-gray-600">
                            {t.currentShares.toLocaleString("en-US")}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-gray-600">
                            {t.targetShares.toLocaleString("en-US")}
                          </td>
                          <td className="px-3 py-2 text-right font-mono font-medium" style={{ color: colorPositiveNegative(t.delta) }}>
                            {t.delta >= 0 ? "+" : ""}{t.delta.toLocaleString("en-US")}
                          </td>
                          <td className="px-3 py-2">
                            <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-semibold ${directionColor}`}>
                              {direction}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-gray-600">
                            {priceFormatter.format(t.price)}
                          </td>
                          <td className="px-3 py-2 text-right font-mono font-medium" style={{ color: colorPositiveNegative(t.notional) }}>
                            {t.notional >= 0 ? "+" : ""}{currencyFormatter.format(t.notional)}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {!data && !isRunning && !errorMessage && (
        <p className="text-gray-400 text-sm">Set conviction levels above and click Size Portfolio.</p>
      )}
    </div>
  )
}
