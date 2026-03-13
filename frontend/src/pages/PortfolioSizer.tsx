import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SegmentedControl, SliderInput, TextInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MetricCard } from "@/components/shared/MetricCard"
import { colorPositiveNegative } from "@/lib/colors"
import { fetchPortfolioPositions, fetchSizerPrefill, runPortfolioSizerAsync } from "@/lib/api"

type SizerTab = "Weights" | "Exposures" | "Constraints" | "Max Scaled"
type ExposureAssetClass = "equity" | "fx" | "commodity" | "bond"
type WeightsViewMode = "basic" | "advanced"

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
  net_beta_spy?: number
  net_beta_iwm?: number
  post_hedge_beta_spy?: number
  post_hedge_beta_iwm?: number
  exposures?: SizerExposures
  constraints?: Record<string, SizerConstraint>
  weights_df?: Record<string, unknown>[]
  hedges_df?: Record<string, unknown>[]
  hedge_spy_weight?: number
  hedge_iwm_weight?: number
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
}

const SIZER_STATE_KEY = ["portfolio-sizer", "state"] as const
const MIN_BOOK_SIZE = 10_000
const MAX_BOOK_SIZE = 10_000_000
const SIZER_TABS: SizerTab[] = ["Weights", "Exposures", "Constraints", "Max Scaled"]
const EXPOSURE_CLASSES: ExposureAssetClass[] = ["equity", "fx", "commodity", "bond"]
const GROSS_LIMITS: Record<ExposureAssetClass, number> = {
  equity: 4.0,
  fx: 2.0,
  commodity: 1.0,
  bond: 3.0,
}

const ALWAYS_HIDDEN_COLUMNS = ["direction_intended", "days_since_new_low"] as const
const BASIC_COLUMNS = new Set([
  "ticker",
  "asset",
  "direction",
  "beta_spy",
  "beta_iwm",
  "realized_vol",
  "weight",
  "price",
  "dollar_weight",
  "shares",
])

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
  distressed: "Distressed",
  conviction: "Conviction",
  drawdown_52w: "Drawdown 52W",
  stabilized_10d: "Stabilized",
  beta_spy: "Beta SPY",
  beta_iwm: "Beta IWM",
  realized_vol: "Vol",
  weight: "Weight",
  dollar_weight: "Dollar",
  price: "Price",
  shares: "Shares",
  type: "Type",
}

function makeRow(ticker = "", direction = "", conviction = 3): SizerRow {
  return {
    id: `row-${Math.random().toString(36).slice(2, 10)}`,
    ticker,
    direction,
    conviction,
  }
}

function toRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value.filter((row): row is Record<string, unknown> => row != null && typeof row === "object")
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

function getHedgeWeightFromRows(rows: Record<string, unknown>[], ticker: string): number | null {
  const row = rows.find(r => String(r.ticker ?? "").trim().toUpperCase() === ticker)
  return row ? toNumber(row.weight) : null
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function clampBookSize(value: number) {
  if (!Number.isFinite(value)) return MIN_BOOK_SIZE
  return Math.min(MAX_BOOK_SIZE, Math.max(MIN_BOOK_SIZE, Math.round(value)))
}

function isPercentColumn(key: string) {
  const normalized = key.toLowerCase()
  if (isCurrencyColumn(normalized)) return false
  return normalized.includes("weight") || normalized.includes("pct") || normalized.includes("percent")
}

function isIntegerColumn(key: string) {
  return key.toLowerCase() === "shares" || key.toLowerCase() === "conviction"
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

function buildCols(rows: Record<string, unknown>[], hiddenKeys?: string[]): ColumnDef[] {
  if (rows.length === 0) return []
  const hidden = new Set(hiddenKeys ?? [])
  return Object.keys(rows[0]).filter(k => k !== "index" && !hidden.has(k)).map(k => ({
    key: k,
    header: COLUMN_LABELS[k] ?? k,
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
    rows: SizerRow[]
    result: SizerResponse | null
  }>(SIZER_STATE_KEY)

  const [bookSize, setBookSize] = useState(cachedState?.bookSize ?? 100_000)
  const [bookSizeInput, setBookSizeInput] = useState(cachedState?.bookSizeInput ?? String(cachedState?.bookSize ?? 100_000))
  const [targetLeverage, setTargetLeverage] = useState(cachedState?.targetLeverage ?? 2.0)
  const [rows, setRows] = useState<SizerRow[]>(cachedState?.rows && cachedState.rows.length > 0 ? cachedState.rows : [])
  const [cachedResult, setCachedResult] = useState<SizerResponse | null>(cachedState?.result ?? null)
  const [tab, setTab] = useState<SizerTab>("Weights")
  const [weightsViewMode, setWeightsViewMode] = useState<WeightsViewMode>("basic")
  const [currentHoldings, setCurrentHoldings] = useState<Record<string, number>>({})

  useEffect(() => {
    if (cachedState?.rows && cachedState.rows.length > 0) return

    let canceled = false
    fetchSizerPrefill()
      .then((data: unknown) => {
        if (canceled) return
        const positions = (data as { positions?: Array<{ ticker?: unknown; conviction?: unknown; direction?: unknown }> })?.positions ?? []
        const prefilled = positions
          .map(p => ({
            ticker: String(p?.ticker ?? "").trim().toUpperCase(),
            direction: String(p?.direction ?? "").trim().toLowerCase(),
            conviction: typeof p?.conviction === "number" ? p.conviction : 3,
          }))
          .filter(p => p.ticker.length > 0)
          .map(p => makeRow(p.ticker, p.direction, p.conviction))

        if (prefilled.length > 0) setRows(prefilled)
      })
      .catch(() => {})

    return () => { canceled = true }
  }, [cachedState?.rows])

  const mutation = useMutation({
    mutationFn: runPortfolioSizerAsync,
    onSuccess: result => {
      setCachedResult((result as SizerResponse) ?? null)
      fetchPortfolioPositions(true)
        .then(({ positions }) => {
          const map: Record<string, number> = {}
          for (const p of positions) {
            if (p.shares != null) map[p.ticker.toUpperCase()] = p.shares
          }
          setCurrentHoldings(map)
        })
        .catch(() => {})
    },
  })

  useEffect(() => {
    queryClient.setQueryData(SIZER_STATE_KEY, {
      bookSize,
      bookSizeInput,
      targetLeverage,
      rows,
      result: cachedResult,
    })
  }, [bookSize, bookSizeInput, targetLeverage, rows, cachedResult, queryClient])

  useEffect(() => {
    setBookSizeInput(String(bookSize))
  }, [bookSize])

  function updateConviction(id: string, conviction: number) {
    setRows(prev => prev.map(row => (row.id === id ? { ...row, conviction } : row)))
  }

  function handleRun() {
    const parsedBook = Number(bookSizeInput)
    const effectiveBook = Number.isFinite(parsedBook) ? clampBookSize(parsedBook) : bookSize
    setBookSize(effectiveBook)
    setBookSizeInput(String(effectiveBook))

    const positions = rows
      .filter(r => r.ticker.trim().length > 0)
      .map(r => ({ ticker: r.ticker.trim().toUpperCase(), conviction: r.conviction }))

    if (positions.length === 0) return

    mutation.mutate({ book: effectiveBook, target_leverage: targetLeverage, positions })
  }

  const data = (mutation.data as SizerResponse | undefined) ?? cachedResult
  const weightsRows = toRows(data?.weights_df)
  const weightsHiddenKeys = weightsViewMode === "basic"
    ? (weightsRows.length === 0
        ? []
        : Object.keys(weightsRows[0]).filter(
            k => !BASIC_COLUMNS.has(k) || ALWAYS_HIDDEN_COLUMNS.includes(k as typeof ALWAYS_HIDDEN_COLUMNS[number]),
          ))
    : [...ALWAYS_HIDDEN_COLUMNS]
  const hedgesRows = toRows(data?.hedges_df)
  const exposures = data?.exposures ?? {}
  const constraints = data?.constraints ?? {}
  const maxScaled = data?.max_scaled
  const maxScaledRows = toRows(maxScaled?.weights_df)
  const maxScaledExposures = maxScaled?.exposures ?? {}

  const volDaily = firstNumber(data?.vol_daily)
  const grossLeverage = firstNumber(data?.gross_leverage)
  const hedgeGross = firstNumber(exposures.hedge_gross, 0) ?? 0
  const hedgeSpyWeight = firstNumber(data?.hedge_spy_weight, getHedgeWeightFromRows(hedgesRows, "SPY"))
  const hedgeIwmWeight = firstNumber(data?.hedge_iwm_weight, getHedgeWeightFromRows(hedgesRows, "IWM"))
  const hedgeDirectionIssues = Array.from(new Set([
    ...(Array.isArray(data?.hedge_direction_issues) ? data.hedge_direction_issues.filter(v => typeof v === "string") : []),
    ...(hedgeSpyWeight != null && hedgeSpyWeight > 0
      ? [`SPY hedge is long (${hedgeSpyWeight >= 0 ? "+" : ""}${hedgeSpyWeight.toFixed(4)}). Long exposure should generally be hedged by shorting SPY.`]
      : []),
    ...(hedgeIwmWeight != null && hedgeIwmWeight < 0
      ? [`IWM hedge is short (${hedgeIwmWeight >= 0 ? "+" : ""}${hedgeIwmWeight.toFixed(4)}). Short exposure should generally be hedged by going long IWM.`]
      : []),
  ]))
  const hedgeDirectionWarning = typeof data?.hedge_direction_warning === "string" && data.hedge_direction_warning.trim().length > 0
    ? data.hedge_direction_warning
    : hedgeDirectionIssues.length > 0
      ? "Potential hedge direction mismatch detected."
      : null
  const equityNet = firstNumber(exposures.equity_net, data?.equity_net)
  const netBetaSpy = firstNumber(data?.net_beta_spy)
  const netBetaIwm = firstNumber(data?.net_beta_iwm)
  const postHedgeBetaSpy = firstNumber(data?.post_hedge_beta_spy)
  const postHedgeBetaIwm = firstNumber(data?.post_hedge_beta_iwm)
  const showHeaderMetrics = [
    volDaily,
    grossLeverage,
    equityNet,
    netBetaSpy,
    netBetaIwm,
    postHedgeBetaSpy,
    postHedgeBetaIwm,
  ].some(v => v != null)

  const trades = useMemo(() => {
    if (!data) return []
    const allRows = [...weightsRows, ...hedgesRows]
    if (allRows.length === 0) return []

    const tradeList: {
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
      const targetShares = toNumber(row.shares) ?? 0
      const price = toNumber(row.price) ?? 0
      const currentShares = currentHoldings[ticker] ?? 0
      const delta = targetShares - currentShares
      const type = row.type === "hedge" ? "Hedge" : "Position"
      const direction = String(row.direction ?? "")

      tradeList.push({
        ticker,
        type,
        direction,
        currentShares: Math.round(currentShares),
        targetShares: Math.round(targetShares),
        delta: Math.round(delta),
        price,
        notional: Math.round(delta) * price,
      })
    }

    return tradeList.sort((a, b) => Math.abs(b.notional) - Math.abs(a.notional))
  }, [data, weightsRows, hedgesRows, currentHoldings])

  const totalNotional = trades.reduce((sum, t) => sum + Math.abs(t.notional), 0)
  const totalBuys = trades.filter(t => t.delta > 0).reduce((sum, t) => sum + t.notional, 0)
  const totalSells = trades.filter(t => t.delta < 0).reduce((sum, t) => sum + t.notional, 0)

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Sizer</h1>
        <p className="text-sm text-gray-400 mt-0.5">Conviction-based portfolio sizing with constraint optimization and beta hedging</p>
      </div>

      <div className="rounded-xl border border-gray-200/80 bg-white p-5 mb-6 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
          <SliderInput
            label="Book Size"
            value={bookSize}
            onChange={v => setBookSize(clampBookSize(v))}
            min={MIN_BOOK_SIZE}
            max={MAX_BOOK_SIZE}
            step={10_000}
            formatValue={v => currencyFormatter.format(v)}
            minLabel="$10k"
            maxLabel="$10M"
          />

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

          <div className="space-y-1">
            <TextInput
              label="Book Size (Manual)"
              type="number"
              value={bookSizeInput}
              onChange={setBookSizeInput}
              placeholder="100000"
            />
            <p className="text-xs text-gray-400">$10k - $10M · applied on run</p>
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
              <div key={row.id} className="grid grid-cols-12 gap-3 items-center">
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
                            backgroundColor: "hsl(var(--success-soft))",
                            color: "hsl(var(--success-soft-foreground))",
                            borderColor: "hsl(var(--success-border))",
                          }
                        : row.direction === "short"
                          ? {
                              backgroundColor: "hsl(var(--danger-soft))",
                              color: "hsl(var(--danger-soft-foreground))",
                              borderColor: "hsl(var(--danger-border))",
                            }
                          : {
                              backgroundColor: "hsl(var(--muted))",
                              color: "hsl(var(--muted-foreground))",
                              borderColor: "hsl(var(--border))",
                            }
                    }
                  >
                    {row.direction || "—"}
                  </span>
                </div>
                <div className="col-span-6">
                  {idx === 0 && <p className="mb-1 text-xs font-medium text-muted">Conviction</p>}
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={row.conviction}
                    onChange={e => updateConviction(row.id, Number(e.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[hsl(var(--muted-3))]"
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

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Sizing portfolio...">
          Size Portfolio
        </ActionButton>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running portfolio sizer..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
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
            <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-4">
              {volDaily != null && <MetricCard title="Daily Volatility" value={`${(volDaily * 100).toFixed(2)}%`} />}
              {grossLeverage != null && <MetricCard title="Gross Leverage (incl. hedges)" value={`${grossLeverage.toFixed(2)}x`} />}
              {equityNet != null && <MetricCard title="Equity Net" value={formatRatioPercent(equityNet, true, 1)} />}
              {netBetaSpy != null && <MetricCard title="Net Beta SPY (pre-hedge)" value={netBetaSpy.toFixed(3)} />}
              {netBetaIwm != null && <MetricCard title="Net Beta IWM (pre-hedge)" value={netBetaIwm.toFixed(3)} />}
              {postHedgeBetaSpy != null && <MetricCard title="Net Beta SPY (post-hedge)" value={postHedgeBetaSpy.toFixed(3)} />}
              {postHedgeBetaIwm != null && <MetricCard title="Net Beta IWM (post-hedge)" value={postHedgeBetaIwm.toFixed(3)} />}
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
                  columns={buildCols(weightsRows, weightsHiddenKeys)}
                  rows={weightsRows}
                />
              )}

              {hedgesRows.length > 0 && (
                <DataTable label="Hedge Positions" columns={buildCols(hedgesRows)} rows={hedgesRows} />
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

          {trades.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-base font-semibold text-gray-900">Buy / Sell Summary</h2>

              <div className="grid grid-cols-3 gap-4">
                <MetricCard title="Total Buys" value={currencyFormatter.format(totalBuys)} />
                <MetricCard title="Total Sells" value={currencyFormatter.format(totalSells)} />
                <MetricCard title="Total Turnover" value={currencyFormatter.format(totalNotional)} />
              </div>

              <div className="rounded-xl border border-gray-200 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-100 bg-gray-50/60">
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Ticker</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Type</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Direction</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Current</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Target</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Delta</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Action</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Price</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Notional</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map(t => {
                      const action =
                        t.delta > 0 ? "BUY" : t.delta < 0 ? "SELL" : "HOLD"
                      const actionColor =
                        action === "BUY"
                          ? "text-emerald-700 bg-emerald-50"
                          : action === "SELL"
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
                            <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-semibold ${actionColor}`}>
                              {action}
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

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Set conviction levels above and click Size Portfolio.</p>
      )}
    </div>
  )
}
