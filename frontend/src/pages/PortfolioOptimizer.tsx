import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { runPortfolioOptimizerAsync } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SliderInput, ActionButton, TextInput, Toggle, SegmentedControl } from "@/components/shared/FormControls"
import { colorPositiveNegative } from "@/lib/colors"

type OptimizerTab = "Weights" | "Exposures" | "Constraints" | "Max Scaled"
type ExposureAssetClass = "equity" | "fx" | "commodity" | "bond"

interface OptimizerExposures {
  equity_gross?: number
  equity_net?: number
  fx_gross?: number
  fx_net?: number
  commodity_gross?: number
  commodity_net?: number
  bond_gross?: number
  bond_net?: number
  total_gross?: number
  total_net?: number
  [key: string]: unknown
}

interface OptimizerConstraint {
  utilization?: number
  current?: number
  limit?: number
  [key: string]: unknown
}

interface OptimizerMaxScaled {
  scale_factor?: number
  vol_daily?: number
  daily_vol?: number
  binding_constraint?: string
  exposures?: OptimizerExposures
  weights_df?: Record<string, unknown>[]
  [key: string]: unknown
}

interface OptimizerResponse {
  vol_daily?: number
  daily_vol?: number
  gross_leverage?: number
  equity_net?: number
  net_beta_spy?: number
  net_beta_iwm?: number
  exposures?: OptimizerExposures
  constraints?: Record<string, OptimizerConstraint>
  weights_df?: Record<string, unknown>[]
  hedges_df?: Record<string, unknown>[]
  max_scaled?: OptimizerMaxScaled
  [key: string]: unknown
}

const OPTIMIZER_STATE_KEY = ["portfolio-optimizer", "state"] as const
const MIN_BOOK_SIZE = 10_000
const MAX_BOOK_SIZE = 10_000_000
const OPTIMIZER_TABS: OptimizerTab[] = ["Weights", "Exposures", "Constraints", "Max Scaled"]
const EXPOSURE_CLASSES: ExposureAssetClass[] = ["equity", "fx", "commodity", "bond"]
const GROSS_LIMITS: Record<ExposureAssetClass, number> = {
  equity: 4.0,
  fx: 2.0,
  commodity: 1.0,
  bond: 3.0,
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

function isPercentColumn(key: string) {
  const normalized = key.toLowerCase()
  if (isCurrencyColumn(normalized)) return false
  return normalized.includes("weight") || normalized.includes("pct") || normalized.includes("percent")
}

function isIntegerColumn(key: string) {
  const normalized = key.toLowerCase()
  return normalized === "shares" || normalized === "index"
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

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function toRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value.filter((row): row is Record<string, unknown> => row != null && typeof row === "object")
}

const COLUMN_LABELS: Record<string, string> = {
  index: "#",
  ticker: "Ticker",
  asset: "Asset",
  direction: "Direction",
  signal: "Signal",
  beta_spy: "Beta SPY",
  beta_iwm: "Beta IWM",
  realized_vol: "Vol",
  weight: "Weight",
  dollar_weight: "Dollar",
  price: "Price",
  shares: "Shares",
  type: "Type",
}

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).filter(k => k !== "index").map(k => ({
    key: k,
    header: COLUMN_LABELS[k] ?? k,
    colorFn: isPercentColumn(k)
      ? colorPositiveNegative : undefined,
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

export function PortfolioOptimizer() {
  const queryClient = useQueryClient()
  const cachedState = queryClient.getQueryData<{
    bookSize: number
    targetLeverage: number
    betaNeutral: boolean
    result: OptimizerResponse | null
  }>(OPTIMIZER_STATE_KEY)

  const [bookSize, setBookSize] = useState(cachedState?.bookSize ?? 100_000)
  const [bookSizeInput, setBookSizeInput] = useState(String(cachedState?.bookSize ?? 100_000))
  const [targetLeverage, setTargetLeverage] = useState(cachedState?.targetLeverage ?? 2.0)
  const [betaNeutral, setBetaNeutral] = useState(cachedState?.betaNeutral ?? true)
  const [tab, setTab] = useState<OptimizerTab>("Weights")
  const [cachedResult, setCachedResult] = useState<OptimizerResponse | null>(cachedState?.result ?? null)

  const mutation = useMutation({
    mutationFn: runPortfolioOptimizerAsync,
    onSuccess: result => setCachedResult((result as OptimizerResponse) ?? null),
  })

  useEffect(() => {
    queryClient.setQueryData(OPTIMIZER_STATE_KEY, { bookSize, targetLeverage, betaNeutral, result: cachedResult })
  }, [bookSize, targetLeverage, betaNeutral, cachedResult, queryClient])

  useEffect(() => {
    setBookSizeInput(String(bookSize))
  }, [bookSize])

  function clampBookSize(value: number) {
    return Math.min(MAX_BOOK_SIZE, Math.max(MIN_BOOK_SIZE, Math.round(value)))
  }

  function handleRun() {
    const parsed = Number(bookSizeInput)
    const effectiveBook = Number.isFinite(parsed) ? clampBookSize(parsed) : bookSize
    setBookSize(effectiveBook)
    setBookSizeInput(String(effectiveBook))
    mutation.mutate({ book: effectiveBook, target_leverage: targetLeverage, beta_neutral: betaNeutral })
  }

  const data = (mutation.data as OptimizerResponse | undefined) ?? cachedResult
  const weightsRows = toRows(data?.weights_df)
  const hedgesRows = toRows(data?.hedges_df)
  const exposures = data?.exposures ?? {}
  const constraints = data?.constraints ?? {}
  const maxScaled = data?.max_scaled
  const maxScaledRows = toRows(maxScaled?.weights_df)
  const maxScaledExposures = maxScaled?.exposures ?? {}

  const volDaily = firstNumber(data?.vol_daily, data?.daily_vol)
  const grossLeverage = firstNumber(data?.gross_leverage)
  const equityNet = firstNumber(exposures.equity_net, data?.equity_net)
  const netBetaSpy = firstNumber(data?.net_beta_spy)
  const netBetaIwm = firstNumber(data?.net_beta_iwm)

  const showHeaderMetrics = [volDaily, grossLeverage, equityNet, netBetaSpy, netBetaIwm].some(v => v != null)

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Optimizer</h1>
        <p className="text-sm text-gray-400 mt-0.5">Beta-neutral portfolio construction with volatility targeting</p>
      </div>

      <div className="rounded-xl border border-gray-200/80 bg-white p-5 mb-6">
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
            step={0.1}
            formatValue={v => `${v.toFixed(1)}x`}
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
            <p className="text-xs text-gray-400">$10k – $10M · applied on optimize</p>
          </div>

          <div className="flex flex-col justify-center gap-2">
            <Toggle
              label="Net Neutral"
              checked={betaNeutral}
              onChange={setBetaNeutral}
              description="Scale equity longs/shorts to 0% net exposure"
            />
            <p className="text-xs text-gray-400">
              Gross 4.0x · FX 2.0x · Cmdty 1.0x · Long +20% · Short −10%
            </p>
          </div>
        </div>

        <div className="mt-5">
          <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Optimizing (can take 1-3 min)...">
            Optimize Portfolio
          </ActionButton>
        </div>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running optimization..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          {showHeaderMetrics && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {volDaily != null && <MetricCard title="Daily Volatility" value={`${(volDaily * 100).toFixed(2)}%`} />}
              {grossLeverage != null && <MetricCard title="Gross Leverage" value={`${grossLeverage.toFixed(2)}x`} />}
              {equityNet != null && <MetricCard title="Equity Net" value={formatRatioPercent(equityNet, true, 1)} />}
              {netBetaSpy != null && <MetricCard title="Net Beta SPY (pre-hedge)" value={netBetaSpy.toFixed(3)} />}
              {netBetaIwm != null && <MetricCard title="Net Beta IWM (pre-hedge)" value={netBetaIwm.toFixed(3)} />}
            </div>
          )}

          <div className="mb-2">
            <SegmentedControl
              options={OPTIMIZER_TABS.map(t => ({ value: t, label: t }))}
              value={tab}
              onChange={setTab}
            />
          </div>

          {tab === "Weights" && (
            <div className="space-y-6">
              {weightsRows.length > 0 && (
                <div>
                  <h2 className="text-base font-semibold mb-2">Portfolio Weights</h2>
                  <DataTable columns={buildCols(weightsRows)} rows={weightsRows} />
                </div>
              )}

              {hedgesRows.length > 0 && (
                <div>
                  <h2 className="text-base font-semibold mb-2">Hedge Positions</h2>
                  <DataTable columns={buildCols(hedgesRows)} rows={hedgesRows} />
                </div>
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
                    <MetricCard
                      title="Total Gross"
                      value={formatRatioPercent(firstNumber(exposures.total_gross, data.gross_leverage, 0) ?? 0, false, 1)}
                    />
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
                        firstNumber(maxScaled.vol_daily, maxScaled.daily_vol) != null
                          ? `${((firstNumber(maxScaled.vol_daily, maxScaled.daily_vol) ?? 0) * 100).toFixed(2)}%`
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
                    <h2 className="text-base font-semibold mb-2">Max Scaled Weights</h2>
                    {maxScaledRows.length > 0
                      ? <DataTable columns={buildCols(maxScaledRows)} rows={maxScaledRows} />
                      : <p className="text-gray-400 text-sm">No max-scaled weights returned.</p>}
                  </div>
                </>
              ) : (
                <p className="text-gray-400 text-sm">No max scaled data available.</p>
              )}
            </div>
          )}
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure settings above and click Optimize Portfolio.</p>
      )}
    </div>
  )
}
