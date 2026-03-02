import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { runPortfolioOptimizerAsync } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SliderInput, ActionButton, ControlPanel, TextInput, Toggle } from "@/components/shared/FormControls"
import { colorPositiveNegative } from "@/lib/colors"

type OptimizerResponse = Record<string, unknown>

const OPTIMIZER_STATE_KEY = ["portfolio-optimizer", "state"] as const
const MIN_BOOK_SIZE = 10_000
const MAX_BOOK_SIZE = 10_000_000

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

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Portfolio Optimizer</h1>
        <p className="text-sm text-gray-400 mt-0.5">Beta-neutral portfolio construction with volatility targeting</p>
      </div>

      <ControlPanel>
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

        <TextInput
          label="Book Size (Manual)"
          type="number"
          value={bookSizeInput}
          onChange={setBookSizeInput}
          placeholder="100000"
          className="max-w-xs"
        />
        <p className="text-xs text-gray-400 -mt-3">
          Enter any value from {currencyFormatter.format(MIN_BOOK_SIZE)} to {currencyFormatter.format(MAX_BOOK_SIZE)}.
          Value is applied when you click Optimize Portfolio.
        </p>

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

        <Toggle
          label="Net Neutral"
          checked={betaNeutral}
          onChange={setBetaNeutral}
          description="Scale down equity longs/shorts so net equity exposure = 0%"
        />

        <div className="rounded-lg bg-gray-50 px-3.5 py-3 text-xs text-gray-400 space-y-0.5">
          <p className="font-medium text-gray-500">Constraints</p>
          <p>Total gross: 4.0x · FX: 2.0x · Commodities: 1.0x · Bonds: 3.0x</p>
          <p>Long max: +20% · Short max: −10% · Equity net: −50% to +100%</p>
        </div>

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Optimizing (can take 1-3 min)...">
          Optimize Portfolio
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && <LoadingSpinner message="Running optimization..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          {(data.daily_vol != null || data.gross_leverage != null) && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {data.daily_vol != null && <MetricCard title="Daily Volatility" value={`${(Number(data.daily_vol) * 100).toFixed(2)}%`} />}
              {data.gross_leverage != null && <MetricCard title="Gross Leverage" value={`${Number(data.gross_leverage).toFixed(2)}x`} />}
              {data.equity_net != null && <MetricCard title="Equity Net" value={`${(Number(data.equity_net) * 100).toFixed(1)}%`} />}
              {data.net_beta_spy != null && <MetricCard title="Net Beta SPY (pre-hedge)" value={Number(data.net_beta_spy).toFixed(3)} />}
              {data.net_beta_iwm != null && <MetricCard title="Net Beta IWM (pre-hedge)" value={Number(data.net_beta_iwm).toFixed(3)} />}
            </div>
          )}

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
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure settings above and click Optimize Portfolio.</p>
      )}
    </div>
  )
}
