import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SliderInput, TextInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MetricCard } from "@/components/shared/MetricCard"
import { colorPositiveNegative } from "@/lib/colors"
import { fetchHedgingToolPrefill, runHedgingToolAsync } from "@/lib/api"

interface HedgingResult {
  input_count?: number
  unique_ticker_count?: number
  net_beta_spy?: number
  net_beta_iwm?: number
  post_hedge_beta_spy?: number
  post_hedge_beta_iwm?: number
  hedge_spy_weight?: number
  hedge_iwm_weight?: number
  hedge_spy_dollar?: number
  hedge_iwm_dollar?: number
  gross_input?: number
  net_input?: number
  positions_df?: Record<string, unknown>[]
  hedges_df?: Record<string, unknown>[]
  [key: string]: unknown
}

interface PositionRow {
  id: string
  ticker: string
  weight: string
}

const HEDGING_STATE_KEY = ["hedging-tool", "state"] as const
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

const COLUMN_LABELS: Record<string, string> = {
  ticker: "Ticker",
  type: "Type",
  direction: "Direction",
  weight: "Weight",
  price: "Price",
  shares: "Shares",
  dollar_weight: "Dollar",
  beta_spy: "Beta SPY",
  beta_iwm: "Beta IWM",
  beta_contribution_spy: "Beta Contribution SPY",
  beta_contribution_iwm: "Beta Contribution IWM",
}

function makeRow(ticker = "", weight = ""): PositionRow {
  return {
    id: `row-${Math.random().toString(36).slice(2, 10)}`,
    ticker,
    weight,
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

function clampBookSize(value: number) {
  if (!Number.isFinite(value)) return MIN_BOOK_SIZE
  return Math.min(MAX_BOOK_SIZE, Math.max(MIN_BOOK_SIZE, Math.round(value)))
}

function formatRatioPercent(value: number, signed = true, precision = 2) {
  const pct = value * 100
  const sign = signed && pct >= 0 ? "+" : ""
  return `${sign}${pct.toFixed(precision)}%`
}

function isCurrencyColumn(key: string) {
  const normalized = key.toLowerCase()
  return normalized.includes("dollar") || normalized.includes("price") || normalized.includes("amount")
}

function isIntegerColumn(key: string) {
  return key.toLowerCase() === "shares"
}

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).filter(k => k !== "index").map(k => ({
    key: k,
    header: COLUMN_LABELS[k] ?? k,
    colorFn: typeof rows[0][k] === "number" ? colorPositiveNegative : undefined,
    format: (v: unknown) => {
      if (typeof v !== "number") return String(v ?? "N/A")
      if (k === "weight") return formatRatioPercent(v, true, 2)
      if (isCurrencyColumn(k) && k.toLowerCase().includes("price")) return priceFormatter.format(v)
      if (isCurrencyColumn(k)) return currencyFormatter.format(v)
      if (isIntegerColumn(k)) return Math.round(v).toLocaleString("en-US")
      return `${v >= 0 ? "+" : ""}${numberFormatter.format(v)}`
    },
  }))
}

function parsePositions(rows: PositionRow[]) {
  const parsed: { ticker: string; weight: number }[] = []
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i]
    const ticker = row.ticker.trim().toUpperCase()
    const weightText = row.weight.trim()

    if (!ticker && !weightText) continue
    if (!ticker) throw new Error(`Row ${i + 1}: ticker is required.`)

    const numericText = weightText.replace(/%/g, "").trim()
    const percent = Number(numericText)
    if (!Number.isFinite(percent)) throw new Error(`Row ${i + 1}: weight must be a valid percent (e.g., 12 or 12%).`)

    const weight = percent / 100.0

    parsed.push({ ticker, weight })
  }
  if (parsed.length === 0) throw new Error("Add at least one ticker and weight before computing hedge.")
  return parsed
}

export function HedgingTool() {
  const queryClient = useQueryClient()
  const cachedState = queryClient.getQueryData<{
    bookSize: number
    bookSizeInput: string
    rows: PositionRow[]
    result: HedgingResult | null
  }>(HEDGING_STATE_KEY)

  const [bookSize, setBookSize] = useState(cachedState?.bookSize ?? 100_000)
  const [bookSizeInput, setBookSizeInput] = useState(cachedState?.bookSizeInput ?? String(cachedState?.bookSize ?? 100_000))
  const [rows, setRows] = useState<PositionRow[]>(cachedState?.rows && cachedState.rows.length > 0 ? cachedState.rows : [makeRow()])
  const [validationError, setValidationError] = useState<string | null>(null)
  const [cachedResult, setCachedResult] = useState<HedgingResult | null>(cachedState?.result ?? null)

  useEffect(() => {
    if (cachedState?.rows && cachedState.rows.length > 0) return

    let canceled = false
    fetchHedgingToolPrefill()
      .then((data: unknown) => {
        if (canceled) return
        const positions = (data as { positions?: Array<{ ticker?: unknown; weight?: unknown }> })?.positions ?? []
        const prefilled = positions
          .map(p => ({
            ticker: String(p?.ticker ?? "").trim().toUpperCase(),
            weight: String(typeof p?.weight === "number" ? p.weight : 0),
          }))
          .filter(p => p.ticker.length > 0)
          .map(p => makeRow(p.ticker, p.weight))

        if (prefilled.length > 0) setRows(prefilled)
      })
      .catch(() => {
        // Keep default single blank row on prefill failure.
      })

    return () => {
      canceled = true
    }
  }, [cachedState?.rows])

  const mutation = useMutation({
    mutationFn: runHedgingToolAsync,
    onSuccess: result => setCachedResult((result as HedgingResult) ?? null),
  })

  useEffect(() => {
    queryClient.setQueryData(HEDGING_STATE_KEY, {
      bookSize,
      bookSizeInput,
      rows,
      result: cachedResult,
    })
  }, [bookSize, bookSizeInput, rows, cachedResult, queryClient])

  const typedRows = rows

  function updateRow(id: string, field: "ticker" | "weight", value: string) {
    setRows(prev => prev.map(row => (row.id === id ? { ...row, [field]: value } : row)))
  }

  function addRow() {
    setRows(prev => [...prev, makeRow()])
  }

  function removeRow(id: string) {
    setRows(prev => {
      if (prev.length <= 1) return [makeRow()]
      const next = prev.filter(row => row.id !== id)
      return next.length > 0 ? next : [makeRow()]
    })
  }

  function handleRun() {
    setValidationError(null)

    const parsedBook = Number(bookSizeInput)
    const effectiveBook = Number.isFinite(parsedBook) ? clampBookSize(parsedBook) : bookSize
    setBookSize(effectiveBook)
    setBookSizeInput(String(effectiveBook))

    let positions: { ticker: string; weight: number }[]
    try {
      positions = parsePositions(typedRows)
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : "Invalid position inputs.")
      return
    }

    mutation.mutate({ book: effectiveBook, positions })
  }

  const data = (mutation.data as HedgingResult | undefined) ?? cachedResult
  const positionsRows = toRows(data?.positions_df)
  const hedgesRows = toRows(data?.hedges_df)

  const summary = useMemo(() => ({
    netBetaSpy: firstNumber(data?.net_beta_spy),
    netBetaIwm: firstNumber(data?.net_beta_iwm),
    postHedgeBetaSpy: firstNumber(data?.post_hedge_beta_spy),
    postHedgeBetaIwm: firstNumber(data?.post_hedge_beta_iwm),
    hedgeSpyWeight: firstNumber(data?.hedge_spy_weight),
    hedgeIwmWeight: firstNumber(data?.hedge_iwm_weight),
    hedgeSpyDollar: firstNumber(data?.hedge_spy_dollar),
    hedgeIwmDollar: firstNumber(data?.hedge_iwm_dollar),
    grossInput: firstNumber(data?.gross_input),
    netInput: firstNumber(data?.net_input),
    inputCount: firstNumber(data?.input_count),
    uniqueTickerCount: firstNumber(data?.unique_ticker_count),
  }), [data])

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Hedging Tool</h1>
        <p className="text-sm text-gray-400 mt-0.5">Compute SPY/IWM hedge legs from custom signed portfolio weights</p>
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
            <p className="text-sm font-medium text-gray-700">Portfolio Weights</p>
            <button
              type="button"
              onClick={addRow}
              className="px-3 py-1.5 rounded-lg border border-gray-200 text-sm text-gray-700 hover:bg-gray-50"
            >
              Add Row
            </button>
          </div>
          <p className="text-xs text-gray-400 mb-2">Enter weights as percentages (example: 12 means 12%).</p>

          <div className="space-y-2">
            {typedRows.map((row, idx) => (
              <div key={row.id} className="grid grid-cols-12 gap-2 items-end">
                <div className="col-span-5">
                  <TextInput
                    label={idx === 0 ? "Ticker" : undefined}
                    value={row.ticker}
                    onChange={v => updateRow(row.id, "ticker", v)}
                    placeholder="AAPL"
                  />
                </div>
                <div className="col-span-5">
                  <TextInput
                    label={idx === 0 ? "Weight (%)" : undefined}
                    type="number"
                    value={row.weight}
                    onChange={v => updateRow(row.id, "weight", v)}
                    placeholder="12"
                  />
                </div>
                <div className="col-span-2">
                  <button
                    type="button"
                    onClick={() => removeRow(row.id)}
                    className="w-full px-2 py-2 rounded-lg border border-gray-200 text-sm text-gray-600 hover:bg-gray-50"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Computing hedge...">
          Compute Hedge
        </ActionButton>
      </div>

      {validationError && <ErrorMessage message={validationError} />}
      {mutation.isPending && <LoadingSpinner message="Running hedging tool..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {summary.inputCount != null && <MetricCard title="Input Rows" value={String(Math.round(summary.inputCount))} />}
            {summary.uniqueTickerCount != null && <MetricCard title="Unique Tickers" value={String(Math.round(summary.uniqueTickerCount))} />}
            {summary.grossInput != null && <MetricCard title="Input Gross" value={formatRatioPercent(summary.grossInput, false, 1)} />}
            {summary.netInput != null && <MetricCard title="Input Net" value={formatRatioPercent(summary.netInput, true, 1)} />}
            {summary.netBetaSpy != null && <MetricCard title="Net Beta SPY (Pre)" value={summary.netBetaSpy.toFixed(3)} />}
            {summary.netBetaIwm != null && <MetricCard title="Net Beta IWM (Pre)" value={summary.netBetaIwm.toFixed(3)} />}
            {summary.postHedgeBetaSpy != null && <MetricCard title="Net Beta SPY (Post)" value={summary.postHedgeBetaSpy.toFixed(3)} />}
            {summary.postHedgeBetaIwm != null && <MetricCard title="Net Beta IWM (Post)" value={summary.postHedgeBetaIwm.toFixed(3)} />}
            {summary.hedgeSpyWeight != null && <MetricCard title="Hedge SPY Weight" value={formatRatioPercent(summary.hedgeSpyWeight, true, 2)} />}
            {summary.hedgeIwmWeight != null && <MetricCard title="Hedge IWM Weight" value={formatRatioPercent(summary.hedgeIwmWeight, true, 2)} />}
            {summary.hedgeSpyDollar != null && <MetricCard title="Hedge SPY Dollar" value={currencyFormatter.format(summary.hedgeSpyDollar)} />}
            {summary.hedgeIwmDollar != null && <MetricCard title="Hedge IWM Dollar" value={currencyFormatter.format(summary.hedgeIwmDollar)} />}
          </div>

          {positionsRows.length > 0 && (
            <DataTable label="Input Positions (Aggregated)" columns={buildCols(positionsRows)} rows={positionsRows} />
          )}

          {hedgesRows.length > 0 && (
            <DataTable label="Hedge Positions" columns={buildCols(hedgesRows)} rows={hedgesRows} />
          )}

          {positionsRows.length === 0 && hedgesRows.length === 0 && (
            <p className="text-gray-400 text-sm">No positions or hedge rows returned.</p>
          )}
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Add ticker weights above and click Compute Hedge.</p>
      )}
    </div>
  )
}
