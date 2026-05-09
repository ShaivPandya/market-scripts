import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Download, Loader2 } from "lucide-react"

import { downloadPriceHistory, runChart, runPriceRatioChart } from "@/lib/api"
import { TimeSeriesChart, type SeriesDef } from "@/components/shared/TimeSeriesChart"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, TextInput, ActionButton } from "@/components/shared/FormControls"
import { MetricCard } from "@/components/shared/MetricCard"

const LOOKBACKS = ["3M", "1Y", "2Y", "5Y"] as const

type ChartMode = "single" | "ratio"
type PriceScale = "linear" | "log"
type RatioWindow = "5Y" | "10Y"

interface RatioPreset {
  label: string
  symbolA: string
  symbolB: string
}

interface RatioPayload {
  symbol_a: string
  symbol_b: string
  start_date?: string
  end_date?: string
}

interface RatioRequestBase {
  symbol_a: string
  symbol_b: string
  end_date: string
}

interface RatioRow {
  date: string
  priceA: number | null
  priceB: number | null
  ratio: number | null
}

const RATIO_PRESETS: RatioPreset[] = [
  { label: "Silver / Gold", symbolA: "SI=F", symbolB: "GC=F" },
  { label: "S&P 500 / S&P Equal Weight", symbolA: "^GSPC", symbolB: "RSP" },
  { label: "VIX / VVIX", symbolA: "^VIX", symbolB: "^VVIX" },
  { label: "HD / LOW", symbolA: "HD", symbolB: "LOW" },
  { label: "V / MA", symbolA: "V", symbolB: "MA" },
  { label: "Russell 2000 / S&P 600", symbolA: "^RUT", symbolB: "^SP600" },
]

const MA_COLUMNS = ["100D SMA", "150D SMA", "200D SMA", "40W SMA", "200W SMA", "10M SMA", "20M SMA"]
const ROC_COLUMNS = ["1M ROC", "3M ROC", "12M ROC"]

const PRICE_SERIES: SeriesDef[] = [
  { key: "Close", color: "#1f77b4", strokeWidth: 2, opacity: 1 },
  { key: "100D SMA", color: "#ff7f0e", strokeWidth: 1, opacity: 0.75 },
  { key: "150D SMA", color: "#2ca02c", strokeWidth: 1, opacity: 0.75 },
  { key: "200D SMA", color: "#d62728", strokeWidth: 1, opacity: 0.75 },
  { key: "40W SMA", color: "#9467bd", strokeWidth: 1, opacity: 0.75 },
  { key: "200W SMA", color: "#8c564b", strokeWidth: 1, opacity: 0.75 },
  { key: "10M SMA", color: "#e377c2", strokeWidth: 1, opacity: 0.75 },
  { key: "20M SMA", color: "#7f7f7f", strokeWidth: 1, opacity: 0.75 },
]

const ROC_SERIES: SeriesDef[] = [
  { key: "1M ROC", color: "#1f77b4" },
  { key: "3M ROC", color: "#ff7f0e" },
  { key: "12M ROC", color: "#2ca02c" },
]

const PRICE_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

function csvFileNameForTicker(ticker: string): string {
  const clean = ticker.trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "")
  return `${clean || "ticker"}_price_history.csv`
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function toDateSortKey(value: string): number | null {
  const ts = Date.parse(value)
  return Number.isFinite(ts) ? ts : null
}

function toDateOnly(value: unknown): string {
  const raw = String(value ?? "").trim()
  if (!raw) return ""
  if (/^\d{4}-\d{2}-\d{2}/.test(raw)) return raw.slice(0, 10)
  return raw
}

function lookbackCutoff(lookback: string): Date {
  const now = new Date()
  switch (lookback) {
    case "3M": return new Date(now.getFullYear(), now.getMonth() - 3, now.getDate())
    case "1Y": return new Date(now.getFullYear() - 1, now.getMonth(), now.getDate())
    case "2Y": return new Date(now.getFullYear() - 2, now.getMonth(), now.getDate())
    default: return new Date(0)
  }
}

function toDateInputValue(date: Date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, "0")
  const d = String(date.getDate()).padStart(2, "0")
  return `${y}-${m}-${d}`
}

function isoDateToday() {
  return toDateInputValue(new Date())
}

function computeWindowStartDate(window: RatioWindow): string {
  const years = window === "10Y" ? 10 : 5
  const end = new Date()
  end.setFullYear(end.getFullYear() - years)
  return toDateInputValue(end)
}

function buildRatioPayload(
  symbolA: string,
  symbolB: string,
  startDate: string,
  endDate: string,
): RatioPayload {
  const payload: RatioPayload = {
    symbol_a: symbolA,
    symbol_b: symbolB,
  }
  const start = startDate.trim()
  const end = endDate.trim()
  if (start) payload.start_date = start
  if (end) payload.end_date = end
  return payload
}

function formatRatio(value: number | null, digits = 4): string {
  if (value == null) return "N/A"
  return value.toFixed(digits)
}

function formatPercentFromDecimal(value: number | null): string {
  if (value == null) return "N/A"
  const pct = value * 100
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const rank = (sorted.length - 1) * p
  const lowerIdx = Math.floor(rank)
  const upperIdx = Math.ceil(rank)
  const lower = sorted[lowerIdx]
  const upper = sorted[upperIdx]
  if (lowerIdx === upperIdx) return lower
  const weight = rank - lowerIdx
  return lower + (upper - lower) * weight
}

export function ChartPage() {
  const [mode, setMode] = useState<ChartMode>("single")

  const [ticker, setTicker] = useState("")
  const [lookback, setLookback] = useState<string>("2Y")
  const [priceScale, setPriceScale] = useState<PriceScale>("linear")
  const [submittedTicker, setSubmittedTicker] = useState<string | null>(null)
  const [isDownloadingHistory, setIsDownloadingHistory] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)

  const [ratioSymbolA, setRatioSymbolA] = useState("")
  const [ratioSymbolB, setRatioSymbolB] = useState("")
  const [ratioWindow, setRatioWindow] = useState<RatioWindow>("5Y")
  const [submittedRatioBase, setSubmittedRatioBase] = useState<RatioRequestBase | null>(null)

  const singleQuery = useQuery({
    queryKey: ["chart", "single", submittedTicker],
    queryFn: () => runChart({ ticker: submittedTicker!, lookback: "5Y" }),
    enabled: Boolean(submittedTicker),
    staleTime: Infinity,
  })

  const ratioQuery5Y = useQuery({
    queryKey: ["chart", "ratio", submittedRatioBase, "5Y"],
    queryFn: () => runPriceRatioChart(buildRatioPayload(
      submittedRatioBase!.symbol_a,
      submittedRatioBase!.symbol_b,
      computeWindowStartDate("5Y"),
      submittedRatioBase!.end_date,
    )),
    enabled: Boolean(submittedRatioBase),
    staleTime: Infinity,
  })

  const ratioQuery10Y = useQuery({
    queryKey: ["chart", "ratio", submittedRatioBase, "10Y"],
    queryFn: () => runPriceRatioChart(buildRatioPayload(
      submittedRatioBase!.symbol_a,
      submittedRatioBase!.symbol_b,
      computeWindowStartDate("10Y"),
      submittedRatioBase!.end_date,
    )),
    enabled: Boolean(submittedRatioBase),
    staleTime: Infinity,
  })

  const activeRatioQuery = ratioWindow === "10Y" ? ratioQuery10Y : ratioQuery5Y
  const isLoading = mode === "ratio"
    ? ratioQuery5Y.isFetching || ratioQuery10Y.isFetching
    : singleQuery.isFetching
  const isError = mode === "ratio" ? activeRatioQuery.isError : singleQuery.isError
  const error = mode === "ratio" ? activeRatioQuery.error : singleQuery.error

  const singleData = (singleQuery.data ?? null) as Record<string, unknown> | null
  const ratioData = (activeRatioQuery.data ?? null) as Record<string, unknown> | null

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()

    if (mode === "single") {
      const normalizedTicker = ticker.trim().toUpperCase()
      if (!normalizedTicker) return
      setSubmittedTicker(normalizedTicker)
      return
    }

    const symbolA = ratioSymbolA.trim().toUpperCase()
    const symbolB = ratioSymbolB.trim().toUpperCase()
    if (!symbolA || !symbolB) return

    const endDate = isoDateToday()
    setRatioSymbolA(symbolA)
    setRatioSymbolB(symbolB)
    setSubmittedRatioBase({ symbol_a: symbolA, symbol_b: symbolB, end_date: endDate })
  }

  function handleRatioPresetClick(preset: RatioPreset) {
    setMode("ratio")
    setRatioSymbolA(preset.symbolA)
    setRatioSymbolB(preset.symbolB)
    const endDate = isoDateToday()
    setSubmittedRatioBase({ symbol_a: preset.symbolA, symbol_b: preset.symbolB, end_date: endDate })
  }

  async function handleDownloadHistory() {
    const activeTicker = (ticker.trim() || submittedTicker || "").toUpperCase()
    if (!activeTicker || isDownloadingHistory) return

    setIsDownloadingHistory(true)
    setDownloadError(null)
    try {
      const blob = await downloadPriceHistory(activeTicker)
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = csvFileNameForTicker(activeTicker)
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.URL.revokeObjectURL(url)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : "Failed to download price history")
    } finally {
      setIsDownloadingHistory(false)
    }
  }

  const allPriceData = useMemo<Record<string, unknown>[]>(() => {
    const rows = Array.isArray(singleData?.price_data) ? singleData.price_data : []
    return rows
      .map((r: Record<string, unknown>) => {
        const pt: Record<string, unknown> = {
          date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
          Close: r["Close"] != null ? Number(r["Close"]) : null,
        }
        for (const col of MA_COLUMNS) {
          const v = r[col]
          pt[col] = v != null && v !== "" ? Number(v) : null
        }
        return pt
      })
      .filter((d: Record<string, unknown>) => d.date)
  }, [singleData])

  const allRocData = useMemo<Record<string, unknown>[]>(() => {
    const rows = Array.isArray(singleData?.roc_data) ? singleData.roc_data : []
    return rows
      .map((r: Record<string, unknown>) => {
        const pt: Record<string, unknown> = {
          date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
        }
        for (const col of ROC_COLUMNS) {
          const v = r[col]
          pt[col] = v != null && v !== "" ? Number(v) : null
        }
        return pt
      })
      .filter((d: Record<string, unknown>) => d.date)
  }, [singleData])

  const cutoff = lookbackCutoff(lookback)
  const priceMultiData = useMemo(
    () => allPriceData.filter(d => new Date(String(d.date)) >= cutoff),
    [allPriceData, cutoff],
  )
  const rocMultiData = useMemo(
    () => allRocData.filter(d => new Date(String(d.date)) >= cutoff),
    [allRocData, cutoff],
  )
  const priceLogAvailable = useMemo(() => {
    let hasFiniteValue = false
    for (const row of priceMultiData) {
      for (const series of PRICE_SERIES) {
        const value = toNumber(row[series.key])
        if (value == null) continue
        hasFiniteValue = true
        if (value <= 0) return false
      }
    }
    return hasFiniteValue
  }, [priceMultiData])
  const effectivePriceScale: PriceScale = priceScale === "log" && priceLogAvailable ? "log" : "linear"

  useEffect(() => {
    if (priceScale === "log" && !priceLogAvailable) {
      setPriceScale("linear")
    }
  }, [priceLogAvailable, priceScale])

  const summaryRows: Record<string, unknown>[] = Array.isArray(singleData?.summary) ? singleData.summary : []
  const summaryCols: ColumnDef[] = summaryRows.length > 0
    ? Object.keys(summaryRows[0]).map(k => ({
      key: k,
      header: k,
      colorFn: k.toLowerCase().includes("bias")
        ? (v: unknown) => (String(v).toLowerCase() === "bullish" ? "green" : "red")
        : undefined,
    }))
    : []

  const displayTicker = String(singleData?.ticker ?? ticker).toUpperCase()
  const displayName = typeof singleData?.name === "string" && singleData.name.trim() ? singleData.name.trim() : displayTicker

  const currentPrice = useMemo<number | null>(() => {
    for (let i = allPriceData.length - 1; i >= 0; i -= 1) {
      const close = allPriceData[i]?.Close
      if (typeof close === "number" && Number.isFinite(close)) return close
    }
    return null
  }, [allPriceData])

  const ratioRows = useMemo<RatioRow[]>(() => {
    const rows = Array.isArray(ratioData?.ratio_data) ? ratioData.ratio_data : []
    return rows
      .map((r: Record<string, unknown>) => ({
        date: toDateOnly(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
        priceA: toNumber(r["Price A"]),
        priceB: toNumber(r["Price B"]),
        ratio: toNumber(r["Ratio"]),
      }))
      .filter(r => r.date.length > 0)
      .sort((a, b) => {
        const aTs = toDateSortKey(a.date)
        const bTs = toDateSortKey(b.date)
        if (aTs != null && bTs != null) return aTs - bTs
        if (aTs != null) return -1
        if (bTs != null) return 1
        return a.date.localeCompare(b.date)
      })
  }, [ratioData])

  const ratioStats = (ratioData?.stats && typeof ratioData.stats === "object")
    ? ratioData.stats as Record<string, unknown>
    : {}

  const activeRatioSymbolA = String(ratioData?.symbol_a ?? ratioSymbolA).trim().toUpperCase()
  const activeRatioSymbolB = String(ratioData?.symbol_b ?? ratioSymbolB).trim().toUpperCase()
  const activeRatioNameA = typeof ratioData?.name_a === "string" ? ratioData.name_a : activeRatioSymbolA
  const activeRatioNameB = typeof ratioData?.name_b === "string" ? ratioData.name_b : activeRatioSymbolB
  const historicalAvg = toNumber(ratioStats.historical_avg) ?? (
    ratioRows.length > 0
      ? (() => {
        const values = ratioRows
          .map(r => r.ratio)
          .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
        if (values.length === 0) return null
        return values.reduce((sum, v) => sum + v, 0) / values.length
      })()
      : null
  )
  const currentRatio = toNumber(ratioStats.end_ratio) ?? (
    ratioRows.length > 0 ? ratioRows[ratioRows.length - 1].ratio : null
  )
  const currentVsHistoricalPct = toNumber(ratioStats.current_vs_historical_avg_pct) ?? (
    historicalAvg != null && historicalAvg !== 0 && currentRatio != null
      ? (currentRatio / historicalAvg) - 1
      : null
  )
  const ratioValues = useMemo(() => (
    ratioRows
      .map(r => r.ratio)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  ), [ratioRows])
  const bottom5Pct = useMemo(() => percentile(ratioValues, 0.05), [ratioValues])
  const bottomDecile = useMemo(() => percentile(ratioValues, 0.1), [ratioValues])
  const topDecile = useMemo(() => percentile(ratioValues, 0.9), [ratioValues])
  const top5Pct = useMemo(() => percentile(ratioValues, 0.95), [ratioValues])
  const ratioChartSeries: SeriesDef[] = useMemo(() => [
    { key: "Ratio", color: "#0f766e", strokeWidth: 2 },
    { key: "Historical Avg", color: "#f59e0b", strokeWidth: 1.5, opacity: 0.9, strokeDasharray: "6 4" },
    { key: "Top 5%", color: "#991b1b", strokeWidth: 1.2, opacity: 0.8, strokeDasharray: "2 3" },
    { key: "Top Decile", color: "#dc2626", strokeWidth: 1.4, opacity: 0.9, strokeDasharray: "4 4" },
    { key: "Bottom Decile", color: "#2563eb", strokeWidth: 1.4, opacity: 0.9, strokeDasharray: "4 4" },
    { key: "Bottom 5%", color: "#1e3a8a", strokeWidth: 1.2, opacity: 0.8, strokeDasharray: "2 3" },
  ], [])

  const ratioMultiChartData = useMemo<Record<string, unknown>[]>(() => (
    ratioRows.map(r => ({
      date: r.date,
      Ratio: r.ratio,
      "Historical Avg": historicalAvg,
      "Top 5%": top5Pct,
      "Top Decile": topDecile,
      "Bottom Decile": bottomDecile,
      "Bottom 5%": bottom5Pct,
    }))
  ), [ratioRows, historicalAvg, top5Pct, topDecile, bottomDecile, bottom5Pct])

  const ratioColumns: ColumnDef[] = useMemo(() => [
    { key: "date", header: "Date" },
    {
      key: "priceA",
      header: `${activeRatioSymbolA} Price`,
      format: (v: unknown) => {
        const n = toNumber(v)
        return n == null ? "N/A" : PRICE_FORMATTER.format(n)
      },
    },
    {
      key: "priceB",
      header: `${activeRatioSymbolB} Price`,
      format: (v: unknown) => {
        const n = toNumber(v)
        return n == null ? "N/A" : PRICE_FORMATTER.format(n)
      },
    },
    {
      key: "ratio",
      header: `${activeRatioSymbolA}/${activeRatioSymbolB}`,
      format: (v: unknown) => formatRatio(toNumber(v), 4),
    },
  ], [activeRatioSymbolA, activeRatioSymbolB])

  const ratioRecentRows = useMemo<Record<string, unknown>[]>(() => (
    [...ratioRows.slice(-250)].reverse().map(r => ({
      date: r.date,
      priceA: r.priceA,
      priceB: r.priceB,
      ratio: r.ratio,
    }))
  ), [ratioRows])

  function handleRatioWindowChange(nextWindow: RatioWindow) {
    setRatioWindow(nextWindow)
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Chart</h1>
      </div>

      <form onSubmit={handleSubmit} className="mb-4 space-y-3">
        <div>
          <label className="block text-sm text-gray-600 mb-1.5">Mode</label>
          <SegmentedControl
            options={[
              { value: "single", label: "Single Symbol" },
              { value: "ratio", label: "Ratio" },
            ]}
            value={mode}
            onChange={setMode}
          />
        </div>

        {mode === "single" && (
          <div className="flex flex-wrap items-end gap-4">
            <TextInput
              label="Ticker"
              value={ticker}
              onChange={setTicker}
              placeholder="SPY"
              className="w-28"
              uppercase
            />
            <div>
              <label className="block text-sm text-gray-600 mb-1.5">Lookback</label>
              <SegmentedControl
                options={LOOKBACKS.map(l => ({ value: l, label: l }))}
                value={lookback}
                onChange={setLookback}
              />
            </div>
            <ActionButton type="submit" loading={isLoading} loadingText="Analyzing..." className="w-auto px-6">
              Analyze
            </ActionButton>
            <button
              type="button"
              onClick={handleDownloadHistory}
              disabled={isDownloadingHistory || !(ticker.trim() || submittedTicker)}
              className="theme-button-secondary inline-flex h-10 items-center gap-2 rounded-lg px-4 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50"
              title="Download complete daily close price history as CSV"
            >
              {isDownloadingHistory ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Price History CSV
            </button>
          </div>
        )}

        {mode === "ratio" && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-end gap-4">
              <TextInput
                label="Symbol A (Numerator)"
                value={ratioSymbolA}
                onChange={setRatioSymbolA}
                placeholder="^GSPC"
                className="w-40"
                uppercase
              />
              <TextInput
                label="Symbol B (Denominator)"
                value={ratioSymbolB}
                onChange={setRatioSymbolB}
                placeholder="RSP"
                className="w-40"
                uppercase
              />
              <div>
                <label className="block text-sm text-gray-600 mb-1.5">View Window</label>
                <SegmentedControl
                  options={[
                    { value: "5Y", label: "5Y" },
                    { value: "10Y", label: "10Y" },
                  ]}
                  value={ratioWindow}
                  onChange={handleRatioWindowChange}
                />
              </div>
              <ActionButton type="submit" loading={isLoading} loadingText="Computing..." className="w-auto px-6">
                Run Ratio
              </ActionButton>
            </div>

            <div className="flex flex-wrap gap-2">
              {RATIO_PRESETS.map(preset => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => handleRatioPresetClick(preset)}
                  className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </form>

      {downloadError && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {downloadError}
        </div>
      )}

      {isLoading && <LoadingSpinner message={mode === "ratio" ? "Fetching ratio data..." : "Fetching chart data..."} />}
      {isError && <ErrorMessage message={String(error)} />}

      {mode === "single" && singleData && !isLoading && (
        <div className="space-y-6">
          {priceMultiData.length > 0 && (
            <div>
              <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold">
                    {displayTicker} - {displayName}
                  </h2>
                  <p className="text-sm text-gray-600">
                    Current Price: {currentPrice != null ? PRICE_FORMATTER.format(currentPrice) : "N/A"}
                  </p>
                </div>
                <div>
                  <label className="block text-sm text-gray-600 mb-1.5">Scale</label>
                  <SegmentedControl
                    options={[
                      { value: "linear", label: "Linear" },
                      {
                        value: "log",
                        label: "Log",
                        disabled: !priceLogAvailable,
                        title: priceLogAvailable
                          ? "Use a logarithmic y-axis"
                          : "Log scale requires all visible price values to be greater than zero",
                      },
                    ]}
                    value={effectivePriceScale}
                    onChange={setPriceScale}
                    size="sm"
                  />
                </div>
              </div>
              <TimeSeriesChart multiData={priceMultiData} series={PRICE_SERIES} height={280} yScale={effectivePriceScale} />
            </div>
          )}

          {rocMultiData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Rate of Change</h2>
              <TimeSeriesChart multiData={rocMultiData} series={ROC_SERIES} height={220} zeroLine />
            </div>
          )}

          {summaryRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Signal Summary</h2>
              <DataTable columns={summaryCols} rows={summaryRows} />
            </div>
          )}
        </div>
      )}

      {mode === "ratio" && ratioData && !isLoading && (
        <div className="space-y-6">
          <div>
            <h2 className="text-base font-semibold">
              {activeRatioSymbolA}/{activeRatioSymbolB}
            </h2>
            <p className="text-sm text-gray-600">
              {activeRatioNameA} vs {activeRatioNameB}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Date Range: {String(ratioStats.start_date ?? "N/A")} to {String(ratioStats.end_date ?? "N/A")}
              {"  "}
              ({String(ratioStats.observations ?? 0)} observations)
            </p>
          </div>

          <div>
            <h2 className="text-base font-semibold mb-2">Price Ratio Over Time</h2>
            <TimeSeriesChart multiData={ratioMultiChartData} series={ratioChartSeries} height={300} />
            <p className="text-xs text-gray-500 mt-2">
              Dashed references show historical average, 5th/95th percentile bands, and 10th/90th percentile decile bands.
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <MetricCard title="Start Ratio" value={formatRatio(toNumber(ratioStats.start_ratio), 4)} />
            <MetricCard title="Current Ratio" value={formatRatio(toNumber(ratioStats.end_ratio), 4)} />
            <MetricCard title="Range Change" value={formatPercentFromDecimal(toNumber(ratioStats.change_pct))} />
            <MetricCard title="Historical Avg" value={formatRatio(historicalAvg, 4)} />
            <MetricCard
              title="Vs Historical Avg"
              value={formatPercentFromDecimal(currentVsHistoricalPct)}
            />
            <MetricCard title="Min Ratio" value={formatRatio(toNumber(ratioStats.min_ratio), 4)} />
            <MetricCard title="Max Ratio" value={formatRatio(toNumber(ratioStats.max_ratio), 4)} />
            <MetricCard title="Bottom 5%" value={formatRatio(bottom5Pct, 4)} />
            <MetricCard title="Bottom Decile" value={formatRatio(bottomDecile, 4)} />
            <MetricCard title="Top Decile" value={formatRatio(topDecile, 4)} />
            <MetricCard title="Top 5%" value={formatRatio(top5Pct, 4)} />
          </div>

          <DataTable
            label="Recent Ratio Data (last 250 rows)"
            columns={ratioColumns}
            rows={ratioRecentRows}
            maxHeight="460px"
          />
        </div>
      )}

      {mode === "single" && !singleData && !isLoading && !isError && (
        <p className="text-gray-400 text-sm">Enter a ticker and click Analyze to view the chart.</p>
      )}

      {mode === "ratio" && !ratioData && !isLoading && !isError && (
        <p className="text-gray-400 text-sm">Set two symbols and click Run Ratio, or choose a preset pair.</p>
      )}
    </div>
  )
}
