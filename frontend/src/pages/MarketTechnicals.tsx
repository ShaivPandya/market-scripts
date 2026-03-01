import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchMarketBreadth,
  fetchTop50Breadth,
  fetchPriceVolumeSignals,
  fetchVixTermStructure,
} from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { TimeSeriesChart } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative, colorSignalFlag, colorVixSignal } from "@/lib/colors"

type Tab = "VIX Term Structure" | "Market Breadth" | "Top 50 Breadth" | "Price/Volume Signals"
const TABS: Tab[] = ["VIX Term Structure", "Market Breadth", "Top 50 Breadth", "Price/Volume Signals"]

// ─── Sub-views ────────────────────────────────────────────────────────────────

function MarketBreadthTab() {
  const { data, isLoading, error } = useApiQuery(["market-breadth"], fetchMarketBreadth)
  if (isLoading) return <LoadingSpinner />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const total = data.total_analyzed ?? 0

  const metrics = [
    { title: "Above 200-DMA", pctKey: "pct_above_200dma", countKey: "above_200dma", high: 80, low: 15, sig: "Signal Active" },
    { title: "Above 20-DMA", pctKey: "pct_above_20dma", countKey: "above_20dma", high: 80, low: 20, sig: "Signal Active" },
    { title: "At 20-Day Highs", pctKey: "pct_at_20day_high", countKey: "at_20day_high", high: 50, low: null, sig: "Signal Active" },
    { title: "At 20-Day Lows", pctKey: "pct_at_20day_low", countKey: "at_20day_low", high: 50, low: null, sig: "Capitulation Signal", sigType: "warning" as const },
    { title: "At 52-Week Highs", pctKey: "pct_at_52wk_high", countKey: "at_52wk_high", high: 15, low: null, sig: "Signal Active" },
    { title: "At 52-Week Lows", pctKey: "pct_at_52wk_low", countKey: "at_52wk_low", high: 15, low: null, sig: "Capitulation Signal", sigType: "warning" as const },
    { title: "At 24-Week Highs", pctKey: "pct_at_24wk_high", countKey: "at_24wk_high", high: 20, low: null, sig: "Signal Active" },
    { title: "At 24-Week Lows", pctKey: "pct_at_24wk_low", countKey: "at_24wk_low", high: 20, low: null, sig: "Capitulation Signal", sigType: "warning" as const },
  ]

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">S&P 500 Market Breadth</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map(m => {
          const pct: number = data[m.pctKey] ?? 0
          const cnt: number = data[m.countKey] ?? 0
          const highlight = pct > m.high || (m.low !== null && pct < m.low)
          return (
            <MetricCard
              key={m.pctKey}
              title={m.title}
              value={`${pct.toFixed(1)}%`}
              subtitle={`${cnt} / ${total}`}
              signal={highlight ? (m.sigType ?? "success") : null}
              signalLabel={highlight ? m.sig : undefined}
            />
          )
        })}
      </div>
    </div>
  )
}

function Top50BreadthTab() {
  const { data, isLoading, error } = useApiQuery(["top50-breadth"], fetchTop50Breadth)
  if (isLoading) return <LoadingSpinner />
  if (error || !data) return <ErrorMessage message={String(error)} />
  if (data.universe_size === 0) return <p className="text-gray-400">No tickers with sufficient data.</p>

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Top 50 S&P 500 Performers — Breadth</h2>
      <div className="grid grid-cols-3 gap-4 mb-4">
        {[
          { key: "pct_below_50dma", label: "% Below 50-DMA", tickersKey: "tickers_below_50dma" },
          { key: "pct_3plus_dist", label: "% with 3+ Distribution Days", tickersKey: "tickers_3plus_dist" },
          { key: "pct_broke_20low", label: "% Broke 20-Day Low", tickersKey: "tickers_broke_20low" },
        ].map(m => (
          <div key={m.key} className="rounded-lg border bg-white p-4 shadow-sm">
            <p className="text-sm text-gray-500 font-medium">{m.label}</p>
            <p className="text-2xl font-bold mt-1">
              {data[m.key] != null ? `${(data[m.key] as number).toFixed(1)}%` : "N/A"}
            </p>
            {(data[m.tickersKey] as string[])?.length > 0 && (
              <p className="text-xs text-gray-400 mt-1">
                {(data[m.tickersKey] as string[]).join(", ")}
              </p>
            )}
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-400">Universe: {data.universe_size} stocks with sufficient data</p>
    </div>
  )
}

function PriceVolumeTab() {
  const { data, isLoading, error } = useApiQuery(["price-volume-signals"], fetchPriceVolumeSignals)
  if (isLoading) return <LoadingSpinner />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const latestCols: ColumnDef[] = [
    { key: "Market", header: "Market" },
    { key: "Date", header: "Date" },
    { key: "DownsideRecordVol", header: "Downside Record Vol", colorFn: colorSignalFlag },
    { key: "NewHigh_LowVol", header: "New High / Low Vol", colorFn: colorSignalFlag },
    { key: "HiVol_Churn", header: "Hi Vol Churn", colorFn: colorSignalFlag },
    { key: "Close", header: "Close" },
    { key: "RetPct", header: "Ret%", colorFn: colorPositiveNegative },
  ]

  const formatRow = (r: Record<string, unknown>) => ({
    ...r,
    DownsideRecordVol: r["DownsideRecordVol"] === true ? "YES" : r["DownsideRecordVol"] === false ? "no" : r["DownsideRecordVol"],
    NewHigh_LowVol: r["NewHigh_LowVol"] === true ? "YES" : r["NewHigh_LowVol"] === false ? "no" : r["NewHigh_LowVol"],
    HiVol_Churn: r["HiVol_Churn"] === true ? "YES" : r["HiVol_Churn"] === false ? "no" : r["HiVol_Churn"],
    RetPct: typeof r["RetPct"] === "number" ? `${(r["RetPct"] as number) >= 0 ? "+" : ""}${(r["RetPct"] as number).toFixed(2)}%` : r["RetPct"],
    Close: typeof r["Close"] === "number" ? (r["Close"] as number).toFixed(2) : r["Close"],
  })

  const latestRows = ((data.latest_df ?? []) as Record<string, unknown>[]).map(formatRow)
  const hitsRows = ((data.hits_df ?? []) as Record<string, unknown>[]).map(formatRow)

  const hitsCols: ColumnDef[] = [
    { key: "Date", header: "Date" },
    { key: "MarketName", header: "Market" },
    { key: "Close", header: "Close" },
    { key: "RetPct", header: "Ret%", colorFn: colorPositiveNegative },
    { key: "DownsideRecordVol", header: "Downside RecVol", colorFn: colorSignalFlag },
    { key: "NewHigh_LowVol", header: "New Hi/Lo Vol", colorFn: colorSignalFlag },
    { key: "HiVol_Churn", header: "HiVol Churn", colorFn: colorSignalFlag },
  ]

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3">Latest Signals</h2>
      <DataTable columns={latestCols} rows={latestRows} />
      {hitsRows.length > 0 && (
        <>
          <h2 className="text-lg font-semibold mt-6 mb-3">Recent Signal History</h2>
          <DataTable columns={hitsCols} rows={hitsRows} />
        </>
      )}
    </div>
  )
}

function VIXTab() {
  const { data, isLoading, error } = useApiQuery(["vix-term-structure"], fetchVixTermStructure)
  if (isLoading) return <LoadingSpinner />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const latest = (data.latest_df as Record<string, unknown>[] | undefined)?.[0]
  const recent: Record<string, unknown>[] = data.recent_df ?? []
  const hits: Record<string, unknown>[] = data.hits_df ?? []

  const chartData = recent
    .map((r: Record<string, unknown>) => ({ date: String(r["Date"]), value: r["Ratio"] as number }))
    .filter(d => d.date && d.value != null)

  const tableCols: ColumnDef[] = [
    { key: "Date", header: "Date" },
    { key: "VIX", header: "VIX", format: v => Number(v).toFixed(2) },
    { key: "VIX3M", header: "VIX3M", format: v => Number(v).toFixed(2) },
    { key: "Ratio", header: "Ratio", format: v => Number(v).toFixed(2) },
    { key: "Signal", header: "Signal", colorFn: colorVixSignal },
  ]

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">VIX Term Structure (3M / 1M)</h2>
      <p className="text-xs text-gray-400 mb-4">
        High ratio (≥ 1.25): later volatility concerns. Low ratio (&lt; 1.0): near-term fear.
      </p>

      {latest && (
        <>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <MetricCard title="3M / 1M Ratio" value={Number(latest["Ratio"]).toFixed(2)} />
            <MetricCard title="VIX" value={Number(latest["VIX"]).toFixed(2)} />
            <MetricCard title={`3M VIX (${latest["UsedTicker"]})`} value={Number(latest["VIX3M"]).toFixed(2)} />
            <MetricCard title="Date" value={String(latest["Date"])} />
          </div>
          {latest["Signal"] === "Fear" && (
            <div className="mb-4 rounded bg-yellow-50 border border-yellow-200 px-4 py-2 text-sm text-yellow-800">
              Signal: Fear (near-term volatility elevated)
            </div>
          )}
          {latest["Signal"] === "Complacency" && (
            <div className="mb-4 rounded bg-blue-50 border border-blue-200 px-4 py-2 text-sm text-blue-800">
              Signal: Complacency (longer-term volatility elevated)
            </div>
          )}
        </>
      )}

      {chartData.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-2">Ratio Over Time (last 12 months)</h3>
          <TimeSeriesChart data={chartData} height={200} zeroLine={false} />
        </div>
      )}

      {recent.length > 0 && (
        <>
          <h3 className="text-sm font-medium mb-2">Recent Ratios</h3>
          <DataTable columns={tableCols} rows={recent.slice(-10)} />
        </>
      )}

      {hits.length > 0 && (
        <>
          <h3 className="text-sm font-medium mt-4 mb-2">Recent Signal Hits</h3>
          <DataTable columns={tableCols} rows={hits} />
        </>
      )}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function MarketTechnicals() {
  const [tab, setTab] = useState<Tab>("VIX Term Structure")

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Market Technicals</h1>
        <RefreshButton />
      </div>

      <div className="flex gap-1 border-b border-gray-200 mb-6">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              tab === t
                ? "border border-b-white border-gray-200 bg-white text-blue-600 -mb-px"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "VIX Term Structure" && <VIXTab />}
      {tab === "Market Breadth" && <MarketBreadthTab />}
      {tab === "Top 50 Breadth" && <Top50BreadthTab />}
      {tab === "Price/Volume Signals" && <PriceVolumeTab />}
    </div>
  )
}
