import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchBreakout } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import {
  ComposedChart,
  Line,
  Area,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

const snapshotCols: ColumnDef[] = [
  { key: "date", header: "Date" },
  { key: "market", header: "Market" },
  { key: "name", header: "Name" },
  { key: "ticker", header: "Ticker" },
  { key: "close", header: "Close", format: v => v != null ? Number(v).toFixed(2) : "N/A" },
  { key: "atr", header: "ATR", format: v => v != null ? Number(v).toFixed(2) : "N/A" },
  { key: "congestion", header: "Congestion" },
  { key: "long_breakout", header: "Long BO" },
  { key: "short_breakout", header: "Short BO" },
]

const eventCols: ColumnDef[] = [
  { key: "date", header: "Date" },
  { key: "market", header: "Market" },
  { key: "name", header: "Name" },
  { key: "ticker", header: "Ticker" },
  { key: "direction", header: "Direction" },
  { key: "close", header: "Close", format: v => v != null ? Number(v).toFixed(2) : "N/A" },
]

function boolFlag(v: unknown): string {
  if (v === true || v === "YES") return "YES"
  if (v === false) return ""
  return String(v ?? "")
}

function formatXAxisDate(isoDate: string): string {
  const d = new Date(isoDate)
  if (Number.isNaN(d.getTime())) return isoDate
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

function formatTooltipDate(v: unknown): string {
  const raw = String(v ?? "")
  const d = new Date(raw)
  if (Number.isNaN(d.getTime())) return raw
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
}

function buildXAxisTicks(data: { date: string }[], targetTicks = 10): string[] {
  if (data.length <= targetTicks) return data.map(d => d.date)
  const step = Math.ceil(data.length / targetTicks)
  const ticks = data.filter((_, i) => i % step === 0).map(d => d.date)
  const last = data[data.length - 1]?.date
  if (last && ticks[ticks.length - 1] !== last) ticks.push(last)
  return ticks
}

export function Breakout() {
  const { data, isLoading, error } = useApiQuery(["breakout"], fetchBreakout)
  const [selectedTicker, setSelectedTicker] = useState<string>("")

  if (isLoading) return <LoadingSpinner message="Fetching breakout data..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load"} />

  const latest: Record<string, unknown>[] = (data.latest ?? []).map((r: Record<string, unknown>) => ({
    ...r,
    date: r["date"] ? String(r["date"]).split("T")[0] : r["date"],
    congestion: boolFlag(r["congestion"]),
    long_breakout: boolFlag(r["long_breakout"]),
    short_breakout: boolFlag(r["short_breakout"]),
  }))

  const events: Record<string, unknown>[] = (data.events ?? []).map((r: Record<string, unknown>) => ({
    ...r,
    date: r["date"] ? String(r["date"]).split("T")[0] : r["date"],
  })).sort((a: Record<string, unknown>, b: Record<string, unknown>) => String(b["date"]).localeCompare(String(a["date"])))

  const history: Record<string, { rows: Record<string, unknown>[] }> = data.history ?? {}

  const congestionCount = latest.filter(r => r["congestion"] === "YES").length
  const longCount = (data.latest ?? []).filter((r: Record<string, unknown>) => r["long_breakout"] === true).length
  const shortCount = (data.latest ?? []).filter((r: Record<string, unknown>) => r["short_breakout"] === true).length

  // Build label list for asset selector
  const assetLabels = (data.latest ?? []).map((r: Record<string, unknown>) =>
    `${r["name"]} (${r["market"]}) [${r["ticker"]}]`
  ).sort()
  const labelToTicker: Record<string, string> = {}
  ;(data.latest ?? []).forEach((r: Record<string, unknown>) => {
    labelToTicker[`${r["name"]} (${r["market"]}) [${r["ticker"]}]`] = String(r["ticker"])
  })

  const currentLabel = selectedTicker
    ? assetLabels.find((l: string) => labelToTicker[l] === selectedTicker) ?? assetLabels[0]
    : assetLabels[0]

  const selTicker = selectedTicker || (currentLabel ? labelToTicker[currentLabel] : "")
  const histRows: Record<string, unknown>[] = selTicker
    ? (history[selTicker]?.rows ?? []).map((r: Record<string, unknown>) => ({
        ...r,
        date: r["date"] ? String(r["date"]).split("T")[0] : r["date"],
        congestion: r["congestion"] === true,
        long_breakout: r["long_breakout"] === true,
        short_breakout: r["short_breakout"] === true,
        close: r["close"] != null ? Number(r["close"]) : null,
        range_high30: r["range_high30"] != null ? Number(r["range_high30"]) : null,
        range_low30: r["range_low30"] != null ? Number(r["range_low30"]) : null,
      }))
    : []

  // Recharts chart data
  const chartData = histRows.slice(-252).map(r => ({
    date: String(r["date"]),
    close: r["close"] as number | null,
    range_high30: r["range_high30"] as number | null,
    range_low30: r["range_low30"] as number | null,
    congestion_fill_high: r["congestion"] ? r["range_high30"] as number | null : null,
    congestion_fill_low: r["congestion"] ? r["range_low30"] as number | null : null,
    long_scatter: r["long_breakout"] ? r["close"] as number : undefined,
    short_scatter: r["short_breakout"] ? r["close"] as number : undefined,
  }))
  const xAxisTicks = buildXAxisTicks(chartData, 10)

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Breakout Detector</h1>
        <RefreshButton queryKeys={[["breakout"]]} />
      </div>
      <p className="text-xs text-gray-400 mb-4">
        Daily congestion regime + prior-bar range breakouts (formula-only)
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <MetricCard title="Assets" value={latest.length} />
        <MetricCard title="In Congestion" value={congestionCount} />
        <MetricCard title="Latest Long" value={longCount} />
        <MetricCard title="Latest Short" value={shortCount} />
        <MetricCard title="Historical Events" value={events.length} />
      </div>

      <hr className="mb-6" />

      <h2 className="text-lg font-semibold mb-3">Latest Snapshot</h2>
      <DataTable columns={snapshotCols} rows={latest} />

      <h2 className="text-lg font-semibold mt-6 mb-3">Historical Breakout Events</h2>
      {events.length === 0 ? (
        <p className="text-gray-400 text-sm">No historical events.</p>
      ) : (
        <DataTable columns={eventCols} rows={events.slice(0, 100)} />
      )}

      <h2 className="text-lg font-semibold mt-6 mb-3">History Charts</h2>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm text-gray-600">Asset:</label>
        <select
          className="border rounded px-2 py-1 text-sm"
          value={currentLabel}
          onChange={e => setSelectedTicker(labelToTicker[e.target.value])}
        >
          {assetLabels.map((l: string) => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              ticks={xAxisTicks}
              interval={0}
              minTickGap={20}
              tickFormatter={formatXAxisDate}
              tick={{ fontSize: 10 }}
              tickLine={false}
            />
            <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={60} domain={['auto', 'auto']} />
            <Tooltip labelFormatter={(label: unknown) => formatTooltipDate(label)} />
            <Legend />
            <Area dataKey="congestion_fill_high" fill="#ffc107" stroke="none" fillOpacity={0.2} name="Congestion" legendType="rect" />
            <Area dataKey="congestion_fill_low" fill="#ffc107" stroke="none" fillOpacity={0} />
            <Line type="monotone" dataKey="close" stroke="#1f77b4" dot={false} strokeWidth={1.8} name="Close" />
            <Line type="monotone" dataKey="range_high30" stroke="#00c853" dot={false} strokeDasharray="4 2" strokeWidth={1.2} name="RangeHigh30" />
            <Line type="monotone" dataKey="range_low30" stroke="#ff1744" dot={false} strokeDasharray="4 2" strokeWidth={1.2} name="RangeLow30" />
            <Scatter dataKey="long_scatter" fill="#00c853" name="Long BO" shape="triangle" />
            <Scatter dataKey="short_scatter" fill="#ff1744" name="Short BO" shape="triangle" />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
