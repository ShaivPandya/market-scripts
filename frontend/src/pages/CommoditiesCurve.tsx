import { useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCommoditiesCurve } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { SelectInput } from "@/components/shared/FormControls"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MetricCard } from "@/components/shared/MetricCard"
import { colorPositiveNegative } from "@/lib/colors"

type CurvePoint = {
  ticker: string
  label: string
  month: number
  year: number
  current: number | null
  historical: number | null
  change: number | null
  change_pct: number | null
  current_date: string | null
  historical_date: string | null
}

interface CurveAnalysis {
  front_month_price: number | null
  back_month_price: number | null
  spread: number | null
  spread_pct: number | null
  shape: string
  contracts_available: number
  contracts_total: number
}

interface CommodityOption {
  code: string
  name: string
}

interface CommoditiesCurveResponse {
  timestamp?: string
  commodity_code: string
  commodity_name: string
  unit: string
  lookback_days: number
  commodities: CommodityOption[]
  analysis: CurveAnalysis
  points: CurvePoint[]
  warnings: string[]
}

const DEFAULT_LOOKBACK_DAYS = 30

function fmtPrice(v: unknown, unit?: string): string {
  if (v == null) return "N/A"
  const n = Number(v)
  const prefix = unit?.startsWith("$") ? "$" : ""
  return `${prefix}${n.toFixed(2)}`
}

function fmtChange(v: unknown): string {
  if (v == null) return "N/A"
  const n = Number(v)
  return `${n >= 0 ? "+" : ""}$${n.toFixed(2)}`
}

function fmtPct(v: unknown): string {
  if (v == null) return "N/A"
  const n = Number(v)
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`
}

function shapeSignal(shape: string): "success" | "warning" | "error" | "info" | null {
  if (shape === "Contango") return "info"
  if (shape === "Backwardation") return "warning"
  if (shape === "Flat") return "success"
  return null
}

export function CommoditiesCurve() {
  const [commodity, setCommodity] = useState("CL")

  const { data, isLoading, error } = useApiQuery<CommoditiesCurveResponse>(
    ["commodities-curve", commodity, DEFAULT_LOOKBACK_DAYS],
    () => fetchCommoditiesCurve(commodity, DEFAULT_LOOKBACK_DAYS),
  )

  if (isLoading) return <LoadingSpinner message="Fetching commodities curve data..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load"} />

  const { points, analysis, warnings } = data
  const lookbackLabel = `${data.lookback_days}d ago`
  const unit = data.unit

  const hasAnyChartData = points.some(p => p.current != null || p.historical != null)

  const chartRows = points.map(p => ({
    label: p.label,
    current: p.current,
    historical: p.historical,
  }))

  const columns: ColumnDef[] = [
    { key: "label", header: "Contract" },
    { key: "ticker", header: "Ticker" },
    { key: "current", header: "Current", format: (v: unknown) => fmtPrice(v, unit) },
    { key: "historical", header: lookbackLabel, format: (v: unknown) => fmtPrice(v, unit) },
    { key: "change", header: "Change", format: fmtChange, colorFn: colorPositiveNegative },
    { key: "change_pct", header: "Change %", format: fmtPct, colorFn: colorPositiveNegative },
    { key: "current_date", header: "Current Date" },
    { key: "historical_date", header: `${lookbackLabel} Date` },
  ]

  return (
    <div>
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">
            Commodities Curve
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Forward curve (term structure) — current vs {lookbackLabel}
          </p>
          {data.timestamp && (
            <p className="text-xs text-gray-400 mt-1">
              Snapshot: {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <RefreshButton queryKeys={[["commodities-curve", commodity, DEFAULT_LOOKBACK_DAYS]]} />
        </div>
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end">
        <SelectInput
          label="Commodity"
          value={commodity}
          onChange={setCommodity}
          options={data.commodities.map(c => ({ value: c.code, label: c.name }))}
          className="w-full sm:w-64"
        />
        <div className="text-xs text-gray-400 sm:pb-2">
          {data.commodity_name} · {unit}
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-xs font-semibold tracking-widest uppercase text-amber-700 mb-1">
            Data Warnings
          </p>
          <div className="space-y-1">
            {warnings.map((w, i) => (
              <p key={`warn-${i}`} className="text-sm text-amber-800">
                {w}
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Front Month"
          value={fmtPrice(analysis.front_month_price, unit)}
        />
        <MetricCard
          title="12M Spread"
          value={fmtChange(analysis.spread)}
          subtitle={analysis.spread_pct != null ? `${fmtPct(analysis.spread_pct)}` : undefined}
        />
        <MetricCard
          title="Curve Shape"
          value={analysis.shape}
          signal={shapeSignal(analysis.shape)}
        />
        <MetricCard
          title="Contracts Available"
          value={`${analysis.contracts_available}/${analysis.contracts_total}`}
        />
      </div>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">
          Forward Curve
        </h2>
        {!hasAnyChartData ? (
          <p className="text-sm text-gray-400">No curve data available.</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={64}
                tickFormatter={v => `$${Number(v).toFixed(2)}`}
                domain={["auto", "auto"]}
              />
              <Tooltip
                formatter={(v: unknown, name: unknown) => [
                  fmtPrice(v, unit),
                  String(name),
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="current"
                name="Current"
                stroke="#1f77b4"
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="historical"
                name={lookbackLabel}
                stroke="#6b7280"
                strokeWidth={1.8}
                strokeDasharray="6 4"
                dot={{ r: 3 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section>
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">
          Contract Details
        </h2>
        <DataTable columns={columns} rows={points} maxHeight="420px" />
      </section>
    </div>
  )
}
