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
import { fetchYieldCurve } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { SelectInput } from "@/components/shared/FormControls"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MetricCard } from "@/components/shared/MetricCard"
import { colorPositiveNegative } from "@/lib/colors"

interface TenorDef {
  tenor: string
  years: number
}

interface CurvePoint {
  tenor: string
  years: number
  current: number | null
  historical: number | null
  change_bps: number | null
  current_date: string | null
  historical_date: string | null
  source_current: string | null
  source_historical: string | null
}

interface CountryCurve {
  code: string
  name: string
  as_of_date: string | null
  historical_target_date: string | null
  points: CurvePoint[]
  warnings: string[]
}

interface YieldCurveResponse {
  timestamp?: string
  lookback_days: number
  tenor_order: TenorDef[]
  countries: CountryCurve[]
}

const DEFAULT_LOOKBACK_DAYS = 90

function fmtYield(v: unknown): string {
  if (v == null) return "N/A"
  return `${Number(v).toFixed(3)}%`
}

function fmtBps(v: unknown): string {
  if (v == null) return "N/A"
  const n = Number(v)
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)} bps`
}

function slope2s10sBps(points: CurvePoint[], side: "current" | "historical"): number | null {
  const y2 = points.find(p => p.tenor === "2Y")?.[side] ?? null
  const y10 = points.find(p => p.tenor === "10Y")?.[side] ?? null
  if (y2 == null || y10 == null) return null
  return (y10 - y2) * 100
}

function fmtSlope(v: number | null): string {
  if (v == null) return "N/A"
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)} bps`
}

function slopeSignal(v: number | null): "success" | "warning" | "error" | "info" | null {
  if (v == null) return null
  if (v > 0) return "success"
  if (v < 0) return "error"
  return "warning"
}

export function YieldCurve() {
  const [countryCode, setCountryCode] = useState("US")

  const { data, isLoading, error } = useApiQuery<YieldCurveResponse>(
    ["yield-curve", DEFAULT_LOOKBACK_DAYS],
    () => fetchYieldCurve(DEFAULT_LOOKBACK_DAYS),
  )

  if (isLoading) return <LoadingSpinner message="Fetching yield curve data..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load"} />

  const countries = Array.isArray(data.countries) ? data.countries : []
  const tenorOrder = Array.isArray(data.tenor_order) ? data.tenor_order : []

  if (countries.length === 0) {
    return <ErrorMessage message="No country data returned." />
  }

  const selectedCountry = countries.find(c => c.code === countryCode) ?? countries[0]
  const selectedCode = selectedCountry.code
  const lookbackLabel = `${data.lookback_days || DEFAULT_LOOKBACK_DAYS}d ago`

  const pointByTenor = new Map(selectedCountry.points.map(p => [p.tenor, p]))

  const chartRows = tenorOrder.map(t => {
    const p = pointByTenor.get(t.tenor)
    return {
      tenor: t.tenor,
      current: p?.current ?? null,
      historical: p?.historical ?? null,
    }
  })

  const tableRows = tenorOrder.map(t => {
    const p = pointByTenor.get(t.tenor)
    const sourceCurrent = p?.source_current ?? ""
    const sourceHistorical = p?.source_historical ?? ""
    const source =
      sourceCurrent && sourceHistorical && sourceCurrent !== sourceHistorical
        ? `${sourceCurrent} / ${sourceHistorical}`
        : sourceCurrent || sourceHistorical || "N/A"

    return {
      tenor: t.tenor,
      current: p?.current ?? null,
      historical: p?.historical ?? null,
      change_bps: p?.change_bps ?? null,
      current_date: p?.current_date ?? "N/A",
      historical_date: p?.historical_date ?? "N/A",
      source,
    }
  })

  const availableCurrent = selectedCountry.points.filter(p => p.current != null).length
  const slopeCurrent = slope2s10sBps(selectedCountry.points, "current")
  const slopeHistorical = slope2s10sBps(selectedCountry.points, "historical")
  const slopeDelta =
    slopeCurrent != null && slopeHistorical != null ? slopeCurrent - slopeHistorical : null
  const hasAnyChartData = chartRows.some(r => r.current != null || r.historical != null)

  const columns: ColumnDef[] = [
    { key: "tenor", header: "Tenor" },
    { key: "current", header: "Current", format: fmtYield },
    { key: "historical", header: lookbackLabel, format: fmtYield },
    { key: "change_bps", header: "Change", format: fmtBps, colorFn: colorPositiveNegative },
    { key: "current_date", header: "Current Date" },
    { key: "historical_date", header: `${lookbackLabel} Date` },
    { key: "source", header: "Source" },
  ]

  return (
    <div>
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Yield Curve</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Current curve vs {lookbackLabel} for US, UK, Germany, and Japan
          </p>
          {data.timestamp && (
            <p className="text-xs text-gray-400 mt-1">
              Snapshot: {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <RefreshButton queryKeys={[["yield-curve", DEFAULT_LOOKBACK_DAYS]]} />
        </div>
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end">
        <SelectInput
          label="Country"
          value={selectedCode}
          onChange={setCountryCode}
          options={countries.map(c => ({ value: c.code, label: c.name }))}
          className="w-full sm:w-64"
        />
        <div className="text-xs text-gray-400 sm:pb-2">
          As of {selectedCountry.as_of_date ?? "N/A"} · Compare target {selectedCountry.historical_target_date ?? "N/A"}
        </div>
      </div>

      {selectedCountry.warnings.length > 0 && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-xs font-semibold tracking-widest uppercase text-amber-700 mb-1">
            Data Warnings
          </p>
          <div className="space-y-1">
            {selectedCountry.warnings.map((w, i) => (
              <p key={`${selectedCountry.code}-warn-${i}`} className="text-sm text-amber-800">
                {w}
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Tenors Available" value={`${availableCurrent}/${tenorOrder.length}`} />
        <MetricCard
          title="2s10s (Current)"
          value={fmtSlope(slopeCurrent)}
          signal={slopeSignal(slopeCurrent)}
          signalLabel={slopeCurrent != null && slopeCurrent < 0 ? "Inverted" : "Normal"}
        />
        <MetricCard
          title={`2s10s (${lookbackLabel})`}
          value={fmtSlope(slopeHistorical)}
          signal={slopeSignal(slopeHistorical)}
          signalLabel={slopeHistorical != null && slopeHistorical < 0 ? "Inverted" : "Normal"}
        />
        <MetricCard title="2s10s Delta" value={fmtSlope(slopeDelta)} subtitle="Current minus historical" />
      </div>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Curve Overlay</h2>
        {!hasAnyChartData ? (
          <p className="text-sm text-gray-400">No curve points available for this country.</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="tenor" tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={56}
                tickFormatter={v => `${Number(v).toFixed(1)}%`}
              />
              <Tooltip
                formatter={(v: unknown, name: unknown) => [fmtYield(v), String(name)]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="current"
                name={`Current (${selectedCountry.as_of_date ?? "N/A"})`}
                stroke="#1f77b4"
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="historical"
                name={`${lookbackLabel} (${selectedCountry.historical_target_date ?? "N/A"})`}
                stroke="#6b7280"
                strokeWidth={1.8}
                strokeDasharray="6 4"
                dot={{ r: 2 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section>
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Curve Data</h2>
        <DataTable columns={columns} rows={tableRows} maxHeight="420px" />
      </section>
    </div>
  )
}
