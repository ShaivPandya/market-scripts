import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { fetchEconomicGrowth, analyzeEconomicGrowth } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { SegmentedControl } from "@/components/shared/FormControls"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative, colorReturnVsBenchmark } from "@/lib/colors"

type EquityView = "table" | "graph"
type ReturnTable = Record<string, Record<string, number | null>>

type EquityChartRow = {
  name: string
  benchmark: string
  returnValue: number
  benchmarkReturn: number
  spread: number
}

const EQUITY_VIEW_OPTIONS: { value: EquityView; label: string }[] = [
  { value: "table", label: "Table" },
  { value: "graph", label: "Graph" },
]

const BENCHMARK_ROWS = new Set(["S&P 500", "STOXX 600"])
const POSITIVE_BAR = "#00c853"
const NEGATIVE_BAR = "#ff1744"
const FLAT_BAR = "#ffc107"

function formatPp(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp`
}

function formatPct(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`
}

function EquityRelativeTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: { payload?: EquityChartRow }[]
}) {
  const row = payload?.[0]?.payload
  if (!active || !row) return null

  return (
    <div className="rounded-lg border border-app bg-card px-3 py-2 text-xs shadow-sm">
      <p className="mb-1 font-semibold text-app">{row.name}</p>
      <div className="space-y-0.5 text-muted">
        <p>Benchmark: <span className="font-medium text-app">{row.benchmark}</span></p>
        <p>Return: <span className="font-medium text-app">{formatPct(row.returnValue)}</span></p>
        <p>Benchmark return: <span className="font-medium text-app">{formatPct(row.benchmarkReturn)}</span></p>
        <p>Spread: <span className="font-medium text-app">{formatPp(row.spread)}</span></p>
      </div>
    </div>
  )
}

function EquityRelativePerformanceChart({
  rows,
}: {
  rows: EquityChartRow[]
}) {
  if (rows.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-xl border border-app bg-card text-sm text-subtle">
        No benchmark-relative equity data available.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-app bg-card px-3 py-4">
      <div className="min-w-[760px]">
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 72 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
            <XAxis
              dataKey="name"
              interval={0}
              tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
              tickLine={false}
              axisLine={{ stroke: "hsl(var(--chart-grid))" }}
              angle={-30}
              textAnchor="end"
              height={78}
            />
            <YAxis
              domain={[
                (dataMin: number) => Math.min(0, dataMin),
                (dataMax: number) => Math.max(0, dataMax),
              ]}
              tickFormatter={formatPp}
              tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
              tickLine={false}
              axisLine={false}
              width={66}
            />
            <Tooltip content={<EquityRelativeTooltip />} cursor={{ fill: "hsl(var(--muted))", opacity: 0.18 }} />
            <ReferenceLine y={0} stroke="hsl(var(--chart-axis))" strokeDasharray="4 2" />
            <Bar dataKey="spread" name="Spread" radius={[4, 4, 0, 0]}>
              {rows.map(row => (
                <Cell
                  key={row.name}
                  fill={row.spread > 0 ? POSITIVE_BAR : row.spread < 0 ? NEGATIVE_BAR : FLAT_BAR}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function EconomicGrowth() {
  const [equityView, setEquityView] = useState<EquityView>("table")
  const [selectedEquityPeriod, setSelectedEquityPeriod] = useState("1-mo")
  const { data, isLoading, error } = useApiQuery(
    ["economic-growth"],
    fetchEconomicGrowth,
  )

  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } = useSessionAiOverview("ai-overview:economic-growth")
  const mutation = useMutation({
    mutationFn: analyzeEconomicGrowth,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })

  if (isLoading) return <LoadingSpinner message="Fetching economic growth data..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load"} />

  const periods: string[] = data.equity_periods ?? ["1-mo", "3-mo", "6-mo", "1-yr"]
  const currencyPeriods: string[] = data.currency_periods ?? ["1-mo", "3-mo", "6-mo", "1-yr"]
  const activeEquityPeriod = periods.includes(selectedEquityPeriod)
    ? selectedEquityPeriod
    : periods[0] ?? "1-mo"

  const periodCols = (periods_: string[], colorFn: (v: unknown) => string): ColumnDef[] =>
    periods_.map(p => ({
      key: p,
      header: p,
      colorFn: colorFn,
    }))

  function formatReturn(value: number | null | undefined) {
    return value !== null && value !== undefined ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}%` : "N/A"
  }

  function formatRelativeReturn(value: number | null | undefined, relative: number | null | undefined) {
    const formatted = formatReturn(value)
    if (formatted === "N/A" || relative === null || relative === undefined) return formatted
    return `${formatted} (${relative >= 0 ? "+" : ""}${relative.toFixed(1)} pp)`
  }

  function resolveEquityBenchmark(name: string) {
    if (BENCHMARK_ROWS.has(name)) return null
    const benchmarks = data.benchmarks ?? {}
    return benchmarks[name] ?? benchmarks.default ?? "S&P 500"
  }

  function buildRelativeReturns(dict: ReturnTable, periods_: string[]) {
    const provided = data.equity_relative_returns
    if (provided && typeof provided === "object") return provided as ReturnTable

    return Object.fromEntries(Object.entries(dict).map(([name, returns]) => {
      const benchmarkName = resolveEquityBenchmark(name)
      const benchmarkReturns = benchmarkName ? dict[benchmarkName] : null
      const relative: Record<string, number | null> = {}

      periods_.forEach(p => {
        const value = returns[p]
        const benchmark = benchmarkReturns?.[p]
        relative[p] = value !== null && value !== undefined && benchmark !== null && benchmark !== undefined
          ? Number((value - benchmark).toFixed(1))
          : null
      })

      return [name, relative]
    }))
  }

  // Build rows from nested dict {name: {period: value}}
  function buildRows(
    dict: ReturnTable,
    nameKey: string,
    periods_: string[],
    relativeReturns?: ReturnTable,
  ) {
    return Object.entries(dict).map(([name, returns]) => {
      const row: Record<string, unknown> = { [nameKey]: name }
      periods_.forEach(p => {
        const val = returns[p]
        row[p] = relativeReturns ? formatRelativeReturn(val, relativeReturns[name]?.[p]) : formatReturn(val)
      })
      return row
    })
  }

  function buildEquityChartRows(
    dict: ReturnTable,
    relativeReturns: ReturnTable,
    period: string,
  ) {
    return Object.entries(dict)
      .filter(([name]) => !BENCHMARK_ROWS.has(name))
      .map(([name, returns]) => {
        const benchmark = resolveEquityBenchmark(name)
        const returnValue = returns[period]
        const benchmarkReturn = benchmark ? dict[benchmark]?.[period] : null
        const spread = relativeReturns[name]?.[period]

        if (
          !benchmark ||
          returnValue === null ||
          returnValue === undefined ||
          benchmarkReturn === null ||
          benchmarkReturn === undefined ||
          spread === null ||
          spread === undefined
        ) {
          return null
        }

        return {
          name,
          benchmark,
          returnValue,
          benchmarkReturn,
          spread,
        }
      })
      .filter((row): row is EquityChartRow => row !== null)
      .sort((a, b) => b.spread - a.spread)
  }

  const commodityRows = buildRows(data.commodities ?? {}, "Name", periods)
  const equityRelativeReturns = buildRelativeReturns(data.equities ?? {}, periods)
  const equityRows = buildRows(data.equities ?? {}, "Name", periods, equityRelativeReturns)
  const equityChartRows = buildEquityChartRows(data.equities ?? {}, equityRelativeReturns, activeEquityPeriod)
  const currencyRows = buildRows(data.currencies ?? {}, "Pair", currencyPeriods)

  const nameCol = (key: string): ColumnDef => ({ key, header: key, width: "160px" })

  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError)

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Economic Growth Dashboard</h1>
          {data.timestamp && (
            <p className="text-sm text-gray-400 mt-0.5">
              As of {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              mutation.mutate({
                commodities: data.commodities ?? {},
                equities: data.equities ?? {},
                currencies: data.currencies ?? {},
                equity_periods: periods,
                currency_periods: currencyPeriods,
              })
              setIsOpen(true)
            }}
            disabled={mutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton queryKeys={[["economic-growth"]]} />
        </div>
      </div>

      {showPanel && (
        <div className="mb-6 rounded-xl border border-blue-200 bg-white overflow-hidden">
          <button
            onClick={() => setIsOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 bg-blue-50 hover:bg-blue-100 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-blue-500" />
              <span className="text-sm font-semibold text-blue-700">AI Overview</span>
            </div>
            <ChevronDown
              size={16}
              className={`text-blue-500 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            />
          </button>

          {isOpen && (
            <div className="px-4 py-4">
              {mutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  Analyzing market data...
                </div>
              )}
              {mutation.isError && (
                <p className="text-sm text-red-600">
                  {String(mutation.error) || "Analysis failed. Please try again."}
                </p>
              )}
              {analysisText && (
                <p className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                  {analysisText}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Commodities</h2>
        <DataTable
          columns={[nameCol("Name"), ...periodCols(periods, colorPositiveNegative)]}
          rows={commodityRows}
        />
      </section>

      <section className="mb-8">
        <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400">Equities vs Benchmark</h2>
          <SegmentedControl
            options={EQUITY_VIEW_OPTIONS}
            value={equityView}
            onChange={setEquityView}
            size="sm"
          />
        </div>

        {equityView === "table" ? (
          <>
            <DataTable
              columns={[nameCol("Name"), ...periodCols(periods, colorReturnVsBenchmark)]}
              rows={equityRows}
            />
            <p className="text-xs text-gray-400 mt-1.5">
              Parentheses show percentage-point return spread vs benchmark. Positive = outperforming, negative = underperforming.
            </p>
          </>
        ) : (
          <div className="space-y-3">
            <div className="flex justify-end">
              <SegmentedControl
                options={periods.map(p => ({ value: p, label: p }))}
                value={activeEquityPeriod}
                onChange={setSelectedEquityPeriod}
                size="sm"
              />
            </div>
            <EquityRelativePerformanceChart rows={equityChartRows} />
          </div>
        )}
      </section>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Currencies</h2>
        <DataTable
          columns={[nameCol("Pair"), ...periodCols(currencyPeriods, colorPositiveNegative)]}
          rows={currencyRows}
        />
      </section>
    </div>
  )
}
