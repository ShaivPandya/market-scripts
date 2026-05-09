/* eslint-disable react-refresh/only-export-components */
import { useMemo, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts"
import type { LegendPayload, YAxisOrientation } from "recharts"

export interface DataPoint {
  date: string
  value: number | null
}

export interface SeriesDef {
  key: string
  name?: string
  color?: string
  strokeWidth?: number
  /** Maps to strokeOpacity on the Line element */
  opacity?: number
  strokeDasharray?: string
}

interface TimeSeriesChartProps {
  data?: DataPoint[]
  height?: number
  color?: string
  label?: string
  /** If true, draw a horizontal reference line at y=0 */
  zeroLine?: boolean
  /** Format y-axis ticks */
  yFormatter?: (v: number) => string
  /** Format tooltip values */
  tooltipFormatter?: (v: number) => string
  /** Sort tooltip rows from highest to lowest numeric value at the hovered point */
  tooltipSortByValueDesc?: boolean
  /** Timeframe string — drives x-axis tick deduplication and formatting */
  timeframe?: string
  /** For multi-series charts: rows with 'date' + one key per series */
  multiData?: Record<string, unknown>[]
  /** Series definitions for multi-series mode */
  series?: SeriesDef[]
  /** Which side should render the y-axis tick labels */
  yAxisOrientation?: YAxisOrientation
  /** Allow clicking legend items to hide/show multi-series lines */
  toggleableLegend?: boolean
  /** Y-axis scale. Log assumes callers pass only positive visible values. */
  yScale?: "linear" | "log"
}

const DEFAULT_SERIES_COLORS = {
  primary: "hsl(var(--accent))",
  positive: "hsl(var(--positive))",
  negative: "hsl(var(--negative))",
  neutral: "hsl(var(--neutral))",
}

function shortDate(isoDate: string): string {
  try {
    const d = new Date(isoDate)
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  } catch {
    return isoDate
  }
}

function shortMonth(isoDate: string): string {
  try {
    return new Date(isoDate).toLocaleDateString("en-US", { month: "short" })
  } catch {
    return isoDate
  }
}

function shortYear(isoDate: string): string {
  try {
    return String(new Date(isoDate).getFullYear())
  } catch {
    return isoDate
  }
}

type ChartRow = DataPoint | Record<string, unknown>
type TooltipValuePayload = {
  value?: number | string | ReadonlyArray<number | string>
}
interface LegendVisibilityState {
  seriesSignature: string
  hiddenKeys: Set<string>
}

const EMPTY_HIDDEN_KEYS = new Set<string>()

function getRowDate(row: ChartRow): string | null {
  return typeof row.date === "string" ? row.date : null
}

function getThisWeekTicks(data: ChartRow[]): string[] {
  const seen = new Set<string>()
  const ticks: string[] = []
  for (const pt of data) {
    const date = getRowDate(pt)
    if (!date) continue
    const day = date.substring(0, 10)
    if (!seen.has(day)) {
      seen.add(day)
      ticks.push(date)
    }
  }
  return ticks
}

function tooltipValueDescSorter(item: TooltipValuePayload): number {
  const value = Array.isArray(item.value) ? Number(item.value[0]) : Number(item.value)
  return Number.isFinite(value) ? -value : Number.POSITIVE_INFINITY
}

export function calcReturn(data: DataPoint[]): number | null {
  const vals = data.filter(p => p.value != null).map(p => p.value as number)
  if (vals.length < 2) return null
  return (vals[vals.length - 1] - vals[0]) / vals[0] * 100
}

function getMonthTicks(data: ChartRow[]): string[] {
  const seen = new Set<string>()
  const ticks: string[] = []
  for (const pt of data) {
    const date = getRowDate(pt)
    if (!date) continue
    const month = date.substring(0, 7)
    if (!seen.has(month)) {
      seen.add(month)
      ticks.push(date)
    }
  }
  return ticks
}

function getYearTicks(data: ChartRow[]): string[] {
  const seen = new Set<string>()
  const ticks: string[] = []
  for (const pt of data) {
    const date = getRowDate(pt)
    if (!date) continue
    const year = date.substring(0, 4)
    if (!seen.has(year)) {
      seen.add(year)
      ticks.push(date)
    }
  }
  return ticks
}

export function TimeSeriesChart({
  data = [],
  height = 200,
  color = DEFAULT_SERIES_COLORS.primary,
  label,
  zeroLine = false,
  yFormatter,
  tooltipFormatter,
  tooltipSortByValueDesc = false,
  timeframe,
  multiData,
  series,
  yAxisOrientation = "left",
  toggleableLegend = false,
  yScale = "linear",
}: TimeSeriesChartProps) {
  const isMulti = multiData != null && series != null
  const chartData: ChartRow[] = isMulti ? multiData : data
  const seriesSignature = useMemo(
    () => (series ?? []).map(s => `${s.key}:${s.name ?? s.key}`).join("|"),
    [series],
  )
  const [legendVisibility, setLegendVisibility] = useState<LegendVisibilityState>(() => ({
    seriesSignature: "",
    hiddenKeys: new Set(),
  }))
  const hiddenSeries = legendVisibility.seriesSignature === seriesSignature
    ? legendVisibility.hiddenKeys
    : EMPTY_HIDDEN_KEYS

  const toggleLegendSeries = (payload: LegendPayload) => {
    if (!toggleableLegend) return
    const dataKey = payload.dataKey != null ? String(payload.dataKey) : null
    if (!dataKey) return

    setLegendVisibility(prev => {
      const prevHidden = prev.seriesSignature === seriesSignature ? prev.hiddenKeys : EMPTY_HIDDEN_KEYS
      const next = new Set(prevHidden)
      if (next.has(dataKey)) {
        next.delete(dataKey)
      } else {
        next.add(dataKey)
      }
      return { seriesSignature, hiddenKeys: next }
    })
  }

  const formatLegendLabel = (value: unknown, entry: LegendPayload) => {
    const dataKey = entry.dataKey != null ? String(entry.dataKey) : null
    const isHidden = dataKey != null && hiddenSeries.has(dataKey)
    return (
      <span
        style={{
          color: isHidden ? "hsl(var(--muted-foreground))" : entry.color,
          opacity: isHidden ? 0.45 : 1,
          textDecoration: isHidden ? "line-through" : "none",
        }}
      >
        {String(value ?? "")}
      </span>
    )
  }

  if (!chartData || chartData.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-sm text-subtle"
      >
        No data
      </div>
    )
  }

  return (
    <div>
      {label && <p className="mb-1 text-xs font-medium text-muted">{label}</p>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{ top: 4, right: yAxisOrientation === "right" ? 0 : 8, left: yAxisOrientation === "right" ? 8 : 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
          <XAxis
            dataKey="date"
            tickFormatter={timeframe === "Monthly" ? shortYear : timeframe === "Weekly" ? shortMonth : shortDate}
            ticks={
              timeframe === "This Week" ? getThisWeekTicks(chartData) :
              timeframe === "Monthly" ? getYearTicks(chartData) :
              timeframe === "Daily" ? getMonthTicks(chartData) :
              undefined
            }
            tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--chart-grid))" }}
            minTickGap={30}
          />
          <YAxis
            orientation={yAxisOrientation}
            scale={yScale === "log" ? "log" : undefined}
            domain={
              yScale === "log"
                ? ["dataMin", "dataMax"]
                : zeroLine
                  ? [(dataMin: number) => Math.min(0, dataMin), "auto"]
                  : ["auto", "auto"]
            }
            tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
            tickLine={false}
            axisLine={false}
            width={50}
            tickFormatter={yFormatter}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--chart-tooltip-bg))",
              borderColor: "hsl(var(--chart-tooltip-border))",
              borderRadius: "0.75rem",
              color: "hsl(var(--foreground))",
            }}
            labelStyle={{ color: "hsl(var(--foreground))" }}
            itemStyle={{ color: "hsl(var(--foreground))" }}
            labelFormatter={(l: unknown) => new Date(String(l)).toLocaleDateString()}
            itemSorter={tooltipSortByValueDesc ? tooltipValueDescSorter : undefined}
            formatter={(v: unknown) => {
              const n = v as number | undefined
              return tooltipFormatter && n != null ? tooltipFormatter(n) : n?.toFixed(2) ?? ""
            }}
          />
          {zeroLine && <ReferenceLine y={0} stroke="hsl(var(--chart-axis))" strokeDasharray="4 2" />}
          {isMulti
            ? series!.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.name ?? s.key}
                  stroke={s.color ?? color}
                  dot={false}
                  strokeWidth={s.strokeWidth ?? 1.5}
                  strokeOpacity={s.opacity ?? 1}
                  strokeDasharray={s.strokeDasharray}
                  hide={hiddenSeries.has(s.key)}
                  connectNulls={false}
                />
              ))
            : (
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  dot={false}
                  strokeWidth={1.8}
                  connectNulls={false}
                />
              )
          }
          {isMulti && (
            <Legend
              wrapperStyle={{
                fontSize: 11,
                color: "hsl(var(--muted-foreground))",
                cursor: toggleableLegend ? "pointer" : "default",
              }}
              onClick={toggleableLegend ? toggleLegendSeries : undefined}
              formatter={toggleableLegend ? formatLegendLabel : undefined}
              inactiveColor="hsl(var(--muted-foreground))"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
