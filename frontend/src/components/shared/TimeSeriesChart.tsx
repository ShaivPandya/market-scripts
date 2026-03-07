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

export interface DataPoint {
  date: string
  value: number | null
}

export interface SeriesDef {
  key: string
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
  /** Timeframe string — drives x-axis tick deduplication and formatting */
  timeframe?: string
  /** For multi-series charts: rows with 'date' + one key per series */
  multiData?: Record<string, unknown>[]
  /** Series definitions for multi-series mode */
  series?: SeriesDef[]
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

function getThisWeekTicks(data: DataPoint[]): string[] {
  const seen = new Set<string>()
  const ticks: string[] = []
  for (const pt of data) {
    const day = pt.date.substring(0, 10)
    if (!seen.has(day)) {
      seen.add(day)
      ticks.push(pt.date)
    }
  }
  return ticks
}

export function calcReturn(data: DataPoint[]): number | null {
  const vals = data.filter(p => p.value != null).map(p => p.value as number)
  if (vals.length < 2) return null
  return (vals[vals.length - 1] - vals[0]) / vals[0] * 100
}

function getYearTicks(data: DataPoint[]): string[] {
  const seen = new Set<string>()
  const ticks: string[] = []
  for (const pt of data) {
    const year = pt.date.substring(0, 4)
    if (!seen.has(year)) {
      seen.add(year)
      ticks.push(pt.date)
    }
  }
  return ticks
}

export function TimeSeriesChart({
  data = [],
  height = 200,
  color = "hsl(var(--accent))",
  label,
  zeroLine = false,
  yFormatter,
  tooltipFormatter,
  timeframe,
  multiData,
  series,
}: TimeSeriesChartProps) {
  const isMulti = multiData != null && series != null
  const chartData = isMulti ? multiData : data

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
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
          <XAxis
            dataKey="date"
            tickFormatter={timeframe === "Monthly" ? shortYear : timeframe === "Weekly" ? shortMonth : shortDate}
            ticks={timeframe === "This Week" ? getThisWeekTicks(data) : timeframe === "Monthly" ? getYearTicks(data) : undefined}
            tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--chart-grid))" }}
            interval={timeframe === "This Week" || timeframe === "Monthly" ? 0 : "preserveStartEnd"}
          />
          <YAxis
            domain={zeroLine ? [(dataMin: number) => Math.min(0, dataMin), "auto"] : ["auto", "auto"]}
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
                  stroke={s.color ?? color}
                  dot={false}
                  strokeWidth={s.strokeWidth ?? 1.5}
                  strokeOpacity={s.opacity ?? 1}
                  strokeDasharray={s.strokeDasharray}
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
          {isMulti && <Legend wrapperStyle={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
