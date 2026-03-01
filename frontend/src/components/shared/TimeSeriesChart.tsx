import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"

export interface DataPoint {
  date: string
  value: number | null
}

interface TimeSeriesChartProps {
  data: DataPoint[]
  height?: number
  color?: string
  label?: string
  /** If true, draw a horizontal reference line at y=0 */
  zeroLine?: boolean
  /** Format y-axis ticks */
  yFormatter?: (v: number) => string
  /** Format tooltip values */
  tooltipFormatter?: (v: number) => string
}

function shortDate(isoDate: string): string {
  try {
    const d = new Date(isoDate)
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  } catch {
    return isoDate
  }
}

export function TimeSeriesChart({
  data,
  height = 200,
  color = "#1f77b4",
  label,
  zeroLine = false,
  yFormatter,
  tooltipFormatter,
}: TimeSeriesChartProps) {
  if (!data || data.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-gray-400 text-sm"
      >
        No data
      </div>
    )
  }

  return (
    <div>
      {label && <p className="text-xs text-gray-500 mb-1 font-medium">{label}</p>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tickFormatter={shortDate}
            tick={{ fontSize: 10 }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={zeroLine ? [(dataMin: number) => Math.min(0, dataMin), "auto"] : ["auto", "auto"]}
            tick={{ fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={50}
            tickFormatter={yFormatter}
          />
          <Tooltip
            labelFormatter={(l: unknown) => new Date(String(l)).toLocaleDateString()}
            formatter={(v: unknown) => {
              const n = v as number | undefined
              return tooltipFormatter && n != null ? tooltipFormatter(n) : n?.toFixed(2) ?? ""
            }}
          />
          {zeroLine && <ReferenceLine y={0} stroke="#ccc" strokeDasharray="4 2" />}
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            dot={false}
            strokeWidth={1.8}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
