import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchMomentum } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative } from "@/lib/colors"

export function Momentum() {
  const { data, isLoading, error } = useApiQuery(["momentum"], fetchMomentum)

  // The momentum module returns a results_df (or similar key) as a list of records
  const rows: Record<string, unknown>[] = data?.results ?? []

  const longs = rows.filter(r => r["direction"] === "long")
  const shorts = rows.filter(r => r["direction"] === "short")

  const COLUMN_LABELS: Record<string, string> = {
    ticker:          "Ticker",
    avg20_roc63:     "20D Avg ROC (63d)",
    rel_roc42:       "Rel ROC (42d)",
    avg10_rel_roc:   "10D Avg Rel ROC",
    benchmark:       "Benchmark",
    direction:       "Direction",
  }

  const HIDDEN_COLUMNS = new Set(["date", "close", "direction", "avg20_vol_roc63"])
  const NEUTRAL_TEXT_COLUMNS = new Set(["ticker", "benchmark"])
  const NUMERIC_TEXT_RE = /^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?\s*$/

  // Dynamically build columns from first row
  const buildColumns = (sample: Record<string, unknown>): ColumnDef[] =>
    Object.keys(sample).filter(k => !HIDDEN_COLUMNS.has(k)).map(k => ({
      key: k,
      header: COLUMN_LABELS[k] ?? k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
      colorFn: (v: unknown) => {
        if (NEUTRAL_TEXT_COLUMNS.has(k)) return ""
        if (typeof v === "number" || (typeof v === "string" && NUMERIC_TEXT_RE.test(v))) {
          return colorPositiveNegative(v)
        }
        return ""
      },
      format: (v: unknown) =>
        typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(2) : String(v ?? "N/A"),
    }))

  const columns = rows.length > 0 ? buildColumns(rows[0]) : []

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Momentum</h1>
        <RefreshButton queryKeys={[["momentum"]]} />
      </div>

      {isLoading && <LoadingSpinner message="Fetching momentum data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}
      {data && !isLoading && rows.length === 0 && <p className="text-sm text-gray-400">No data returned.</p>}
      {data && !isLoading && longs.length > 0 && (
        <div className="mb-8">
          <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Longs</p>
          <DataTable columns={columns} rows={longs} />
        </div>
      )}
      {data && !isLoading && shorts.length > 0 && (
        <div>
          <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Shorts</p>
          <DataTable columns={columns} rows={shorts} />
        </div>
      )}
    </div>
  )
}
