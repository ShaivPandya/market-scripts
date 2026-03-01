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

  // Dynamically build columns from first row
  const columns: ColumnDef[] = rows.length > 0
    ? Object.keys(rows[0]).map(k => ({
        key: k,
        header: k,
        colorFn: (v: unknown) => {
          if (typeof v === "number" || (typeof v === "string" && !isNaN(parseFloat(v)))) {
            return colorPositiveNegative(v)
          }
          return ""
        },
        format: (v: unknown) =>
          typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(2) : String(v ?? "N/A"),
      }))
    : []

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Momentum</h1>
        <RefreshButton queryKeys={[["momentum"]]} />
      </div>

      {isLoading && <LoadingSpinner message="Fetching momentum data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}
      {data && !isLoading && rows.length === 0 && <p className="text-gray-400">No data returned.</p>}
      {data && !isLoading && rows.length > 0 && (
        <DataTable columns={columns} rows={rows} />
      )}
    </div>
  )
}
