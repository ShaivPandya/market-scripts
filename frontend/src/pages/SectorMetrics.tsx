import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchSectorMetrics } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative } from "@/lib/colors"

const fmtPp = (v: unknown) => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}pp` : "N/A"
const fmtPct = (v: unknown) => v != null ? `${Number(v).toFixed(1)}%` : "N/A"

const columns: ColumnDef[] = [
  { key: "index", header: "Sector" },
  { key: "Weight_Now", header: "Weight Now", format: fmtPct },
  { key: "Chg_1M_pp", header: "1M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Chg_3M_pp", header: "3M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Chg_6M_pp", header: "6M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_1M_pp", header: "Rel Perf 1M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_3M_pp", header: "Rel Perf 3M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_6M_pp", header: "Rel Perf 6M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_12M_pp", header: "Rel Perf 12M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Pct_Above_200DMA", header: "% Above 200DMA", format: fmtPct },
]

export function SectorMetrics() {
  const { data, isLoading, error } = useApiQuery(
    ["sector-metrics"],
    fetchSectorMetrics,
    60 * 60 * 1000,
  )

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Sector Metrics</h1>
        <RefreshButton queryKeys={[["sector-metrics"]]} />
      </div>

      {isLoading && <LoadingSpinner message="Fetching sector metrics..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {data.timestamp && (
            <p className="text-xs text-gray-400 mb-4">As of: {new Date(data.timestamp as string).toLocaleString()}</p>
          )}
          <DataTable columns={columns} rows={(data.weights_df ?? []) as Record<string, unknown>[]} />
        </>
      )}
    </div>
  )
}
