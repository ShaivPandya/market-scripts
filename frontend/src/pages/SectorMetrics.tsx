import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { fetchSectorMetrics, analyzeSectorMetrics, refreshMarketSnapshots } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative } from "@/lib/colors"

const fmtPp = (v: unknown) => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}pp` : "N/A"
const fmtPct = (v: unknown) => v != null ? `${Number(v).toFixed(1)}%` : "N/A"

const columns: ColumnDef[] = [
  { key: "Sector", header: "Sector" },
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
  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } = useSessionAiOverview("ai-overview:sector-metrics")
  const mutation = useMutation({
    mutationFn: analyzeSectorMetrics,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })

  const { data, isLoading, error } = useApiQuery(
    ["sector-metrics"],
    fetchSectorMetrics,
    60 * 60 * 1000,
  )
  const rows = (data?.weights_df ?? []) as Record<string, unknown>[]
  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Sector Metrics</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              mutation.mutate({
                rows,
                timestamp: typeof data?.timestamp === "string" ? data.timestamp : null,
              })
              setIsOpen(true)
            }}
            disabled={mutation.isPending || rows.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton queryKeys={[["sector-metrics"]]} beforeRefetch={refreshMarketSnapshots} />
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
                  Analyzing sector data...
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

      {isLoading && <LoadingSpinner message="Fetching sector metrics..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {data.timestamp && (
            <p className="text-xs text-gray-400 mb-4">As of: {new Date(data.timestamp as string).toLocaleString()}</p>
          )}
          <DataTable columns={columns} rows={rows} />
        </>
      )}
    </div>
  )
}
