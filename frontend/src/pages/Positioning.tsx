import { useMemo, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPositioningSummary, fetchPositioningTimeseries, fetchPositioningInstruments, analyzePositioning } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { TimeSeriesChart } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SelectInput } from "@/components/shared/FormControls"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { colorPositiveNegative, colorZscore, colorForcedFlow } from "@/lib/colors"

const DEFAULT_INSTRUMENTS = ["SP500", "NASDAQ", "RUSSELL", "US10Y", "EUR"]

const formatForcedFlow = (v: unknown): string => {
  if (typeof v !== "string" || !v) return "—"
  return v.replace(/_/g, " ").replace(/^\w/, c => c.toUpperCase())
}

type View = "summary" | "single"

const summaryCols: ColumnDef[] = [
  { key: "instrument", header: "Instrument" },
  { key: "report_date", header: "Report Date" },
  { key: "lf_net", header: "Net Position", format: v => v != null ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }) : "N/A" },
  { key: "lf_net_pct_oi", header: "Net % Open Int", colorFn: colorPositiveNegative, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)}%` : "N/A" },
  { key: "lf_z", header: "Position Z", colorFn: colorZscore, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}` : "N/A" },
  { key: "lf_deleveraging_z", header: "Delev Z", colorFn: colorZscore, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}` : "N/A" },
  { key: "lf_forced", header: "Forced Flow", colorFn: colorForcedFlow, format: formatForcedFlow },
]

export function Positioning() {
  const [view, setView] = useState<View>("summary")
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>(DEFAULT_INSTRUMENTS)
  const [selectedAlias, setSelectedAlias] = useState<string>("SP500")
  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } = useSessionAiOverview("ai-overview:positioning:summary")
  const mutation = useMutation({
    mutationFn: analyzePositioning,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })

  // Instrument list
  const { data: instrData } = useApiQuery(["positioning-instruments"], fetchPositioningInstruments)
  const allInstruments: string[] = instrData?.instruments ? Object.keys(instrData.instruments) : DEFAULT_INSTRUMENTS

  // Summary data
  const { data: summaryData, isLoading: summaryLoading, error: summaryError } = useApiQuery(
    ["positioning-summary", selectedInstruments.join(",")],
    () => fetchPositioningSummary({ instruments: selectedInstruments.join(","), start: "2015-01-01" }),
    60 * 60 * 1000,
    // Only fetch when view is summary
  )

  // Single instrument data
  const instrumentsMap: Record<string, string> = instrData?.instruments ?? {}
  const marketName = instrumentsMap[selectedAlias] ?? ""
  const { data: tsData, isLoading: tsLoading, error: tsError } = useApiQuery(
    ["positioning-ts", selectedAlias],
    () => fetchPositioningTimeseries({ market: marketName, start: "2015-01-01" }),
    60 * 60 * 1000,
  )

  const summaryRows: Record<string, unknown>[] = Array.isArray(summaryData) ? summaryData : []

  // Register screen context for agent chat
  const screenCtx = useMemo(() => {
    if (summaryRows.length === 0) return null
    const metrics: Record<string, string> = {}
    const instrParts = summaryRows.slice(0, 8).map(r => {
      const instr = String(r.instrument ?? "")
      const pctOi = r.lf_net_pct_oi != null ? `${Number(r.lf_net_pct_oi).toFixed(1)}%` : "N/A"
      const z = r.lf_z != null ? `z=${Number(r.lf_z).toFixed(2)}` : ""
      const forced = r.lf_forced ? ` ${String(r.lf_forced)}` : ""
      return `${instr}(${pctOi} ${z}${forced})`
    })
    if (instrParts.length > 0) metrics["Instruments"] = instrParts.join(", ")
    return {
      pageName: "Positioning",
      metrics,
      filters: { view, instruments: selectedInstruments.join(", ") },
      summary: `CFTC positioning for ${selectedInstruments.length} instruments, ${view} view`,
      correspondingTools: ["get_positioning"],
    }
  }, [summaryRows, view, selectedInstruments])
  useRegisterScreenContext(screenCtx)

  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError)

  return (
    <div>
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">CFTC Positioning</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            COT participant positioning + forced-flow proxies via the CFTC PRE/Socrata API
          </p>
        </div>
        <div className="flex w-full flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center md:w-auto md:justify-end">
          {view === "summary" && (
            <button
              onClick={() => {
                mutation.mutate({ rows: summaryRows })
                setIsOpen(true)
              }}
              disabled={mutation.isPending || summaryRows.length === 0}
              className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-600 transition-colors hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
            >
              <Sparkles size={14} />
              AI Overview
            </button>
          )}
          <div className="inline-flex w-full items-center rounded-full bg-gray-100 p-0.5 sm:w-auto">
            {(["summary", "single"] as View[]).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`flex-1 rounded-full px-3.5 py-1.5 text-center text-sm transition-all duration-150 sm:flex-none ${
                  view === v
                    ? "bg-white text-gray-900 font-medium shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {v === "summary" ? "Summary" : "Single Market"}
              </button>
            ))}
          </div>
          <div className="w-full sm:w-auto [&>button]:w-full sm:[&>button]:w-auto">
            <RefreshButton />
          </div>
        </div>
      </div>

      {view === "summary" && (
        <div>
          {showPanel && (
            <div className="mb-5 rounded-xl border border-blue-200 bg-white overflow-hidden">
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
                      Analyzing positioning data...
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

          <div className="flex flex-wrap gap-1.5 mb-4">
            {allInstruments.map(alias => (
              <button
                key={alias}
                onClick={() => setSelectedInstruments(prev =>
                  prev.includes(alias) ? prev.filter(a => a !== alias) : [...prev, alias]
                )}
                className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                  selectedInstruments.includes(alias)
                    ? "bg-gray-800 text-white"
                    : "bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700"
                }`}
              >
                {alias}
              </button>
            ))}
          </div>
          {summaryLoading && <LoadingSpinner message="Fetching positioning summary..." />}
          {!summaryLoading && summaryError && <ErrorMessage message={String(summaryError)} />}
          {!summaryLoading && summaryRows.length > 0 && (
            <DataTable columns={summaryCols} rows={summaryRows} />
          )}
          {!summaryLoading && summaryRows.length === 0 && !summaryError && (
            <p className="text-gray-400 text-sm">No results returned.</p>
          )}
        </div>
      )}

      {view === "single" && (
        <div>
          <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end">
            <SelectInput
              label="Instrument"
              value={selectedAlias}
              onChange={setSelectedAlias}
              options={allInstruments.sort().map(a => ({ value: a, label: a }))}
              className="w-full sm:w-56"
            />
            {marketName && <span className="text-xs text-gray-400 sm:pb-2">{marketName}</span>}
          </div>

          {tsLoading && <LoadingSpinner message="Fetching time series..." />}
          {!tsLoading && tsError && <ErrorMessage message={String(tsError)} />}
          {!tsLoading && Array.isArray(tsData) && tsData.length > 0 && (
            <SingleMarketView rows={tsData as Record<string, unknown>[]} />
          )}
        </div>
      )}
    </div>
  )
}

function SingleMarketView({ rows }: { rows: Record<string, unknown>[] }) {
  const sorted = [...rows].sort((a, b) => String(a["report_date"]).localeCompare(String(b["report_date"])))
  const latest = sorted[sorted.length - 1]

  const mkSeries = (key: string) =>
    sorted
      .filter(r => r[key] != null)
      .map(r => ({ date: String(r["report_date"]), value: Number(r[key]) }))

  const histCols: ColumnDef[] = [
    { key: "report_date", header: "Date" },
    { key: "lf_net", header: "Net Position", format: v => v != null ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }) : "N/A" },
    { key: "lf_net_pct_oi", header: "Net % OI", colorFn: colorPositiveNegative, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)}%` : "N/A" },
    { key: "lf_z", header: "Pos Z", colorFn: colorZscore, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}` : "N/A" },
    { key: "lf_deleveraging_z", header: "Delev Z", colorFn: colorZscore, format: v => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}` : "N/A" },
    { key: "lf_forced", header: "Forced Flow", colorFn: colorForcedFlow, format: formatForcedFlow },
  ]

  return (
    <div>
      <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Latest Reading</h2>
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <MetricCard title="Report Date" value={String(latest["report_date"] ?? "N/A")} />
        <MetricCard title="Net Position" value={latest["lf_net"] != null ? Number(latest["lf_net"]).toLocaleString(undefined, { maximumFractionDigits: 0 }) : "N/A"} />
        <MetricCard title="Net % Open Interest" value={latest["lf_net_pct_oi"] != null ? `${Number(latest["lf_net_pct_oi"]) >= 0 ? "+" : ""}${Number(latest["lf_net_pct_oi"]).toFixed(1)}%` : "N/A"} />
        <MetricCard title="Z-Score" value={latest["lf_z"] != null ? `${Number(latest["lf_z"]) >= 0 ? "+" : ""}${Number(latest["lf_z"]).toFixed(2)}` : "N/A"} />
        <MetricCard title="Deleveraging Z" value={latest["lf_deleveraging_z"] != null ? `${Number(latest["lf_deleveraging_z"]) >= 0 ? "+" : ""}${Number(latest["lf_deleveraging_z"]).toFixed(2)}` : "N/A"} />
        <MetricCard title="Forced Flow" value={formatForcedFlow(latest["lf_forced"])} />
      </div>

      <div className="space-y-4 mb-6">
        <TimeSeriesChart data={mkSeries("lf_net")} height={180} label="Net Position Over Time" zeroLine />
        <TimeSeriesChart data={mkSeries("lf_net_pct_oi")} height={180} label="Net % Open Interest" zeroLine />
        <TimeSeriesChart data={mkSeries("lf_z")} height={180} label="Z-Score" zeroLine />
        <TimeSeriesChart data={mkSeries("lf_deleveraging_z")} height={180} label="Deleveraging Z (forced flows proxy)" zeroLine />
      </div>

      <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Recent History</h2>
      <DataTable columns={histCols} rows={sorted.slice(-20)} />
    </div>
  )
}
