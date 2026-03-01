import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPositioningSummary, fetchPositioningTimeseries, fetchPositioningInstruments } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { TimeSeriesChart } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
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

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">CFTC Positioning</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            COT participant positioning + forced-flow proxies via the CFTC PRE/Socrata API
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <div className="inline-flex items-center rounded-full bg-gray-100 p-0.5">
            {(["summary", "single"] as View[]).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3.5 py-1.5 text-sm rounded-full transition-all duration-150 ${
                  view === v
                    ? "bg-white text-gray-900 font-medium shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {v === "summary" ? "Summary" : "Single Market"}
              </button>
            ))}
          </div>
          <RefreshButton />
        </div>
      </div>

      {view === "summary" && (
        <div>
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
          <div className="flex gap-3 items-center mb-6">
            <label className="text-sm text-gray-500">Instrument:</label>
            <select
              value={selectedAlias}
              onChange={e => setSelectedAlias(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
            >
              {allInstruments.sort().map(a => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
            {marketName && <span className="text-xs text-gray-400">{marketName}</span>}
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
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
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
