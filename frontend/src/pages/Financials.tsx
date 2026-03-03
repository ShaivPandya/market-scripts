import { useState } from "react"
import { useQuery } from "@tanstack/react-query"

import { runFinancials } from "@/lib/api"
import { MetricCard } from "@/components/shared/MetricCard"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, TextInput, ActionButton } from "@/components/shared/FormControls"

type ViewMode = "annual" | "quarterly"

type FinancialRow = {
  period_label?: string
  period_end?: string
  value?: number | null
  yoy_growth?: number | null
  form?: string
  filed?: string
  accn?: string
  filing_url?: string
}

type BreakdownRow = {
  label?: string
  value?: number | null
  pct_of_total?: number | null
}

function formatPct(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`
}

function formatPctWhole(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${Math.round(v * 100)}%`
}

function formatRevenue(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function formatRevenueWhole(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  if (Math.abs(v) >= 1e9) return `$${Math.round(v / 1e9).toLocaleString()}B`
  if (Math.abs(v) >= 1e6) return `$${Math.round(v / 1e6).toLocaleString()}M`
  return `$${Math.round(v).toLocaleString()}`
}

function formatEps(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`
}

function formatDate(v: unknown): string {
  if (typeof v !== "string" || !v) return "N/A"
  return v
}

function HistoryTable({
  title,
  rows,
  valueFormatter,
}: {
  title: string
  rows: FinancialRow[]
  valueFormatter: (v: unknown) => string
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
      </div>
      {rows.length === 0 ? (
        <p className="px-4 py-4 text-sm text-gray-400">No data available.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">Period</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">Period End</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">Value</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">YoY</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">Filing</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const periodLabel = r.period_label || "N/A"
                const hasUrl = typeof r.filing_url === "string" && r.filing_url.length > 0
                return (
                  <tr key={`${periodLabel}-${idx}`} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                    <td className="px-3 py-2 whitespace-nowrap">
                      {hasUrl ? (
                        <a
                          href={r.filing_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-blue-700 hover:underline decoration-blue-300 underline-offset-2 font-medium"
                        >
                          {periodLabel}
                        </a>
                      ) : (
                        periodLabel
                      )}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap">{formatDate(r.period_end)}</td>
                    <td className="px-3 py-2 whitespace-nowrap">{valueFormatter(r.value)}</td>
                    <td className={`px-3 py-2 whitespace-nowrap ${typeof r.yoy_growth === "number" ? (r.yoy_growth >= 0 ? "text-green-600" : "text-red-600") : "text-gray-500"}`}>
                      {formatPct(r.yoy_growth)}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-gray-600">
                      {r.form || ""}
                      {r.form && r.filed ? " · " : ""}
                      {r.filed || ""}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const breakdownCols: ColumnDef[] = [
  { key: "label", header: "Label" },
  { key: "value_str", header: "Revenue" },
  { key: "pct_str", header: "% of Total" },
]

export function Financials() {
  const [ticker, setTicker] = useState("")
  const [view, setView] = useState<ViewMode>("annual")

  const [submittedTicker, setSubmittedTicker] = useState<string | null>(null)

  const { data: rawData, isFetching, isError, error } = useQuery({
    queryKey: ["financials", submittedTicker],
    queryFn: () => runFinancials({ ticker: submittedTicker! }),
    enabled: Boolean(submittedTicker),
    staleTime: Infinity,
  })

  const isLoading = isFetching

  function handleRun(e?: React.FormEvent) {
    if (e) e.preventDefault()
    const normalizedTicker = ticker.trim().toUpperCase()
    if (!normalizedTicker) return
    setSubmittedTicker(normalizedTicker)
  }

  const data = rawData as Record<string, unknown> | undefined
  const metrics = (data?.metrics ?? {}) as Record<string, unknown>

  const annual = (data?.annual ?? {}) as Record<string, unknown>
  const quarterly = (data?.quarterly ?? {}) as Record<string, unknown>

  const revenueRows = (view === "annual" ? annual.revenue : quarterly.revenue) as FinancialRow[] | undefined
  const epsRows = (view === "annual" ? annual.eps : quarterly.eps) as FinancialRow[] | undefined

  const breakdown = (data?.breakdown ?? {}) as Record<string, unknown>
  const sourceFiling = (breakdown?.source_filing ?? null) as Record<string, unknown> | null

  const segmentRowsRaw = (breakdown?.by_segment ?? []) as BreakdownRow[]
  const regionRowsRaw = (breakdown?.by_region ?? []) as BreakdownRow[]

  const segmentRows = segmentRowsRaw.map(r => ({
    label: r.label ?? "N/A",
    value_str: formatRevenueWhole(r.value),
    pct_str: formatPctWhole(r.pct_of_total),
  }))
  const regionRows = regionRowsRaw.map(r => ({
    label: r.label ?? "N/A",
    value_str: formatRevenueWhole(r.value),
    pct_str: formatPctWhole(r.pct_of_total),
  }))

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Financials</h1>
      </div>

      <form onSubmit={handleRun} className="flex flex-wrap items-end gap-4 mb-6">
        <TextInput
          label="Ticker"
          value={ticker}
          onChange={v => setTicker(v.toUpperCase())}
          placeholder="AAPL"
          className="w-40"
        />
        <ActionButton type="submit" loading={isLoading} loadingText="Loading..." className="w-auto px-6">
          Analyze
        </ActionButton>
      </form>

      {isLoading && <LoadingSpinner message="Fetching SEC EDGAR financials..." />}
      {isError && <ErrorMessage message={String(error)} />}

      {data && !isLoading && (
        <div className="space-y-6">
          <div>
            <p className="text-sm text-gray-500">
              {(data.company_name as string) || "Company"} ({(data.ticker as string) || ""}) · CIK {(data.cik as string) || "N/A"}
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard title="3Y Revenue CAGR" value={formatPct(metrics.revenue_cagr_3y)} />
            <MetricCard title="3Y EPS CAGR" value={formatPct(metrics.eps_cagr_3y)} />
            <MetricCard title="Avg YoY EPS Growth (3Q)" value={formatPct(metrics.avg_yoy_eps_growth_3q)} />
            <MetricCard title="Avg YoY Revenue Growth (3Q)" value={formatPct(metrics.avg_yoy_revenue_growth_3q)} />
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-1.5">History View</label>
            <SegmentedControl
              options={[
                { value: "annual" as const, label: "Annual" },
                { value: "quarterly" as const, label: "Quarterly" },
              ]}
              value={view}
              onChange={setView}
            />
          </div>

          <div className="space-y-4">
            <HistoryTable title="Revenue" rows={revenueRows ?? []} valueFormatter={formatRevenue} />
            <HistoryTable title="EPS" rows={epsRows ?? []} valueFormatter={formatEps} />
          </div>

          <div className="space-y-4">
            <h2 className="text-base font-semibold">Latest Filing Revenue Breakdown</h2>
            {sourceFiling ? (
              <p className="text-sm text-gray-500">
                Source: {(sourceFiling.form as string) || ""}
                {(sourceFiling.filed as string) ? ` filed ${sourceFiling.filed as string}` : ""}
                {(sourceFiling.accn as string) ? ` · ${sourceFiling.accn as string}` : ""}
                {typeof sourceFiling.filing_url === "string" && sourceFiling.filing_url ? (
                  <>
                    {" "}
                    <a
                      href={sourceFiling.filing_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-700 hover:underline"
                    >
                      Open filing
                    </a>
                  </>
                ) : null}
              </p>
            ) : (
              <p className="text-sm text-gray-400">No filing metadata available.</p>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div>
                <DataTable columns={breakdownCols} rows={segmentRows} maxHeight="420px" label="By Segment" />
              </div>
              <div>
                <DataTable columns={breakdownCols} rows={regionRows} maxHeight="420px" label="By Region" />
              </div>
            </div>
          </div>
        </div>
      )}

      {!data && !isLoading && !isError && (
        <p className="text-gray-400 text-sm">Enter a ticker and click Analyze.</p>
      )}
    </div>
  )
}
