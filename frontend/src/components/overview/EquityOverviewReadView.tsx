import { useState } from "react"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { MetricCard } from "@/components/shared/MetricCard"
import { SegmentedControl } from "@/components/shared/FormControls"
import { useApiQuery } from "@/hooks/useApiQuery"
import { runFinancials } from "@/lib/api"
import type {
  OutlookPoint,
  ParsedFinancials,
  ParsedOutlookSection,
  ParsedOverview,
  PorterForce,
  SensitivityRow,
} from "@/lib/overviewTypes"
import { cn } from "@/lib/utils"

type FinancialViewMode = "annual" | "quarterly"

type FinancialRow = {
  period_label?: string
  period_end?: string
  value?: number | null
  yoy_growth?: number | null
  form?: string
  filed?: string
  filing_url?: string
}

function formatPct(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`
}

function formatRevenue(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function formatEps(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`
}

const revenueHistoryCols: ColumnDef[] = [
  { key: "period_label", header: "Period" },
  { key: "period_end", header: "Period End" },
  { key: "value_str", header: "Value" },
  { key: "yoy_str", header: "YoY", colorFn: (_, row) => {
    const g = row.yoy_growth as number | null
    if (typeof g !== "number") return ""
    return g >= 0 ? "#16a34a" : "#dc2626"
  }},
  { key: "filing_info", header: "Filing" },
]

function mapFinancialRows(rows: FinancialRow[], valueFmt: (v: unknown) => string) {
  return rows.map(r => ({
    period_label: r.period_label ?? "N/A",
    period_end: r.period_end ?? "N/A",
    value_str: valueFmt(r.value),
    yoy_growth: r.yoy_growth,
    yoy_str: formatPct(r.yoy_growth),
    filing_info: [r.form, r.filed].filter(Boolean).join(" / "),
  }))
}

const debtCols: ColumnDef[] = [
  { key: "tranche", header: "Tranche" },
  { key: "rate", header: "Rate" },
  { key: "maturity", header: "Maturity" },
]

function FinancialsSection({ ticker, parsed }: { ticker: string; parsed: ParsedFinancials | null }) {
  const [view, setView] = useState<FinancialViewMode>("annual")

  const { data: rawData, isLoading, error } = useApiQuery<Record<string, unknown>>(
    ["financials-overview-v9", ticker],
    () => runFinancials({ ticker }),
    300_000,
  )

  const metrics = (rawData?.metrics ?? {}) as Record<string, unknown>
  const annual = (rawData?.annual ?? {}) as Record<string, unknown>
  const quarterly = (rawData?.quarterly ?? {}) as Record<string, unknown>
  const revenueRows = (view === "annual" ? annual.revenue : quarterly.revenue) as FinancialRow[] | undefined
  const epsRows = (view === "annual" ? annual.eps : quarterly.eps) as FinancialRow[] | undefined

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-app">Financials</h3>

      {isLoading && <LoadingSpinner message="Loading SEC EDGAR financials..." />}
      {error && <p className="text-xs text-red-500">Live financials unavailable: {String(error)}</p>}
      {rawData && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard title="3Y Revenue CAGR" value={formatPct(metrics.revenue_cagr_3y)} />
            <MetricCard title="3Y EPS CAGR" value={formatPct(metrics.eps_cagr_3y)} />
            <MetricCard title="Avg YoY Revenue (3Q)" value={formatPct(metrics.avg_yoy_revenue_growth_3q)} />
            <MetricCard title="Avg YoY EPS (3Q)" value={formatPct(metrics.avg_yoy_eps_growth_3q)} />
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-subtle">History</span>
            <SegmentedControl
              options={[
                { value: "annual" as const, label: "Annual" },
                { value: "quarterly" as const, label: "Quarterly" },
              ]}
              value={view}
              onChange={setView}
              size="sm"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(revenueRows ?? [], formatRevenue)} maxHeight="300px" label="Revenue" />
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(epsRows ?? [], formatEps)} maxHeight="300px" label="EPS" />
          </div>
        </>
      )}

      {!rawData && !isLoading && parsed && (
        <div className="space-y-2 text-sm text-muted">
          {parsed.revenue_growth && <p><span className="font-medium text-subtle">Revenue Growth:</span> {parsed.revenue_growth.context}</p>}
          {parsed.eps_growth && <p><span className="font-medium text-subtle">EPS Growth:</span> {parsed.eps_growth.context}</p>}
        </div>
      )}

      {parsed?.debt && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase text-subtle">Debt</h4>
          <p className="text-sm text-muted">{parsed.debt.summary}</p>
          {parsed.debt.tranches.length > 0 && (
            <DataTable columns={debtCols} rows={parsed.debt.tranches.map(t => ({ ...t }))} maxHeight="200px" />
          )}
        </div>
      )}
      {parsed?.reinvestment && (
        <div className="space-y-1">
          <h4 className="text-xs font-semibold uppercase text-subtle">Reinvestment Costs</h4>
          <p className="text-sm text-muted">{parsed.reinvestment}</p>
        </div>
      )}
    </div>
  )
}

const PORTER_RATING_CONFIG: Record<string, { width: string; bg: string; text: string }> = {
  Low: { width: "20%", bg: "bg-green-500", text: "text-green-700 dark:text-green-400" },
  "Low-Medium": { width: "35%", bg: "bg-lime-500", text: "text-lime-700 dark:text-lime-400" },
  Medium: { width: "50%", bg: "bg-yellow-500", text: "text-yellow-700 dark:text-yellow-400" },
  "Medium-High": { width: "70%", bg: "bg-orange-500", text: "text-orange-700 dark:text-orange-400" },
  High: { width: "90%", bg: "bg-red-500", text: "text-red-700 dark:text-red-400" },
}

function PortersForcesSection({ forces }: { forces: PorterForce[] }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-app">Porter&apos;s Five Forces</h3>
      <div className="space-y-4">
        {forces.map(f => {
          const cfg = PORTER_RATING_CONFIG[f.rating] ?? PORTER_RATING_CONFIG.Medium
          return (
            <div key={f.force}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-sm font-medium text-app">{f.force}</span>
                <span className={cn("text-xs font-semibold", cfg.text)}>{f.rating}</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div className={cn("h-2 rounded-full transition-all", cfg.bg)} style={{ width: cfg.width }} />
              </div>
              <p className="mt-1 text-xs text-muted">{f.description}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function sensitivityColor(val: unknown): string {
  const s = String(val).toLowerCase()
  if (s === "low") return "#16a34a"
  if (s === "low-medium") return "#65a30d"
  if (s === "medium") return "#ca8a04"
  if (s === "medium-high") return "#ea580c"
  if (s === "high") return "#dc2626"
  return ""
}

const sensitivityCols: ColumnDef[] = [
  { key: "factor", header: "Factor" },
  { key: "sensitivity", header: "Sensitivity", colorFn: val => sensitivityColor(val) },
  { key: "capacity", header: "Capacity to Deal" },
]

function SensitivitySection({ rows }: { rows: SensitivityRow[] }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-app">Sensitivity to Extrinsic Factors</h3>
      <DataTable columns={sensitivityCols} rows={rows.map(r => ({ ...r }))} maxHeight="400px" />
    </div>
  )
}

const OUTLOOK_BADGE: Record<string, string> = {
  Strong: "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-400",
  Medium: "bg-yellow-50 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400",
  Weak: "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-400",
}

function SupplyDemandOutlookSection({
  supply,
  demand,
}: {
  supply: ParsedOutlookSection | null
  demand: ParsedOutlookSection | null
}) {
  const renderPoints = (points: (string | OutlookPoint)[]) =>
    points.map((p, i) => {
      if (typeof p === "string") {
        return <li key={i} className="relative pl-3 text-sm text-muted before:absolute before:left-0 before:top-[9px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">{p}</li>
      }
      return (
        <li key={i} className="relative pl-3 text-sm text-muted before:absolute before:left-0 before:top-[9px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">
          {p.label && <span className="font-medium text-app">{p.label}: </span>}
          {p.text}
        </li>
      )
    })

  const renderSection = (title: string, data: ParsedOutlookSection) => (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <h4 className="text-xs font-semibold uppercase text-subtle">{title}</h4>
        {data.rating && (
          <span className={cn("rounded px-2 py-0.5 text-xs font-medium", OUTLOOK_BADGE[data.rating] ?? "")}>{data.rating}</span>
        )}
      </div>
      <ul className="space-y-1.5">{renderPoints(data.points)}</ul>
    </div>
  )

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-app">Supply &amp; Demand Outlook</h3>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {supply && renderSection("Supply Outlook", supply)}
        {demand && renderSection("Demand Outlook", demand)}
      </div>
    </div>
  )
}

export function EquityOverviewReadView({
  content,
  parsed,
  ticker,
}: {
  content: string | null
  parsed: ParsedOverview | null
  ticker: string
}) {
  if (parsed) {
    return (
      <div className="space-y-8">
        <FinancialsSection ticker={ticker} parsed={parsed.financials} />
        {parsed.porters_five_forces && <PortersForcesSection forces={parsed.porters_five_forces} />}
        {parsed.sensitivity && <SensitivitySection rows={parsed.sensitivity} />}
        {(parsed.supply_outlook || parsed.demand_outlook) && (
          <SupplyDemandOutlookSection supply={parsed.supply_outlook} demand={parsed.demand_outlook} />
        )}
      </div>
    )
  }

  if (!content) return null

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <MarkdownRenderer content={content} />
    </div>
  )
}
