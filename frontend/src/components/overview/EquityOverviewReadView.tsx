import { useState } from "react"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { MetricCard } from "@/components/shared/MetricCard"
import { SegmentedControl } from "@/components/shared/FormControls"
import { useApiQuery } from "@/hooks/useApiQuery"
import { runFinancials } from "@/lib/api"
import { cleanDossierDisplayText } from "@/lib/dossierText"
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

function currencyPrefix(currency: string): string {
  const normalized = currency.toUpperCase()
  if (normalized === "USD") return "$"
  if (normalized === "TWD") return "NT$"
  return `${normalized} `
}

function formatRevenue(v: unknown, currency = "USD"): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  const prefix = currencyPrefix(currency)
  if (Math.abs(v) >= 1e9) return `${prefix}${(v / 1e9).toFixed(2)}B`
  if (Math.abs(v) >= 1e6) return `${prefix}${(v / 1e6).toFixed(2)}M`
  return `${prefix}${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
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
    ["financials-overview-v10", ticker],
    () => runFinancials({ ticker }),
    300_000,
  )

  const metrics = (rawData?.metrics ?? {}) as Record<string, unknown>
  const annual = (rawData?.annual ?? {}) as Record<string, unknown>
  const quarterly = (rawData?.quarterly ?? {}) as Record<string, unknown>
  const revenueRows = (view === "annual" ? annual.revenue : quarterly.revenue) as FinancialRow[] | undefined
  const epsRows = (view === "annual" ? annual.eps : quarterly.eps) as FinancialRow[] | undefined
  const dataSource = typeof rawData?.data_source === "string" ? rawData.data_source : "sec_edgar"
  const financialCurrency = typeof rawData?.financial_currency === "string" ? rawData.financial_currency : "USD"
  const revenueFormatter = (v: unknown) => formatRevenue(v, financialCurrency)

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-app">Financials</h3>

      {isLoading && <LoadingSpinner message="Loading live financials..." />}
      {error && <p className="text-xs text-red-500">Live financials unavailable: {String(error)}</p>}
      {rawData && (
        <>
          {dataSource === "yfinance" && (
            <p className="text-xs text-subtle">
              Yahoo Finance fallback. Quarterly YoY metrics and filing breakdown are unavailable.
            </p>
          )}
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
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(revenueRows ?? [], revenueFormatter)} maxHeight="300px" label="Revenue" />
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(epsRows ?? [], formatEps)} maxHeight="300px" label="EPS" />
          </div>
        </>
      )}

      {!rawData && !isLoading && parsed && (
        <div className="space-y-2 text-sm text-muted">
          {parsed.revenue_growth && <p><span className="font-medium text-subtle">Revenue Growth:</span> {cleanDossierDisplayText(parsed.revenue_growth.context)}</p>}
          {parsed.eps_growth && <p><span className="font-medium text-subtle">EPS Growth:</span> {cleanDossierDisplayText(parsed.eps_growth.context)}</p>}
        </div>
      )}

      {parsed?.debt && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase text-subtle">Debt</h4>
          <p className="text-sm text-muted">{cleanDossierDisplayText(parsed.debt.summary)}</p>
          {parsed.debt.tranches.length > 0 && (
            <DataTable
              columns={debtCols}
              rows={parsed.debt.tranches.map(t => ({
                tranche: cleanDossierDisplayText(t.tranche),
                rate: cleanDossierDisplayText(t.rate),
                maturity: cleanDossierDisplayText(t.maturity),
              }))}
              maxHeight="200px"
            />
          )}
        </div>
      )}
      {parsed?.reinvestment && (
        <div className="space-y-1">
          <h4 className="text-xs font-semibold uppercase text-subtle">Reinvestment Costs</h4>
          <p className="text-sm text-muted">{cleanDossierDisplayText(parsed.reinvestment)}</p>
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
          const force = cleanDossierDisplayText(f.force)
          const rating = cleanDossierDisplayText(f.rating)
          const description = cleanDossierDisplayText(f.description)
          const cfg = PORTER_RATING_CONFIG[rating] ?? PORTER_RATING_CONFIG.Medium
          return (
            <div key={force || f.force}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-sm font-medium text-app">{force}</span>
                <span className={cn("text-xs font-semibold", cfg.text)}>{rating}</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div className={cn("h-2 rounded-full transition-all", cfg.bg)} style={{ width: cfg.width }} />
              </div>
              <p className="mt-1 text-xs text-muted">{description}</p>
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
      <DataTable
        columns={sensitivityCols}
        rows={rows.map(r => ({
          factor: cleanDossierDisplayText(r.factor),
          sensitivity: cleanDossierDisplayText(r.sensitivity),
          capacity: cleanDossierDisplayText(r.capacity),
        }))}
        maxHeight="400px"
      />
    </div>
  )
}

type OutlookRatingLabel = "Strong" | "Medium" | "Weak" | "Not rated"
type CleanOutlookPoint = { label: string; text: string }

const OUTLOOK_BADGE: Record<OutlookRatingLabel, string> = {
  Strong: "border-green-200 bg-green-50 text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-300",
  Medium: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300",
  Weak: "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300",
  "Not rated": "border-app bg-[hsl(var(--muted-2))] text-muted",
}

const STRONG_OUTLOOK_RE = /\b(strong|robust|resilient|durable|improving|expanding|accelerating|recovering|favorable|healthy)\b/i
const WEAK_OUTLOOK_RE = /\b(weak|declining|slowing|contracting|deteriorating|challenged|falling|fell)\b/i
const MEDIUM_OUTLOOK_RE = /\b(medium|moderate|mixed|stabilizing|stable|balanced|gradual|normalizing)\b/i

function canonicalOutlookRating(value?: string | null): OutlookRatingLabel | null {
  const normalized = cleanDossierDisplayText(value).toLowerCase()
  if (!normalized) return null
  if (normalized.includes("strong")) return "Strong"
  if (normalized.includes("weak")) return "Weak"
  if (normalized.includes("medium") || normalized.includes("moderate") || normalized.includes("mixed")) return "Medium"
  return null
}

function normalizeOutlookPoint(point: string | OutlookPoint): CleanOutlookPoint | null {
  if (typeof point === "string") {
    const text = cleanDossierDisplayText(point)
    return text ? { label: "", text } : null
  }
  const label = cleanDossierDisplayText(point.label)
  const text = cleanDossierDisplayText(point.text)
  return label || text ? { label, text } : null
}

function inferOutlookRating(points: CleanOutlookPoint[]): OutlookRatingLabel {
  const text = points.map(p => `${p.label} ${p.text}`).join(" ")
  const hasStrongSignal = STRONG_OUTLOOK_RE.test(text)
  const hasWeakSignal = WEAK_OUTLOOK_RE.test(text)
  const hasMediumSignal = MEDIUM_OUTLOOK_RE.test(text)
  if (hasStrongSignal && !hasWeakSignal) return "Strong"
  if (hasWeakSignal && !hasStrongSignal) return "Weak"
  if (hasStrongSignal || hasWeakSignal || hasMediumSignal) return "Medium"
  return "Not rated"
}

function OutlookRatingBadge({ rating }: { rating: OutlookRatingLabel }) {
  return (
    <span className={cn("inline-flex shrink-0 items-center rounded border px-2 py-0.5 text-xs font-semibold", OUTLOOK_BADGE[rating])}>
      {rating}
    </span>
  )
}

function SupplyDemandOutlookSection({
  supply,
  demand,
}: {
  supply: ParsedOutlookSection | null
  demand: ParsedOutlookSection | null
}) {
  const renderPoints = (points: CleanOutlookPoint[]) =>
    points.map((p, i) => (
      <li key={`${p.label || p.text}-${i}`} className="relative pl-3 text-sm text-muted before:absolute before:left-0 before:top-[9px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">
        {p.label && <span className="font-medium text-app">{p.label}{p.text ? ": " : ""}</span>}
        {p.text}
      </li>
    ))

  const renderSection = (title: string, data: ParsedOutlookSection) => {
    const points = data.points
      .map(normalizeOutlookPoint)
      .filter((point): point is CleanOutlookPoint => Boolean(point))
    if (!points.length) return null
    const rating = canonicalOutlookRating(data.rating) ?? inferOutlookRating(points)

    return (
      <div>
        <div className="mb-2 flex items-center gap-2">
          <h4 className="text-xs font-semibold uppercase text-subtle">{title}</h4>
          <OutlookRatingBadge rating={rating} />
        </div>
        <ul className="space-y-1.5">{renderPoints(points)}</ul>
      </div>
    )
  }

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
