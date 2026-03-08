import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchIndustryMonitor } from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { colorEconomicSignal, colorSentiment } from "@/lib/colors"

const SIGNAL_CLASSES: Record<string, string> = {
  expanding: "bg-green-100 text-green-700",
  stable: "bg-blue-100 text-blue-700",
  slowing: "bg-yellow-100 text-yellow-700",
  contracting: "bg-red-100 text-red-700",
}

const SENTIMENT_CLASSES: Record<string, string> = {
  bullish: "bg-green-100 text-green-700",
  neutral: "bg-gray-100 text-gray-600",
  bearish: "bg-red-100 text-red-700",
}

interface Company {
  ticker: string
  company_name: string
  sector: string
  sub_sector: string
  sentiment: string
  quarter: number
  year: number
  summary_headline: string
  demand_trends: string
  pricing_commentary: string
  guidance_outlook: string
  macro_quotes: string[]
  price_reaction_2d: number | null
  is_stale: boolean
}

interface SectorSummary {
  sector_headline: string
  key_themes: string[]
  economic_signal: string
  fresh_companies: number
  total_companies: number
}

interface Sector {
  type: string
  sector_summary: SectorSummary
  companies: Company[]
}

export function IndustryMonitor() {
  const [refresh, setRefresh] = useState(false)
  const { data, isLoading, error } = useApiQuery(
    ["industry-monitor", refresh],
    () => fetchIndustryMonitor(refresh),
    60 * 60 * 1000,
  )

  const bySector: Record<string, Sector> = data?.by_sector ?? {}
  const sectors = Object.entries(bySector)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Industry Monitor</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setRefresh(r => !r)}
            className="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-200 bg-white text-gray-700 shadow-sm hover:bg-gray-50 transition-colors"
          >
            Refresh Data
          </button>
        </div>
      </div>
      <p className="text-xs text-gray-400 mb-4">
        Macro signals from earnings call transcripts — Housing, Trucking, Banks, Retail, Capital Goods
      </p>

      {isLoading && <LoadingSpinner message="Fetching industry data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <div className="space-y-6">
          {sectors.map(([sectorName, sectorData]) => (
            <SectorCard key={sectorName} name={sectorName} data={sectorData} />
          ))}
        </div>
      )}
    </div>
  )
}

function SectorCard({ name, data }: { name: string; data: Sector }) {
  const [expanded, setExpanded] = useState(false)
  const summary = data.sector_summary
  const signal = summary?.economic_signal ?? "stable"
  const signalClass = SIGNAL_CLASSES[signal] ?? "bg-gray-100 text-gray-600"

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-gray-800">{name}</h2>
            <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-500 font-medium">
              {data.type}
            </span>
            <span className={`text-xs px-2 py-0.5 rounded font-medium capitalize ${signalClass}`}>
              {signal}
            </span>
          </div>
          <span className="text-xs text-gray-400">
            {summary?.fresh_companies ?? 0} / {summary?.total_companies ?? 0} companies
          </span>
        </div>
        {summary?.sector_headline && (
          <p className="text-sm text-gray-700 mb-2">{summary.sector_headline}</p>
        )}
        {(summary?.key_themes ?? []).length > 0 && (
          <div className="flex flex-wrap gap-1">
            {summary.key_themes.map((t, i) => (
              <span key={i} className="text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-600">{t}</span>
            ))}
          </div>
        )}
      </div>

      <div className="px-5 py-3">
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-blue-500 hover:underline"
        >
          {expanded ? `Hide ${data.companies?.length ?? 0} companies` : `Show ${data.companies?.length ?? 0} companies`}
        </button>

        {expanded && (data.companies ?? []).map((company, i) => (
          <CompanyRow key={i} company={company} />
        ))}
      </div>
    </div>
  )
}

function CompanyRow({ company }: { company: Company }) {
  const [expanded, setExpanded] = useState(false)
  const sentClass = SENTIMENT_CLASSES[company.sentiment] ?? "bg-gray-100 text-gray-600"

  return (
    <div className="py-3 border-t border-gray-100 first:border-t-0">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs font-bold text-gray-700">{company.ticker}</span>
          <span className="text-sm text-gray-600">{company.company_name}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded capitalize ${sentClass}`}>
            {company.sentiment}
          </span>
          {company.is_stale && (
            <span className="text-xs text-yellow-600 bg-yellow-50 px-1.5 py-0.5 rounded">stale</span>
          )}
        </div>
        <div className="text-xs text-gray-400">
          Q{company.quarter} {company.year}
          {company.price_reaction_2d != null && (
            <span className={`ml-2 ${colorSentiment(company.price_reaction_2d > 0 ? "bullish" : "bearish")}`}>
              {company.price_reaction_2d >= 0 ? "+" : ""}{company.price_reaction_2d.toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {company.summary_headline && (
        <p className="text-xs text-gray-600 mt-1">{company.summary_headline}</p>
      )}

      <button
        onClick={() => setExpanded(e => !e)}
        className="text-xs text-blue-400 hover:underline mt-1"
      >
        {expanded ? "Less" : "More"}
      </button>

      {expanded && (
        <div className="mt-2 space-y-1 text-xs text-gray-600">
          {company.demand_trends && <p><strong>Demand:</strong> {company.demand_trends}</p>}
          {company.pricing_commentary && <p><strong>Pricing:</strong> {company.pricing_commentary}</p>}
          {company.guidance_outlook && <p><strong>Guidance:</strong> {company.guidance_outlook}</p>}
          {(company.macro_quotes ?? []).length > 0 && (
            <ul className="list-disc list-inside">
              {company.macro_quotes.map((q, i) => <li key={i} className="italic">"{q}"</li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
