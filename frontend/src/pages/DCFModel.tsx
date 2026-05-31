import { useState, useEffect, useMemo, type Dispatch, type SetStateAction } from "react"
import { useQuery, useMutation } from "@tanstack/react-query"
import { Download, Plus, Trash2 } from "lucide-react"

import { downloadDCFModel, fetchDCFHistorical, runDCFValuation, type DCFValuationRequest } from "@/lib/api"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, TextInput, ActionButton } from "@/components/shared/FormControls"
import { Notice } from "@/components/shared/Notice"

type TabMode = "historical" | "dcf"

const MIN_PROJECTION_YEARS = 5
const MAX_PROJECTION_YEARS = 8
const DEFAULT_REVENUE_GROWTH = ["10", "8", "7", "6", "5"]
const DEFAULT_EBITDA_MARGIN = "25"
const DEFAULT_TAX_RATE = "21"
const DEFAULT_DA_PCT = "3"
const DEFAULT_NWC_PCT = "5"
const DEFAULT_CAPEX_PCT = "3"

function repeatAssumption(value: string, years = MIN_PROJECTION_YEARS): string[] {
  return Array.from({ length: years }, () => value)
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function fmtB(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return "N/A"
  const abs = Math.abs(v)
  if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}M`
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return "N/A"
  return `${v.toFixed(1)}%`
}

function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return "N/A"
  return `$${v.toFixed(2)}`
}

function fmtX(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return "N/A"
  return `${v.toFixed(1)}x`
}

function fmtUpside(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`
}

function dcfFileNameForTicker(ticker: string): string {
  const clean = ticker.trim().toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "")
  return `${clean || "ticker"}_dcf_model.xlsx`
}

function parseNumberInput(raw: string): number | null {
  const trimmed = raw.trim()
  if (!trimmed) return null
  const value = Number(trimmed)
  return Number.isFinite(value) ? value : null
}

function pctBoundError(
  label: string,
  value: number,
  min: number | null,
  max: number | null,
  minInclusive: boolean,
  maxInclusive: boolean,
): string | null {
  if (min !== null) {
    if (minInclusive && value < min) return `${label} must be at least ${min}%`
    if (!minInclusive && value <= min) return `${label} must be greater than ${min}%`
  }
  if (max !== null) {
    if (maxInclusive && value > max) return `${label} must be at most ${max}%`
    if (!maxInclusive && value >= max) return `${label} must be less than ${max}%`
  }
  return null
}

// ---------------------------------------------------------------------------
// Transposed historical table
// ---------------------------------------------------------------------------

interface TransposedRow {
  label: string
  values: (string | null)[]
  isBold?: boolean
  isHighlight?: boolean
}

function TransposedTable({
  title,
  columns,
  rows,
  avgLabel,
}: {
  title: string
  columns: string[]
  rows: TransposedRow[]
  avgLabel?: string
}) {
  return (
    <div className="rounded-xl border border-border bg-surface shadow-sm overflow-hidden">
      <div className="px-4 py-2.5 border-b border-border bg-muted/30">
        <h3 className="text-sm font-semibold text-app">{title}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-left font-semibold text-muted whitespace-nowrap min-w-[160px]" />
              {columns.map(col => (
                <th
                  key={col}
                  className="px-3 py-2 text-right font-semibold text-muted whitespace-nowrap min-w-[100px]"
                >
                  {col}
                </th>
              ))}
              {avgLabel && (
                <th className="px-3 py-2 text-right font-semibold text-muted whitespace-nowrap min-w-[100px] bg-accent/5">
                  {avgLabel}
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr
                key={row.label + idx}
                className={`border-b border-border/50 hover:bg-muted/20 transition-colors ${row.isHighlight ? "bg-accent/5" : ""}`}
              >
                <td
                  className={`px-3 py-2 whitespace-nowrap ${row.isBold ? "font-semibold text-app" : "text-muted"}`}
                >
                  {row.label}
                </td>
                {row.values.slice(0, columns.length).map((val, ci) => (
                  <td
                    key={ci}
                    className={`px-3 py-2 text-right whitespace-nowrap ${row.isBold ? "font-semibold text-app" : "text-app"}`}
                  >
                    {val ?? "N/A"}
                  </td>
                ))}
                {avgLabel && (
                  <td
                    className={`px-3 py-2 text-right whitespace-nowrap bg-accent/5 ${row.isBold ? "font-bold text-app" : "font-medium text-app"}`}
                  >
                    {row.values[columns.length] ?? ""}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Valuation card
// ---------------------------------------------------------------------------

function ValuationCard({
  label,
  scenario,
  data,
  currentPrice,
}: {
  label: string
  scenario: string
  data: { per_share?: number; upside?: number; error?: string } | undefined
  currentPrice: number
}) {
  if (!data) return null
  if (data.error) {
    return (
      <div className="theme-surface rounded-xl p-4">
        <p className="text-xs font-medium text-muted">{label} ({scenario})</p>
        <p className="mt-1 text-sm text-red-600">{data.error}</p>
      </div>
    )
  }
  const upside = data.upside ?? 0
  const signal = upside > 10 ? "success" : upside < -10 ? "error" : "warning"
  return (
    <MetricCard
      title={`${label} (${scenario})`}
      value={fmtPrice(data.per_share)}
      subtitle={`vs ${fmtPrice(currentPrice)} (${fmtUpside(data.upside)})`}
      signal={signal}
      signalLabel={upside > 0 ? "Upside" : "Downside"}
    />
  )
}

// ---------------------------------------------------------------------------
// Input row helper
// ---------------------------------------------------------------------------

function InputRow({
  label,
  value,
  onChange,
  suffix = "%",
  width = "w-20",
}: {
  label: string
  value: string
  onChange: (v: string) => void
  suffix?: string
  width?: string
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-muted min-w-[140px]">{label}</label>
      <div className="flex items-center gap-1">
        <input
          type="number"
          step="0.1"
          value={value}
          onChange={e => onChange(e.target.value)}
          className={`theme-input rounded-lg px-2 py-1.5 text-sm text-right ${width}`}
        />
        <span className="text-xs text-muted">{suffix}</span>
      </div>
    </div>
  )
}

function ScenarioInputRow({
  label,
  bear,
  base,
  bull,
  onBear,
  onBase,
  onBull,
  suffix = "",
}: {
  label: string
  bear: string
  base: string
  bull: string
  onBear: (v: string) => void
  onBase: (v: string) => void
  onBull: (v: string) => void
  suffix?: string
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-muted min-w-[140px]">{label}</label>
      <div className="flex items-center gap-3">
        {[
          { label: "Bear", value: bear, onChange: onBear },
          { label: "Base", value: base, onChange: onBase },
          { label: "Bull", value: bull, onChange: onBull },
        ].map(s => (
          <div key={s.label} className="flex items-center gap-1">
            <span className="text-xs text-muted w-8">{s.label}</span>
            <input
              type="number"
              step="0.1"
              value={s.value}
              onChange={e => s.onChange(e.target.value)}
              className="theme-input rounded-lg px-2 py-1.5 text-sm text-right w-16"
            />
            {suffix && <span className="text-xs text-muted">{suffix}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/no-explicit-any */

export function DCFModel() {
  const [tab, setTab] = useState<TabMode>("historical")
  const [ticker, setTicker] = useState("")
  const [submittedTicker, setSubmittedTicker] = useState<string | null>(null)

  // Historical query
  const {
    data: hist,
    isFetching,
    isError,
    error,
  } = useQuery({
    queryKey: ["dcf-historical", submittedTicker],
    queryFn: () => fetchDCFHistorical(submittedTicker!),
    enabled: Boolean(submittedTicker),
    staleTime: Infinity,
  })

  // DCF assumptions state
  const [revGrowth, setRevGrowth] = useState(() => [...DEFAULT_REVENUE_GROWTH])
  const [ebitdaMargin, setEbitdaMargin] = useState(() => repeatAssumption(DEFAULT_EBITDA_MARGIN))
  const [taxRate, setTaxRate] = useState(() => repeatAssumption(DEFAULT_TAX_RATE))
  const [daPct, setDaPct] = useState(() => repeatAssumption(DEFAULT_DA_PCT))
  const [nwcPct, setNwcPct] = useState(() => repeatAssumption(DEFAULT_NWC_PCT))
  const [capexPct, setCapexPct] = useState(() => repeatAssumption(DEFAULT_CAPEX_PCT))
  const [wacc, setWacc] = useState("10")

  // Terminal growth (bear/base/bull)
  const [tgrBear, setTgrBear] = useState("2")
  const [tgrBase, setTgrBase] = useState("3")
  const [tgrBull, setTgrBull] = useState("4")

  // EV/EBITDA exit
  const [eveBear, setEveBear] = useState("10")
  const [eveBase, setEveBase] = useState("12")
  const [eveBull, setEveBull] = useState("14")

  // EV/Revenue exit
  const [evrBear, setEvrBear] = useState("3")
  const [evrBase, setEvrBase] = useState("4")
  const [evrBull, setEvrBull] = useState("5")
  const [lastSuccessfulDCFRequest, setLastSuccessfulDCFRequest] = useState<DCFValuationRequest | null>(null)
  const [isDownloadingDCF, setIsDownloadingDCF] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Pre-populate from historical averages
  useEffect(() => {
    if (!hist) return
    const avg = hist.historical_averages
    if (avg) {
      if (avg.ebitda_margin_avg != null) setEbitdaMargin(prev => prev.map(() => String(avg.ebitda_margin_avg)))
      if (avg.da_pct_avg != null) setDaPct(prev => prev.map(() => String(avg.da_pct_avg)))
      if (avg.nwc_pct_avg != null) setNwcPct(prev => prev.map(() => String(avg.nwc_pct_avg)))
      if (avg.capex_pct_avg != null) setCapexPct(prev => prev.map(() => String(avg.capex_pct_avg)))
    }
    const w = hist.wacc_inputs
    if (w?.wacc != null) setWacc(String(w.wacc))
    if (w?.tax_rate != null) setTaxRate(prev => prev.map(() => String(w.tax_rate)))

    // Pre-populate EV/EBITDA exit from historical average
    if (hist.ev_ebitda?.length) {
      const avg20 = hist.ev_ebitda[0]?.avg
      if (avg20 != null) {
        setEveBear(String(Math.round(avg20 * 0.8)))
        setEveBase(String(Math.round(avg20)))
        setEveBull(String(Math.round(avg20 * 1.2)))
      }
    }
    // Pre-populate EV/Revenue exit from historical average
    if (hist.rev_multiple?.length) {
      const avg20 = hist.rev_multiple[0]?.avg
      if (avg20 != null) {
        setEvrBear(String((avg20 * 0.8).toFixed(1)))
        setEvrBase(String(avg20.toFixed(1)))
        setEvrBull(String((avg20 * 1.2).toFixed(1)))
      }
    }
  }, [hist])

  // DCF mutation
  const dcfMutation = useMutation({
    mutationFn: runDCFValuation,
    onSuccess: (_data, variables) => {
      setLastSuccessfulDCFRequest(variables)
      setDownloadError(null)
    },
  })

  const resetProjectionAssumptions = () => {
    setRevGrowth([...DEFAULT_REVENUE_GROWTH])
    setEbitdaMargin(repeatAssumption(DEFAULT_EBITDA_MARGIN))
    setTaxRate(repeatAssumption(DEFAULT_TAX_RATE))
    setDaPct(repeatAssumption(DEFAULT_DA_PCT))
    setNwcPct(repeatAssumption(DEFAULT_NWC_PCT))
    setCapexPct(repeatAssumption(DEFAULT_CAPEX_PCT))
    setWacc("10")
    setTgrBear("2")
    setTgrBase("3")
    setTgrBull("4")
    setEveBear("10")
    setEveBase("12")
    setEveBull("14")
    setEvrBear("3")
    setEvrBase("4")
    setEvrBull("5")
  }

  const updateYearlyAssumption = (
    setter: Dispatch<SetStateAction<string[]>>,
    index: number,
    value: string,
  ) => {
    setter(prev => prev.map((existing, i) => (i === index ? value : existing)))
  }

  const addProjectionYear = () => {
    if (revGrowth.length >= MAX_PROJECTION_YEARS) return
    setRevGrowth(prev => [...prev, prev[prev.length - 1] ?? "5"])
    setEbitdaMargin(prev => [...prev, prev[prev.length - 1] ?? DEFAULT_EBITDA_MARGIN])
    setTaxRate(prev => [...prev, prev[prev.length - 1] ?? DEFAULT_TAX_RATE])
    setDaPct(prev => [...prev, prev[prev.length - 1] ?? DEFAULT_DA_PCT])
    setNwcPct(prev => [...prev, prev[prev.length - 1] ?? DEFAULT_NWC_PCT])
    setCapexPct(prev => [...prev, prev[prev.length - 1] ?? DEFAULT_CAPEX_PCT])
  }

  const removeProjectionYear = (index: number) => {
    if (revGrowth.length <= MIN_PROJECTION_YEARS || index < MIN_PROJECTION_YEARS) return
    setRevGrowth(prev => prev.filter((_, i) => i !== index))
    setEbitdaMargin(prev => prev.filter((_, i) => i !== index))
    setTaxRate(prev => prev.filter((_, i) => i !== index))
    setDaPct(prev => prev.filter((_, i) => i !== index))
    setNwcPct(prev => prev.filter((_, i) => i !== index))
    setCapexPct(prev => prev.filter((_, i) => i !== index))
  }

  const assumptionValidation = useMemo(() => {
    const errors: string[] = []
    const years = revGrowth.length
    const expectedLengths: Array<[string, number]> = [
      ["EBITDA Margin", ebitdaMargin.length],
      ["Tax Rate", taxRate.length],
      ["D&A", daPct.length],
      ["NWC", nwcPct.length],
      ["CapEx", capexPct.length],
    ]

    if (years < MIN_PROJECTION_YEARS || years > MAX_PROJECTION_YEARS) {
      errors.push(`Projection must use ${MIN_PROJECTION_YEARS}-${MAX_PROJECTION_YEARS} years.`)
    }
    expectedLengths.forEach(([label, length]) => {
      if (length !== years) errors.push(`${label} must have ${years} yearly values.`)
    })

    const parseSeries = (
      label: string,
      values: string[],
      min: number | null,
      max: number | null,
      minInclusive = true,
      maxInclusive = true,
    ) => values.map((raw, i) => {
      const itemLabel = `${label} Year ${i + 1}`
      const parsed = parseNumberInput(raw)
      if (parsed === null) {
        errors.push(`${itemLabel} must be a number.`)
        return 0
      }
      const boundError = pctBoundError(itemLabel, parsed, min, max, minInclusive, maxInclusive)
      if (boundError) errors.push(boundError)
      return parsed / 100
    })

    const revenueGrowthRates = parseSeries("Revenue Growth", revGrowth, -100, null, false)
    const ebitdaMargins = parseSeries("EBITDA Margin", ebitdaMargin, 0, 100, false, false)
    const taxRates = parseSeries("Tax Rate", taxRate, 0, 100, true, false)
    const daPcts = parseSeries("D&A", daPct, 0, 100, true, false)
    const nwcPcts = parseSeries("NWC", nwcPct, -100, 100)
    const capexPcts = parseSeries("CapEx", capexPct, 0, 100, true, false)

    const parsedWacc = parseNumberInput(wacc)
    if (parsedWacc === null) errors.push("WACC must be a number.")
    const waccFraction = (parsedWacc ?? 0) / 100
    const waccBoundError = parsedWacc === null
      ? null
      : pctBoundError("WACC", parsedWacc, 0, 100, false, false)
    if (waccBoundError) errors.push(waccBoundError)

    const terminalGrowthRates = {
      bear: parseNumberInput(tgrBear),
      base: parseNumberInput(tgrBase),
      bull: parseNumberInput(tgrBull),
    }
    Object.entries(terminalGrowthRates).forEach(([scenario, value]) => {
      const label = `${scenario.charAt(0).toUpperCase() + scenario.slice(1)} terminal growth`
      if (value === null) {
        errors.push(`${label} must be a number.`)
      } else if (parsedWacc !== null && waccFraction <= value / 100) {
        errors.push(`${label} must be below WACC.`)
      }
    })

    const parseMultiple = (label: string, raw: string) => {
      const parsed = parseNumberInput(raw)
      if (parsed === null) {
        errors.push(`${label} must be a number.`)
        return 0
      }
      if (parsed <= 0) errors.push(`${label} must be greater than 0x.`)
      return parsed
    }

    const exitEvEbitda = {
      bear: parseMultiple("Bear EV/EBITDA exit", eveBear),
      base: parseMultiple("Base EV/EBITDA exit", eveBase),
      bull: parseMultiple("Bull EV/EBITDA exit", eveBull),
    }
    const exitEvRevenue = {
      bear: parseMultiple("Bear EV/Revenue exit", evrBear),
      base: parseMultiple("Base EV/Revenue exit", evrBase),
      bull: parseMultiple("Bull EV/Revenue exit", evrBull),
    }

    const body: DCFValuationRequest | null = submittedTicker && errors.length === 0
      ? {
          ticker: submittedTicker,
          revenue_growth_rates: revenueGrowthRates,
          ebitda_margin: ebitdaMargins,
          tax_rate: taxRates,
          da_pct_revenue: daPcts,
          nwc_pct_revenue: nwcPcts,
          capex_pct_revenue: capexPcts,
          wacc: waccFraction,
          terminal_growth_rates: {
            bear: (terminalGrowthRates.bear ?? 0) / 100,
            base: (terminalGrowthRates.base ?? 0) / 100,
            bull: (terminalGrowthRates.bull ?? 0) / 100,
          },
          exit_ev_ebitda: exitEvEbitda,
          exit_ev_revenue: exitEvRevenue,
        }
      : null

    return { errors, body }
  }, [
    revGrowth,
    ebitdaMargin,
    taxRate,
    daPct,
    nwcPct,
    capexPct,
    wacc,
    tgrBear,
    tgrBase,
    tgrBull,
    eveBear,
    eveBase,
    eveBull,
    evrBear,
    evrBase,
    evrBull,
    submittedTicker,
  ])

  const handleAnalyze = () => {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    if (submittedTicker !== t) {
      dcfMutation.reset()
      resetProjectionAssumptions()
      setLastSuccessfulDCFRequest(null)
      setDownloadError(null)
    }
    setSubmittedTicker(t)
  }

  const handleRunDCF = () => {
    if (!assumptionValidation.body) return
    dcfMutation.mutate(assumptionValidation.body)
  }

  // ---------------------------------------------------------------------------
  // Build historical tables
  // ---------------------------------------------------------------------------

  function buildHistoricalRows(
    data: any[],
    periodKey: string,
    metricKey: string,
    pctKey: string,
    metricLabel: string,
    pctLabel: string,
    metricFmt: (v: any) => string,
  ): { columns: string[]; rows: TransposedRow[] } {
    if (!data?.length) return { columns: [], rows: [] }
    const columns = data.map((d: any) => d[periodKey])
    const avg = data[0]?.avg

    return {
      columns,
      rows: [
        {
          label: "Revenue",
          values: [...data.map((d: any) => fmtB(d.revenue)), ""],
        },
        {
          label: metricLabel,
          values: [...data.map((d: any) => metricFmt(d[metricKey])), ""],
          isBold: true,
        },
        {
          label: pctLabel,
          values: [
            ...data.map((d: any) => fmtPct(d[pctKey])),
            avg != null ? fmtPct(avg) : "",
          ],
          isHighlight: true,
        },
      ],
    }
  }

  function buildMultipleRows(
    data: any[],
    multipleKey: string,
  ): { columns: string[]; rows: TransposedRow[] } {
    if (!data?.length) return { columns: [], rows: [] }
    const columns = data.map((d: any) => d.quarter_end)
    const avg = data[0]?.avg
    return {
      columns,
      rows: [
        {
          label: "Enterprise Value",
          values: [...data.map((d: any) => fmtB(d.ev)), ""],
        },
        {
          label: "Multiple",
          values: [
            ...data.map((d: any) => fmtX(d[multipleKey])),
            avg != null ? fmtX(avg) : "",
          ],
          isBold: true,
          isHighlight: true,
        },
      ],
    }
  }

  // ---------------------------------------------------------------------------
  // Build projection table
  // ---------------------------------------------------------------------------

  function buildProjectionTable(projection: any[]): {
    columns: string[]
    rows: TransposedRow[]
  } {
    if (!projection?.length) return { columns: [], rows: [] }
    const columns = projection.map((p: any) => p.year_label)
    const metrics: { label: string; key: string; fmt: (v: any) => string; bold?: boolean; highlight?: boolean }[] = [
      { label: "Revenue", key: "revenue", fmt: fmtB, bold: true },
      { label: "Revenue Growth", key: "revenue_growth", fmt: fmtPct },
      { label: "", key: "_spacer1", fmt: () => "" },
      { label: "EBITDA", key: "ebitda", fmt: fmtB, bold: true },
      { label: "EBITDA Margin", key: "ebitda_margin", fmt: fmtPct },
      { label: "", key: "_spacer2", fmt: () => "" },
      { label: "Operating Income (EBIT)", key: "ebit", fmt: fmtB },
      { label: "Tax Rate", key: "tax_rate", fmt: fmtPct },
      { label: "NOPAT", key: "nopat", fmt: fmtB },
      { label: "", key: "_spacer3", fmt: () => "" },
      { label: "D&A", key: "da", fmt: fmtB },
      { label: "NWC", key: "nwc", fmt: fmtB },
      { label: "\u0394 NWC", key: "delta_nwc", fmt: fmtB },
      { label: "CapEx", key: "capex", fmt: fmtB },
      { label: "", key: "_spacer4", fmt: () => "" },
      { label: "UFCF", key: "ufcf", fmt: fmtB, bold: true, highlight: true },
      { label: "", key: "_spacer5", fmt: () => "" },
      { label: "Discount Rate", key: "discount_rate", fmt: fmtPct },
      { label: "PV of UFCFs", key: "pv_ufcf", fmt: fmtB, highlight: true },
    ]
    return {
      columns,
      rows: metrics.map(m => ({
        label: m.label,
        values: m.key.startsWith("_spacer")
          ? columns.map(() => "")
          : projection.map((p: any) => m.fmt(p[m.key])),
        isBold: m.bold,
        isHighlight: m.highlight,
      })),
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const ebitdaTable = hist ? buildHistoricalRows(
    hist.ebitda, "fiscal_year", "ebitda", "ebitda_margin",
    "EBITDA", "EBITDA Margin (%)", fmtB,
  ) : null

  const daTable = hist ? buildHistoricalRows(
    hist.depreciation, "fiscal_year", "da", "da_pct_rev",
    "D&A", "% of Revenue", fmtB,
  ) : null

  const capexTable = hist ? buildHistoricalRows(
    hist.capex, "fiscal_year", "capex", "capex_pct_rev",
    "Capital Expenditures", "% of Revenue", fmtB,
  ) : null

  const nwcTable = hist ? buildHistoricalRows(
    hist.nwc, "fiscal_year", "nwc", "nwc_pct_rev",
    "Net Working Capital", "% of Revenue", fmtB,
  ) : null

  const evEbitdaTable = hist ? buildMultipleRows(hist.ev_ebitda, "ev_ebitda") : null
  const evRevTable = hist ? buildMultipleRows(hist.rev_multiple, "ev_revenue") : null

  const activeDCFData = dcfMutation.data?.ticker === submittedTicker ? dcfMutation.data : null
  const projTable = activeDCFData ? buildProjectionTable(activeDCFData.projection) : null
  const valuations = activeDCFData?.valuations
  const currentPrice = activeDCFData?.current_price ?? hist?.current_price ?? 0
  const yearlyAssumptionRows = [
    { label: "Revenue Growth", values: revGrowth, setter: setRevGrowth, step: "0.5" },
    { label: "EBITDA Margin", values: ebitdaMargin, setter: setEbitdaMargin, step: "0.1" },
    { label: "Tax Rate", values: taxRate, setter: setTaxRate, step: "0.1" },
    { label: "D&A", values: daPct, setter: setDaPct, step: "0.1" },
    { label: "NWC", values: nwcPct, setter: setNwcPct, step: "0.1" },
    { label: "CapEx", values: capexPct, setter: setCapexPct, step: "0.1" },
  ]
  const currentDCFRequestKey = assumptionValidation.body ? JSON.stringify(assumptionValidation.body) : null
  const lastSuccessfulDCFRequestKey = lastSuccessfulDCFRequest ? JSON.stringify(lastSuccessfulDCFRequest) : null
  const canDownloadDCF = Boolean(
    activeDCFData &&
    lastSuccessfulDCFRequest &&
    currentDCFRequestKey &&
    currentDCFRequestKey === lastSuccessfulDCFRequestKey,
  )

  const handleDownloadDCF = async () => {
    if (!lastSuccessfulDCFRequest || !canDownloadDCF || isDownloadingDCF) return

    setIsDownloadingDCF(true)
    setDownloadError(null)
    try {
      const blob = await downloadDCFModel(lastSuccessfulDCFRequest)
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = dcfFileNameForTicker(lastSuccessfulDCFRequest.ticker)
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.URL.revokeObjectURL(url)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : "Failed to download DCF model")
    } finally {
      setIsDownloadingDCF(false)
    }
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-app">DCF Model</h1>
        <p className="mt-1 text-sm text-muted">
          Discounted Cash Flow valuation with multiple exit methods
        </p>
      </div>

      {/* Ticker input */}
      <form onSubmit={e => { e.preventDefault(); handleAnalyze() }} className="flex items-end gap-3">
        <div className="w-48">
          <TextInput
            label="Ticker"
            value={ticker}
            onChange={setTicker}
            placeholder="e.g. AAPL"
            uppercase
          />
        </div>
        <ActionButton type="submit" loading={isFetching} disabled={!ticker.trim()}>
          Analyze
        </ActionButton>
      </form>

      {isError && (
        <ErrorMessage message={(error as any)?.message ?? "Failed to fetch data"} />
      )}

      {isFetching && <LoadingSpinner message="Fetching historical data..." />}

      {hist && !isFetching && (
        <>
          {/* Company info */}
          <div className="flex items-baseline gap-3">
            <h2 className="text-lg font-semibold text-app">{hist.company_name}</h2>
            <span className="text-sm text-muted">{hist.ticker}</span>
            {hist.current_price && (
              <span className="text-sm font-medium text-app">
                {fmtPrice(hist.current_price)}
              </span>
            )}
            <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-blue-50 text-blue-700 border border-blue-200">
              yFinance
            </span>
            {hist.data_source === "edgar" && (
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-green-50 text-green-700 border border-green-200">
                + SEC EDGAR (multiples)
              </span>
            )}
          </div>

          {/* Tab switcher */}
          <SegmentedControl
            options={[
              { value: "historical" as TabMode, label: "Historical" },
              { value: "dcf" as TabMode, label: "DCF Model" },
            ]}
            value={tab}
            onChange={setTab}
          />

          {/* ── Historical Tab ────────────────────────────────────── */}
          {tab === "historical" && (
            <div className="space-y-6">
              {/* WACC summary cards */}
              {hist.wacc_inputs && (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
                  <MetricCard title="Beta" value={hist.wacc_inputs.beta} />
                  <MetricCard
                    title="Risk-Free Rate"
                    value={`${hist.wacc_inputs.risk_free_rate}%`}
                  />
                  <MetricCard
                    title="Cost of Equity"
                    value={`${hist.wacc_inputs.cost_of_equity}%`}
                  />
                  <MetricCard
                    title="Cost of Debt"
                    value={`${hist.wacc_inputs.cost_of_debt}%`}
                    subtitle={hist.wacc_inputs.debt_warning ? "Default (5%)" : undefined}
                  />
                  <MetricCard
                    title="WACC"
                    value={`${hist.wacc_inputs.wacc}%`}
                    signal="info"
                    signalLabel={`E: ${hist.wacc_inputs.equity_weight}% / D: ${hist.wacc_inputs.debt_weight}%`}
                  />
                </div>
              )}

              {/* EBITDA */}
              {ebitdaTable && ebitdaTable.columns.length > 0 && (
                <TransposedTable
                  title="EBITDA"
                  columns={ebitdaTable.columns}
                  rows={ebitdaTable.rows}
                  avgLabel="5-Year Average"
                />
              )}

              {/* D&A */}
              {daTable && daTable.columns.length > 0 && (
                <TransposedTable
                  title="Depreciation & Amortization"
                  columns={daTable.columns}
                  rows={daTable.rows}
                  avgLabel="5-Year Average"
                />
              )}

              {/* CapEx */}
              {capexTable && capexTable.columns.length > 0 && (
                <TransposedTable
                  title="Capital Expenditures"
                  columns={capexTable.columns}
                  rows={capexTable.rows}
                  avgLabel="5-Year Average"
                />
              )}

              {/* NWC */}
              {nwcTable && nwcTable.columns.length > 0 && (
                <TransposedTable
                  title="Net Working Capital"
                  columns={nwcTable.columns}
                  rows={nwcTable.rows}
                  avgLabel="5-Year Average"
                />
              )}

              {/* EV/EBITDA multiples */}
              {evEbitdaTable && evEbitdaTable.columns.length > 0 && (
                <TransposedTable
                  title="EV / EBITDA Multiple"
                  columns={evEbitdaTable.columns}
                  rows={evEbitdaTable.rows}
                  avgLabel="Average"
                />
              )}

              {/* Revenue multiples */}
              {evRevTable && evRevTable.columns.length > 0 && (
                <TransposedTable
                  title="EV / Revenue Multiple"
                  columns={evRevTable.columns}
                  rows={evRevTable.rows}
                  avgLabel="Average"
                />
              )}
            </div>
          )}

          {/* ── DCF Model Tab ─────────────────────────────────────── */}
          {tab === "dcf" && (
            <div className="space-y-6">
              {/* Assumptions panel */}
              <div className="rounded-xl border border-border bg-surface shadow-sm p-5 space-y-5">
                <h3 className="text-sm font-semibold text-app">Assumptions</h3>

                {/* Yearly operating assumptions */}
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs font-medium text-muted">Operating Assumptions (% per year)</p>
                    <button
                      type="button"
                      onClick={addProjectionYear}
                      disabled={revGrowth.length >= MAX_PROJECTION_YEARS}
                      className="theme-button-base theme-button-secondary inline-flex min-h-8 items-center gap-2 px-3 text-xs disabled:pointer-events-none disabled:opacity-50"
                    >
                      <Plus size={14} aria-hidden="true" />
                      Add Year
                    </button>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-border">
                    <table className="w-full min-w-[720px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted/30">
                          <th className="px-3 py-2 text-left text-xs font-semibold text-muted min-w-[150px]" />
                          {revGrowth.map((_, i) => (
                            <th key={i} className="px-3 py-2 text-center text-xs font-semibold text-muted min-w-[86px]">
                              <div className="flex items-center justify-center gap-1">
                                <span>Year {i + 1}</span>
                                {i >= MIN_PROJECTION_YEARS && (
                                  <button
                                    type="button"
                                    onClick={() => removeProjectionYear(i)}
                                    title={`Remove Year ${i + 1}`}
                                    aria-label={`Remove Year ${i + 1}`}
                                    className="theme-button-ghost inline-flex h-6 w-6 items-center justify-center rounded-md text-muted hover:text-app"
                                  >
                                    <Trash2 size={13} aria-hidden="true" />
                                  </button>
                                )}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {yearlyAssumptionRows.map(row => (
                          <tr key={row.label} className="border-b border-border/50 last:border-b-0">
                            <td className="px-3 py-2 text-sm font-medium text-muted whitespace-nowrap">
                              {row.label}
                            </td>
                            {row.values.map((value, i) => (
                              <td key={i} className="px-3 py-2">
                                <div className="flex items-center justify-center gap-1">
                                  <input
                                    type="number"
                                    step={row.step}
                                    value={value}
                                    onChange={e => updateYearlyAssumption(row.setter, i, e.target.value)}
                                    className="theme-input w-16 rounded-lg px-2 py-1.5 text-right text-sm"
                                  />
                                  <span className="text-xs text-muted">%</span>
                                </div>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Discount rate */}
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <InputRow label="WACC" value={wacc} onChange={setWacc} />
                </div>

                {/* Scenario inputs */}
                <div className="space-y-3 pt-2 border-t border-border">
                  <p className="text-xs font-medium text-muted">Exit Assumptions</p>
                  <ScenarioInputRow
                    label="Terminal Growth"
                    bear={tgrBear} base={tgrBase} bull={tgrBull}
                    onBear={setTgrBear} onBase={setTgrBase} onBull={setTgrBull}
                    suffix="%"
                  />
                  <ScenarioInputRow
                    label="EV/EBITDA Exit"
                    bear={eveBear} base={eveBase} bull={eveBull}
                    onBear={setEveBear} onBase={setEveBase} onBull={setEveBull}
                    suffix="x"
                  />
                  <ScenarioInputRow
                    label="EV/Revenue Exit"
                    bear={evrBear} base={evrBase} bull={evrBull}
                    onBear={setEvrBear} onBase={setEvrBase} onBull={setEvrBull}
                    suffix="x"
                  />
                </div>

                {submittedTicker && assumptionValidation.errors.length > 0 && (
                  <Notice tone="warning">
                    <p className="text-sm font-medium">Fix assumptions before running DCF.</p>
                    <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs">
                      {assumptionValidation.errors.slice(0, 5).map(errorText => (
                        <li key={errorText}>{errorText}</li>
                      ))}
                      {assumptionValidation.errors.length > 5 && (
                        <li>{assumptionValidation.errors.length - 5} more validation errors</li>
                      )}
                    </ul>
                  </Notice>
                )}

                <div className="flex flex-wrap gap-3">
                  <ActionButton
                    onClick={handleRunDCF}
                    loading={dcfMutation.isPending}
                    disabled={!submittedTicker || !assumptionValidation.body}
                    className="w-auto px-6"
                  >
                    Run DCF
                  </ActionButton>
                  <button
                    type="button"
                    onClick={handleDownloadDCF}
                    disabled={!canDownloadDCF || isDownloadingDCF}
                    title={canDownloadDCF ? "Download DCF model as Excel" : "Run DCF before downloading the latest assumptions"}
                    className="theme-button-base theme-button-secondary inline-flex min-h-10 items-center gap-2 px-4 disabled:pointer-events-none disabled:opacity-50"
                  >
                    <Download size={16} aria-hidden="true" />
                    {isDownloadingDCF ? "Downloading..." : "Download Excel"}
                  </button>
                </div>
              </div>

              {dcfMutation.isError && (
                <ErrorMessage
                  message={(dcfMutation.error as any)?.message ?? "DCF calculation failed"}
                />
              )}

              {downloadError && <ErrorMessage message={downloadError} />}

              {dcfMutation.isPending && <LoadingSpinner message="Running DCF valuation..." />}

              {/* Projection table */}
              {projTable && projTable.columns.length > 0 && (
                <>
                  <TransposedTable
                    title={`${projTable.columns.length}-Year Projection`}
                    columns={projTable.columns}
                    rows={projTable.rows}
                  />

                  {/* PV of FCFs summary */}
                  {activeDCFData && (
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <MetricCard
                        title="PV of UFCFs"
                        value={fmtB(activeDCFData.pv_fcfs)}
                      />
                      <MetricCard
                        title="Net Debt"
                        value={fmtB(activeDCFData.net_debt)}
                      />
                      <MetricCard
                        title="Shares Outstanding"
                        value={activeDCFData.shares_outstanding
                          ? `${(activeDCFData.shares_outstanding / 1e9).toFixed(2)}B`
                          : "N/A"}
                      />
                      <MetricCard
                        title="Current Price"
                        value={fmtPrice(activeDCFData.current_price)}
                      />
                    </div>
                  )}
                </>
              )}

              {/* Valuation results */}
              {valuations && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-app">Implied Share Price</h3>

                  {/* Gordon Growth */}
                  <div>
                    <p className="text-xs font-medium text-muted mb-2">Gordon Growth Model</p>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      {(["bear", "base", "bull"] as const).map(s => (
                        <ValuationCard
                          key={s}
                          label="Gordon Growth"
                          scenario={s.charAt(0).toUpperCase() + s.slice(1)}
                          data={valuations.gordon_growth?.[s]}
                          currentPrice={currentPrice}
                        />
                      ))}
                    </div>
                  </div>

                  {/* EV/EBITDA exit */}
                  <div>
                    <p className="text-xs font-medium text-muted mb-2">EV/EBITDA Exit Multiple</p>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      {(["bear", "base", "bull"] as const).map(s => (
                        <ValuationCard
                          key={s}
                          label="EV/EBITDA"
                          scenario={s.charAt(0).toUpperCase() + s.slice(1)}
                          data={valuations.ev_ebitda_exit?.[s]}
                          currentPrice={currentPrice}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Revenue exit */}
                  <div>
                    <p className="text-xs font-medium text-muted mb-2">Revenue Exit Multiple</p>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      {(["bear", "base", "bull"] as const).map(s => (
                        <ValuationCard
                          key={s}
                          label="EV/Revenue"
                          scenario={s.charAt(0).toUpperCase() + s.slice(1)}
                          data={valuations.ev_revenue_exit?.[s]}
                          currentPrice={currentPrice}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
