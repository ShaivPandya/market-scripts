import { useState, useEffect } from "react"
import { useQuery, useMutation } from "@tanstack/react-query"

import { fetchDCFHistorical, runDCFValuation, type DCFValuationRequest } from "@/lib/api"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, TextInput, ActionButton } from "@/components/shared/FormControls"

type TabMode = "historical" | "dcf"

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
  const [revGrowth, setRevGrowth] = useState(["10", "8", "7", "6", "5"])
  const [ebitdaMargin, setEbitdaMargin] = useState("25")
  const [taxRate, setTaxRate] = useState("21")
  const [daPct, setDaPct] = useState("3")
  const [nwcPct, setNwcPct] = useState("5")
  const [capexPct, setCapexPct] = useState("3")
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

  // Pre-populate from historical averages
  useEffect(() => {
    if (!hist) return
    const avg = hist.historical_averages
    if (avg) {
      if (avg.ebitda_margin_avg != null) setEbitdaMargin(String(avg.ebitda_margin_avg))
      if (avg.da_pct_avg != null) setDaPct(String(avg.da_pct_avg))
      if (avg.nwc_pct_avg != null) setNwcPct(String(avg.nwc_pct_avg))
      if (avg.capex_pct_avg != null) setCapexPct(String(avg.capex_pct_avg))
    }
    const w = hist.wacc_inputs
    if (w?.wacc != null) setWacc(String(w.wacc))
    if (w?.tax_rate != null) setTaxRate(String(w.tax_rate))

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
  })

  const handleAnalyze = () => {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    setSubmittedTicker(t)
  }

  const handleRunDCF = () => {
    if (!submittedTicker) return
    const body: DCFValuationRequest = {
      ticker: submittedTicker,
      revenue_growth_rates: revGrowth.map(v => parseFloat(v) / 100),
      ebitda_margin: parseFloat(ebitdaMargin) / 100,
      tax_rate: parseFloat(taxRate) / 100,
      da_pct_revenue: parseFloat(daPct) / 100,
      nwc_pct_revenue: parseFloat(nwcPct) / 100,
      capex_pct_revenue: parseFloat(capexPct) / 100,
      wacc: parseFloat(wacc) / 100,
      terminal_growth_rates: {
        bear: parseFloat(tgrBear) / 100,
        base: parseFloat(tgrBase) / 100,
        bull: parseFloat(tgrBull) / 100,
      },
      exit_ev_ebitda: {
        bear: parseFloat(eveBear),
        base: parseFloat(eveBase),
        bull: parseFloat(eveBull),
      },
      exit_ev_revenue: {
        bear: parseFloat(evrBear),
        base: parseFloat(evrBase),
        bull: parseFloat(evrBull),
      },
    }
    dcfMutation.mutate(body)
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
    hist.nwc, "quarter_end", "nwc", "nwc_pct_rev",
    "Net Working Capital", "% of Revenue", fmtB,
  ) : null

  const evEbitdaTable = hist ? buildMultipleRows(hist.ev_ebitda, "ev_ebitda") : null
  const evRevTable = hist ? buildMultipleRows(hist.rev_multiple, "ev_revenue") : null

  const projTable = dcfMutation.data ? buildProjectionTable(dcfMutation.data.projection) : null
  const valuations = dcfMutation.data?.valuations
  const currentPrice = dcfMutation.data?.current_price ?? hist?.current_price ?? 0

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
      <div className="flex items-end gap-3">
        <div className="w-48">
          <TextInput
            label="Ticker"
            value={ticker}
            onChange={setTicker}
            placeholder="e.g. AAPL"
            onKeyDown={e => {
              if (e.key === "Enter") handleAnalyze()
            }}
          />
        </div>
        <ActionButton onClick={handleAnalyze} loading={isFetching} disabled={!ticker.trim()}>
          Analyze
        </ActionButton>
      </div>

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
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              hist.data_source === "edgar"
                ? "bg-green-50 text-green-700 border border-green-200"
                : "bg-blue-50 text-blue-700 border border-blue-200"
            }`}>
              {hist.data_source === "edgar" ? "SEC EDGAR" : "yFinance"}
            </span>
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
                  title="Net Working Capital (Quarterly)"
                  columns={nwcTable.columns}
                  rows={nwcTable.rows}
                  avgLabel="5-Quarter Average"
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

                {/* Revenue growth rates */}
                <div>
                  <p className="text-xs font-medium text-muted mb-2">Revenue Growth (% per year)</p>
                  <div className="flex items-center gap-3">
                    {revGrowth.map((v, i) => (
                      <div key={i} className="flex flex-col items-center gap-1">
                        <span className="text-xs text-muted">Yr {i + 1}</span>
                        <input
                          type="number"
                          step="0.5"
                          value={v}
                          onChange={e => {
                            const next = [...revGrowth]
                            next[i] = e.target.value
                            setRevGrowth(next)
                          }}
                          className="theme-input rounded-lg px-2 py-1.5 text-sm text-right w-16"
                        />
                        <span className="text-xs text-muted">%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Margin / cost assumptions */}
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <InputRow label="EBITDA Margin" value={ebitdaMargin} onChange={setEbitdaMargin} />
                  <InputRow label="Tax Rate" value={taxRate} onChange={setTaxRate} />
                  <InputRow label="D&A (% Rev)" value={daPct} onChange={setDaPct} />
                  <InputRow label="NWC (% Rev)" value={nwcPct} onChange={setNwcPct} />
                  <InputRow label="CapEx (% Rev)" value={capexPct} onChange={setCapexPct} />
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

                <ActionButton
                  onClick={handleRunDCF}
                  loading={dcfMutation.isPending}
                  disabled={!submittedTicker}
                >
                  Run DCF
                </ActionButton>
              </div>

              {dcfMutation.isError && (
                <ErrorMessage
                  message={(dcfMutation.error as any)?.message ?? "DCF calculation failed"}
                />
              )}

              {dcfMutation.isPending && <LoadingSpinner message="Running DCF valuation..." />}

              {/* Projection table */}
              {projTable && projTable.columns.length > 0 && (
                <>
                  <TransposedTable
                    title="5-Year Projection"
                    columns={projTable.columns}
                    rows={projTable.rows}
                  />

                  {/* PV of FCFs summary */}
                  {dcfMutation.data && (
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <MetricCard
                        title="PV of UFCFs"
                        value={fmtB(dcfMutation.data.pv_fcfs)}
                      />
                      <MetricCard
                        title="Net Debt"
                        value={fmtB(dcfMutation.data.net_debt)}
                      />
                      <MetricCard
                        title="Shares Outstanding"
                        value={dcfMutation.data.shares_outstanding
                          ? `${(dcfMutation.data.shares_outstanding / 1e9).toFixed(2)}B`
                          : "N/A"}
                      />
                      <MetricCard
                        title="Current Price"
                        value={fmtPrice(dcfMutation.data.current_price)}
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
