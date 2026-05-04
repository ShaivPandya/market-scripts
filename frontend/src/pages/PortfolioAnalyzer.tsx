import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SegmentedControl, SliderInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative } from "@/lib/colors"
import { runPortfolioAnalyzerAsync, type AnalyzerScenarioRequest } from "@/lib/api"

interface AnalyzerResponse {
  weights_df?: Record<string, unknown>[]
  [key: string]: unknown
}

type ScenarioPreset = "balanced" | "quality" | "momentum" | "defensive" | "value" | "custom"

interface AnalyzerScenarioState {
  preset: ScenarioPreset
  factor_weights: {
    quality: number
    price_momentum: number
    fundamental_momentum: number
    valuation: number
  }
  fundamental_momentum_weights: {
    revenue: number
    eps: number
  }
  valuation_weights: {
    price_sales: number
    price_operating_income: number
    price_fcf: number
    price_earnings: number
  }
  brakes: {
    drawdown_sensitivity: number
    contrarian_penalty: number
    short_squeeze_brake: number
  }
}

const ANALYZER_STATE_KEY = ["portfolio-analyzer", "state"] as const

const SCENARIO_PRESETS: Record<Exclude<ScenarioPreset, "custom">, AnalyzerScenarioState> = {
  balanced: {
    preset: "balanced",
    factor_weights: { quality: 0.30, price_momentum: 0.40, fundamental_momentum: 0.30, valuation: 0.0 },
    fundamental_momentum_weights: { revenue: 0.67, eps: 0.33 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.25, price_earnings: 0.25 },
    brakes: { drawdown_sensitivity: 0.0, contrarian_penalty: 0.0, short_squeeze_brake: 0.0 },
  },
  quality: {
    preset: "quality",
    factor_weights: { quality: 0.45, price_momentum: 0.20, fundamental_momentum: 0.25, valuation: 0.10 },
    fundamental_momentum_weights: { revenue: 0.60, eps: 0.40 },
    valuation_weights: { price_sales: 0.20, price_operating_income: 0.30, price_fcf: 0.30, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.15, contrarian_penalty: 0.15, short_squeeze_brake: 0.15 },
  },
  momentum: {
    preset: "momentum",
    factor_weights: { quality: 0.15, price_momentum: 0.45, fundamental_momentum: 0.35, valuation: 0.05 },
    fundamental_momentum_weights: { revenue: 0.55, eps: 0.45 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.25, price_earnings: 0.25 },
    brakes: { drawdown_sensitivity: 0.10, contrarian_penalty: 0.10, short_squeeze_brake: 0.20 },
  },
  defensive: {
    preset: "defensive",
    factor_weights: { quality: 0.35, price_momentum: 0.20, fundamental_momentum: 0.20, valuation: 0.25 },
    fundamental_momentum_weights: { revenue: 0.50, eps: 0.50 },
    valuation_weights: { price_sales: 0.15, price_operating_income: 0.30, price_fcf: 0.35, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.55, contrarian_penalty: 0.45, short_squeeze_brake: 0.55 },
  },
  value: {
    preset: "value",
    factor_weights: { quality: 0.20, price_momentum: 0.15, fundamental_momentum: 0.25, valuation: 0.40 },
    fundamental_momentum_weights: { revenue: 0.55, eps: 0.45 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.30, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.20, contrarian_penalty: 0.20, short_squeeze_brake: 0.25 },
  },
}

const PRESET_OPTIONS: { value: ScenarioPreset; label: string }[] = [
  { value: "balanced", label: "Balanced" },
  { value: "quality", label: "Quality" },
  { value: "momentum", label: "Momentum" },
  { value: "defensive", label: "Defensive" },
  { value: "value", label: "Value" },
  { value: "custom", label: "Custom" },
]

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const percentFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const COLUMN_LABELS: Record<string, string> = {
  ticker: "Ticker",
  asset: "Asset",
  direction: "Direction",
  contrarian: "Contrarian",
  drawdown_52w: "Drawdown 52W",
  stabilized_10d: "Stabilized",
  days_since_new_low: "Days Since New Low",
  no_new_high_20d: "No New High 20D",
  days_since_high: "Days Since High",
  avg20_roc63: "Avg20 ROC(63)",
  avg10_rel_roc: "Avg10 Rel ROC",
  signal: "Signal",
  baseline_score: "Baseline",
  scenario_score: "Scenario",
  score_delta: "Delta",
  scenario_driver: "Driver",
  scenario_penalty: "Penalty",
  quality_signal: "Quality",
  eps_mom_signal: "EPS Momentum",
  rev_mom_signal: "Revenue Momentum",
  price_mom_signal: "Price Momentum",
  fundamental_momentum_signal: "Fundamental Momentum",
  valuation_signal: "Valuation",
  price_sales: "P/S",
  price_operating_income: "P/OI",
  price_fcf: "P/FCF",
  price_earnings: "P/E",
}

const COLUMN_ORDER = [
  "ticker",
  "asset",
  "direction",
  "scenario_score",
  "score_delta",
  "scenario_driver",
  "scenario_penalty",
  "baseline_score",
  "signal",
  "valuation_signal",
  "fundamental_momentum_signal",
  "quality_signal",
  "price_mom_signal",
  "rev_mom_signal",
  "eps_mom_signal",
  "price_sales",
  "price_operating_income",
  "price_fcf",
  "price_earnings",
  "contrarian",
  "drawdown_52w",
  "stabilized_10d",
  "days_since_new_low",
  "no_new_high_20d",
  "days_since_high",
  "avg20_roc63",
  "avg10_rel_roc",
]

function cloneScenario<T extends AnalyzerScenarioState>(scenario: T): T {
  return {
    preset: scenario.preset,
    factor_weights: { ...scenario.factor_weights },
    fundamental_momentum_weights: { ...scenario.fundamental_momentum_weights },
    valuation_weights: { ...scenario.valuation_weights },
    brakes: { ...scenario.brakes },
  } as T
}

function toRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value.filter((row): row is Record<string, unknown> => row != null && typeof row === "object")
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function toBoolean(value: unknown): boolean | null {
  if (typeof value === "boolean") return value
  if (typeof value === "number") {
    if (value === 1) return true
    if (value === 0) return false
    return null
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (["1", "true", "t", "yes", "y"].includes(normalized)) return true
    if (["0", "false", "f", "no", "n"].includes(normalized)) return false
  }
  return null
}

function isScoreColumn(key: string) {
  return key === "signal" || key.endsWith("_signal") || key.endsWith("_score") || key === "score_delta"
}

function isMultipleColumn(key: string) {
  return key === "price_sales" || key === "price_operating_income" || key === "price_fcf" || key === "price_earnings"
}

function formatScore(value: unknown) {
  const num = toNumber(value)
  if (num == null) return "N/A"
  return `${num >= 0 ? "+" : ""}${numberFormatter.format(num)}`
}

function buildColumns(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  const available = new Set(Object.keys(rows[0]))

  return COLUMN_ORDER
    .filter(key => available.has(key))
    .map(key => ({
      key,
      header: COLUMN_LABELS[key] ?? key,
      colorFn: isScoreColumn(key) ? colorPositiveNegative : undefined,
      format: (value: unknown) => {
        if (key === "contrarian" || key === "stabilized_10d" || key === "no_new_high_20d") {
          const parsed = toBoolean(value)
          return parsed == null ? "N/A" : parsed ? "Yes" : "No"
        }

        if (key === "drawdown_52w") {
          const num = toNumber(value)
          return num == null ? "N/A" : `${percentFormatter.format(num * 100)}%`
        }

        if (key === "days_since_new_low" || key === "days_since_high") {
          const num = toNumber(value)
          return num == null ? "N/A" : String(Math.round(num))
        }

        if (isMultipleColumn(key)) {
          const num = toNumber(value)
          return num == null ? "N/A" : `${num.toFixed(1)}x`
        }

        if (key === "scenario_penalty") {
          const num = toNumber(value)
          return num == null ? "N/A" : numberFormatter.format(num)
        }

        if (isScoreColumn(key)) return formatScore(value)

        return String(value ?? "N/A")
      },
    }))
}

function toScenarioRequest(scenario: AnalyzerScenarioState): AnalyzerScenarioRequest {
  return {
    preset: scenario.preset,
    factor_weights: { ...scenario.factor_weights },
    fundamental_momentum_weights: { ...scenario.fundamental_momentum_weights },
    valuation_weights: { ...scenario.valuation_weights },
    brakes: { ...scenario.brakes },
  }
}

function ImpactList({ title, rows }: { title: string; rows: Record<string, unknown>[] }) {
  return (
    <section className="theme-surface rounded-xl p-4">
      <h2 className="section-title text-sm">{title}</h2>
      <div className="mt-3 space-y-2">
        {rows.length === 0 && <p className="text-sm text-subtle">No scenario deltas yet.</p>}
        {rows.map(row => (
          <div key={`${title}-${String(row.ticker)}`} className="flex items-center gap-3 text-sm">
            <span className="w-16 shrink-0 font-mono font-semibold text-app">{String(row.ticker ?? "")}</span>
            <span className="w-16 shrink-0 font-semibold text-app">{formatScore(row.score_delta)}</span>
            <span className="min-w-0 truncate text-muted">{String(row.scenario_driver ?? "Scenario mix")}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export function PortfolioAnalyzer() {
  const queryClient = useQueryClient()
  const cachedState = queryClient.getQueryData<{
    result: AnalyzerResponse | null
    scenario: AnalyzerScenarioState
  }>(ANALYZER_STATE_KEY)

  const [scenario, setScenario] = useState<AnalyzerScenarioState>(
    cloneScenario(cachedState?.scenario ?? SCENARIO_PRESETS.balanced),
  )
  const [cachedResult, setCachedResult] = useState<AnalyzerResponse | null>(cachedState?.result ?? null)

  const mutation = useMutation({
    mutationFn: (nextScenario: AnalyzerScenarioState) => runPortfolioAnalyzerAsync({ scenario: toScenarioRequest(nextScenario) }),
    onSuccess: result => setCachedResult((result as AnalyzerResponse) ?? null),
  })

  useEffect(() => {
    queryClient.setQueryData(ANALYZER_STATE_KEY, { result: cachedResult, scenario })
  }, [cachedResult, queryClient, scenario])

  const data = (mutation.data as AnalyzerResponse | undefined) ?? cachedResult
  const rows = toRows(data?.weights_df)
  const impactRows = useMemo(
    () => rows.filter(row => toNumber(row.score_delta) != null),
    [rows],
  )
  const upgrades = useMemo(
    () => [...impactRows].sort((a, b) => (toNumber(b.score_delta) ?? 0) - (toNumber(a.score_delta) ?? 0)).slice(0, 5),
    [impactRows],
  )
  const downgrades = useMemo(
    () => [...impactRows].sort((a, b) => (toNumber(a.score_delta) ?? 0) - (toNumber(b.score_delta) ?? 0)).slice(0, 5),
    [impactRows],
  )

  function applyPreset(preset: ScenarioPreset) {
    if (preset === "custom") return
    setScenario(cloneScenario(SCENARIO_PRESETS[preset]))
  }

  function setFactorWeight(key: keyof AnalyzerScenarioState["factor_weights"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", factor_weights: { ...prev.factor_weights, [key]: value } }))
  }

  function setFundamentalWeight(key: keyof AnalyzerScenarioState["fundamental_momentum_weights"], value: number) {
    setScenario(prev => ({
      ...prev,
      preset: "custom",
      fundamental_momentum_weights: { ...prev.fundamental_momentum_weights, [key]: value },
    }))
  }

  function setValuationWeight(key: keyof AnalyzerScenarioState["valuation_weights"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", valuation_weights: { ...prev.valuation_weights, [key]: value } }))
  }

  function setBrake(key: keyof AnalyzerScenarioState["brakes"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", brakes: { ...prev.brakes, [key]: value } }))
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Analyzer</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          Signal and factor diagnostics to guide conviction inputs for Portfolio Sizer.
        </p>
      </div>

      <div className="rounded-xl border border-gray-200/80 bg-white p-5 mb-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="section-title">Scenario Workbench</h2>
            <div className="mt-3">
              <SegmentedControl options={PRESET_OPTIONS} value={scenario.preset} onChange={applyPreset} size="sm" />
            </div>
          </div>
          <ActionButton
            onClick={() => mutation.mutate(scenario)}
            loading={mutation.isPending}
            loadingText="Running scenario..."
          >
            Run Scenario
          </ActionButton>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-6 xl:grid-cols-4">
          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-app">Factor Mix</h3>
            <SliderInput label="Quality" value={scenario.factor_weights.quality} onChange={v => setFactorWeight("quality", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="Price Momentum" value={scenario.factor_weights.price_momentum} onChange={v => setFactorWeight("price_momentum", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="Fundamental Momentum" value={scenario.factor_weights.fundamental_momentum} onChange={v => setFactorWeight("fundamental_momentum", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="Valuation" value={scenario.factor_weights.valuation} onChange={v => setFactorWeight("valuation", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-app">Fundamental Momentum</h3>
            <SliderInput label="Revenue" value={scenario.fundamental_momentum_weights.revenue} onChange={v => setFundamentalWeight("revenue", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="EPS" value={scenario.fundamental_momentum_weights.eps} onChange={v => setFundamentalWeight("eps", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-app">Valuation</h3>
            <SliderInput label="P/S" value={scenario.valuation_weights.price_sales} onChange={v => setValuationWeight("price_sales", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="P/Operating Income" value={scenario.valuation_weights.price_operating_income} onChange={v => setValuationWeight("price_operating_income", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="P/FCF" value={scenario.valuation_weights.price_fcf} onChange={v => setValuationWeight("price_fcf", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="P/E" value={scenario.valuation_weights.price_earnings} onChange={v => setValuationWeight("price_earnings", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-app">Risk Brakes</h3>
            <SliderInput label="Drawdown" value={scenario.brakes.drawdown_sensitivity} onChange={v => setBrake("drawdown_sensitivity", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="Contrarian" value={scenario.brakes.contrarian_penalty} onChange={v => setBrake("contrarian_penalty", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
            <SliderInput label="Short Squeeze" value={scenario.brakes.short_squeeze_brake} onChange={v => setBrake("short_squeeze_brake", v)} min={0} max={1} step={0.05} formatValue={v => `${Math.round(v * 100)}%`} />
          </section>
        </div>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running portfolio analyzer..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && !mutation.isError && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ImpactList title="Top Upgrades" rows={upgrades} />
            <ImpactList title="Top Downgrades" rows={downgrades} />
          </div>
          <DataTable
            label="Scenario Metrics"
            columns={buildColumns(rows)}
            rows={rows}
          />
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Click Run Scenario to load the signal metrics table.</p>
      )}
    </div>
  )
}
