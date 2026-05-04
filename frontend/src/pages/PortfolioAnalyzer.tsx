import { useEffect, useMemo, useState, type ReactNode } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ChevronDown, ChevronRight, Sparkles } from "lucide-react"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SliderInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import { colorPositiveNegative } from "@/lib/colors"
import {
  fetchLLMSettings,
  generatePortfolioAnalyzerBrief,
  runPortfolioAnalyzerAsync,
  type AnalyzerCourseAction,
  type AnalyzerCourseOfAction,
  type AnalyzerFactorBreakdown,
  type AnalyzerScenarioRequest,
  type LLMSettings,
} from "@/lib/api"

interface AnalyzerResponse {
  weights_df?: Record<string, unknown>[]
  course_of_action?: AnalyzerCourseOfAction
  [key: string]: unknown
}

type ScenarioPreset =
  | "balanced"
  | "capital_preservation"
  | "momentum_exploitation"
  | "value_dislocation"
  | "short_defense"
  | "custom"

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
const LLM_SETTINGS_QUERY_KEY = ["llm-settings"] as const

const SCENARIO_PRESETS: Record<Exclude<ScenarioPreset, "custom">, AnalyzerScenarioState> = {
  balanced: {
    preset: "balanced",
    factor_weights: { quality: 0.30, price_momentum: 0.40, fundamental_momentum: 0.30, valuation: 0.0 },
    fundamental_momentum_weights: { revenue: 0.67, eps: 0.33 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.25, price_earnings: 0.25 },
    brakes: { drawdown_sensitivity: 0.0, contrarian_penalty: 0.0, short_squeeze_brake: 0.0 },
  },
  capital_preservation: {
    preset: "capital_preservation",
    factor_weights: { quality: 0.35, price_momentum: 0.15, fundamental_momentum: 0.20, valuation: 0.30 },
    fundamental_momentum_weights: { revenue: 0.50, eps: 0.50 },
    valuation_weights: { price_sales: 0.15, price_operating_income: 0.30, price_fcf: 0.35, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.55, contrarian_penalty: 0.45, short_squeeze_brake: 0.55 },
  },
  momentum_exploitation: {
    preset: "momentum_exploitation",
    factor_weights: { quality: 0.15, price_momentum: 0.50, fundamental_momentum: 0.30, valuation: 0.05 },
    fundamental_momentum_weights: { revenue: 0.55, eps: 0.45 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.25, price_earnings: 0.25 },
    brakes: { drawdown_sensitivity: 0.10, contrarian_penalty: 0.10, short_squeeze_brake: 0.20 },
  },
  value_dislocation: {
    preset: "value_dislocation",
    factor_weights: { quality: 0.20, price_momentum: 0.10, fundamental_momentum: 0.20, valuation: 0.50 },
    fundamental_momentum_weights: { revenue: 0.55, eps: 0.45 },
    valuation_weights: { price_sales: 0.25, price_operating_income: 0.25, price_fcf: 0.30, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.20, contrarian_penalty: 0.20, short_squeeze_brake: 0.25 },
  },
  short_defense: {
    preset: "short_defense",
    factor_weights: { quality: 0.30, price_momentum: 0.35, fundamental_momentum: 0.25, valuation: 0.10 },
    fundamental_momentum_weights: { revenue: 0.55, eps: 0.45 },
    valuation_weights: { price_sales: 0.20, price_operating_income: 0.25, price_fcf: 0.35, price_earnings: 0.20 },
    brakes: { drawdown_sensitivity: 0.25, contrarian_penalty: 0.20, short_squeeze_brake: 0.70 },
  },
}

const MISSION_OPTIONS: { value: Exclude<ScenarioPreset, "custom">; label: string; description: string }[] = [
  { value: "balanced", label: "Balanced", description: "Balanced alpha and risk evidence for current positions." },
  { value: "capital_preservation", label: "Capital Preservation", description: "Favor quality, valuation, and stronger risk brakes." },
  { value: "momentum_exploitation", label: "Momentum Exploitation", description: "Prioritize price and fundamental trend continuation." },
  { value: "value_dislocation", label: "Value Dislocation", description: "Elevate valuation signals while retaining risk checks." },
  { value: "short_defense", label: "Short Defense", description: "Stress-test shorts and squeeze-prone exposures." },
]

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const percentFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
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
    factor_weights: normalizeWeightGroup(scenario.factor_weights),
    fundamental_momentum_weights: normalizeWeightGroup(scenario.fundamental_momentum_weights),
    valuation_weights: normalizeWeightGroup(scenario.valuation_weights),
    brakes: { ...scenario.brakes },
  } as T
}

function normalizeScenarioState(value: AnalyzerScenarioState | undefined): AnalyzerScenarioState {
  if (!value) return cloneScenario(SCENARIO_PRESETS.balanced)
  const rawPreset = value.preset
  const preset: ScenarioPreset = rawPreset === "custom" || rawPreset in SCENARIO_PRESETS ? rawPreset : "balanced"
  const base = preset === "custom" ? SCENARIO_PRESETS.balanced : SCENARIO_PRESETS[preset]
  return {
    preset: value.preset === "custom" ? "custom" : preset as ScenarioPreset,
    factor_weights: normalizeWeightGroup(value.factor_weights ?? base.factor_weights),
    fundamental_momentum_weights: normalizeWeightGroup(value.fundamental_momentum_weights ?? base.fundamental_momentum_weights),
    valuation_weights: normalizeWeightGroup(value.valuation_weights ?? base.valuation_weights),
    brakes: { ...base.brakes, ...(value.brakes ?? {}) },
  }
}

function clampUnit(value: number) {
  if (!Number.isFinite(value)) return 0
  return Math.min(1, Math.max(0, value))
}

function normalizeWeightGroup<T extends Record<string, number>>(weights: T): T {
  const entries = Object.entries(weights) as [keyof T, number][]
  const total = entries.reduce((sum, [, value]) => sum + Math.max(0, Number.isFinite(value) ? value : 0), 0)

  if (total <= 0) {
    const equalWeight = entries.length > 0 ? 1 / entries.length : 0
    return Object.fromEntries(entries.map(([key]) => [key, equalWeight])) as T
  }

  return Object.fromEntries(
    entries.map(([key, value]) => [key, Math.max(0, Number.isFinite(value) ? value : 0) / total]),
  ) as T
}

function rebalanceWeightGroup<T extends Record<string, number>>(weights: T, key: keyof T, value: number): T {
  const nextValue = clampUnit(value)
  const entries = Object.entries(weights) as [keyof T, number][]
  const otherEntries = entries.filter(([entryKey]) => entryKey !== key)
  const remaining = 1 - nextValue
  const otherTotal = otherEntries.reduce(
    (sum, [, entryValue]) => sum + Math.max(0, Number.isFinite(entryValue) ? entryValue : 0),
    0,
  )

  const nextEntries = entries.map(([entryKey, entryValue]) => {
    if (entryKey === key) return [entryKey, nextValue]
    const adjustedValue =
      otherTotal > 0
        ? (Math.max(0, Number.isFinite(entryValue) ? entryValue : 0) / otherTotal) * remaining
        : remaining / Math.max(1, otherEntries.length)
    return [entryKey, adjustedValue]
  })

  return normalizeWeightGroup(Object.fromEntries(nextEntries) as T)
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

function formatPercent(value: unknown) {
  const num = toNumber(value)
  if (num == null) return "N/A"
  return `${percentFormatter.format(num * 100)}%`
}

function buildColumns(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  const available = new Set(Object.keys(rows[0]))

  return COLUMN_ORDER
    .filter(key => available.has(key))
    .map(key => ({
      key,
      header: COLUMN_LABELS[key] ?? key.replace(/_/g, " "),
      colorFn: isScoreColumn(key) ? colorPositiveNegative : undefined,
      format: (value: unknown) => {
        if (value == null || (typeof value === "number" && !Number.isFinite(value))) return "N/A"

        const bool = toBoolean(value)
        if (bool != null && typeof value !== "number" && !isScoreColumn(key)) return bool ? "Yes" : "No"

        if (typeof value === "number" && !isScoreColumn(key)) {
          if (key.startsWith("days_since")) return String(Math.round(value))
          if (key === "drawdown_52w") return `${(value * 100).toFixed(1)}%`
          return numberFormatter.format(value)
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
    factor_weights: normalizeWeightGroup(scenario.factor_weights),
    fundamental_momentum_weights: normalizeWeightGroup(scenario.fundamental_momentum_weights),
    valuation_weights: normalizeWeightGroup(scenario.valuation_weights),
    brakes: { ...scenario.brakes },
  }
}

function missionLabel(value: string | undefined) {
  return MISSION_OPTIONS.find(option => option.value === value)?.label ?? "Custom"
}

function toneForAction(action: string) {
  if (["Increase Long", "Press Short", "Research Long", "Research Short"].includes(action)) return "success"
  if (["Trim Long", "Cover Short", "Exit Review"].includes(action)) return "error"
  if (["Review", "Squeeze Review"].includes(action)) return "warning"
  return "neutral"
}

function toneForGate(gate: string) {
  if (gate === "pass") return "success"
  if (gate === "review") return "warning"
  return "neutral"
}

function badgeClass(tone: string) {
  if (tone === "success") return "theme-badge-success"
  if (tone === "warning") return "theme-badge-warning"
  if (tone === "error") return "theme-badge-error"
  if (tone === "info") return "theme-badge-info"
  return "theme-badge-neutral"
}

function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`theme-badge ${badgeClass(tone)}`}>{children}</span>
}

function actionIsHold(action: string) {
  return action === "Hold Long" || action === "Hold Short" || action === "Watch"
}

function SummaryCard({ title, value, detail }: { title: string; value: string; detail?: string }) {
  return (
    <section className="theme-surface-muted p-4">
      <p className="label-text">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-app">{value}</p>
      {detail && <p className="mt-1 text-xs text-subtle">{detail}</p>}
    </section>
  )
}

function FactorBreakdown({ factors }: { factors: AnalyzerFactorBreakdown[] }) {
  const activeFactors = factors.filter(factor => factor.status !== "disabled")
  if (activeFactors.length === 0) return <p className="text-sm text-subtle">No active factor evidence.</p>

  return (
    <div className="space-y-3">
      {activeFactors.map(factor => {
        const value = toNumber(factor.value)
        const width = value == null ? 0 : Math.min(100, Math.abs(value) / 3 * 100)
        const positive = value != null && value >= 0
        return (
          <div key={factor.factor} className="space-y-1.5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <span className="text-sm font-medium text-app">{factor.label}</span>
                <span className="ml-2 text-xs text-subtle">{factor.status.replace("_", " ")}</span>
              </div>
              <span className="mono-text text-sm font-semibold" style={{ color: value == null ? undefined : colorPositiveNegative(value) }}>
                {value == null ? "N/A" : formatScore(value)}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-card-muted">
              <div
                className={positive ? "h-full rounded-full bg-green-500" : "h-full rounded-full bg-red-500"}
                style={{ width: `${width}%` }}
              />
            </div>
            {factor.reason && <p className="text-xs text-subtle">{factor.reason}</p>}
          </div>
        )
      })}
    </div>
  )
}

function ActionQueue({
  actions,
  selectedTicker,
  onSelect,
}: {
  actions: AnalyzerCourseAction[]
  selectedTicker: string | null
  onSelect: (action: AnalyzerCourseAction) => void
}) {
  if (actions.length === 0) {
    return <p className="py-4 text-sm text-subtle">Run a mission to generate the action queue.</p>
  }

  return (
    <div className="overflow-x-auto rounded-[1rem] border border-app bg-card">
      <table className="w-full min-w-[900px] border-collapse text-sm">
        <thead className="bg-card-muted">
          <tr>
            <th className="px-4 py-3 text-left label-text">Ticker</th>
            <th className="px-4 py-3 text-left label-text">Action</th>
            <th className="px-4 py-3 text-left label-text">Direction</th>
            <th className="px-4 py-3 text-right label-text">Scenario</th>
            <th className="px-4 py-3 text-right label-text">Delta</th>
            <th className="px-4 py-3 text-right label-text">Confidence</th>
            <th className="px-4 py-3 text-left label-text">Band</th>
            <th className="px-4 py-3 text-left label-text">Gate</th>
            <th className="px-4 py-3 text-left label-text">Sizing</th>
          </tr>
        </thead>
        <tbody>
          {actions.map(action => {
            const selected = action.ticker === selectedTicker
            return (
              <tr
                key={action.ticker}
                onClick={() => onSelect(action)}
                className={`cursor-pointer border-t border-app transition-colors hover:bg-hover ${selected ? "bg-selected" : ""}`}
              >
                <td className="px-4 py-3 mono-text font-semibold text-app">{action.ticker}</td>
                <td className="px-4 py-3"><Badge tone={toneForAction(action.action)}>{action.action}</Badge></td>
                <td className="px-4 py-3 text-muted capitalize">{action.direction || "n/a"}</td>
                <td className="px-4 py-3 text-right mono-text font-semibold" style={{ color: colorPositiveNegative(action.scenario_score) }}>
                  {formatScore(action.scenario_score)}
                </td>
                <td className="px-4 py-3 text-right mono-text" style={{ color: colorPositiveNegative(action.score_delta) }}>
                  {formatScore(action.score_delta)}
                </td>
                <td className="px-4 py-3 text-right mono-text">{formatPercent(action.confidence)}</td>
                <td className="px-4 py-3 capitalize"><Badge tone={action.conviction_band === "none" ? "neutral" : "info"}>{action.conviction_band}</Badge></td>
                <td className="px-4 py-3"><Badge tone={toneForGate(action.gate_status)}>{action.gate_status}</Badge></td>
                <td className="px-4 py-3 text-muted">{action.sizing_implication?.implication ?? "review before sizing"}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ActionDetail({
  action,
  llmReady,
  brief,
  briefLoading,
  briefError,
  onGenerateBrief,
}: {
  action: AnalyzerCourseAction | null
  llmReady: boolean
  brief: string | null
  briefLoading: boolean
  briefError: string | null
  onGenerateBrief: () => void
}) {
  if (!action) {
    return (
      <section className="theme-surface p-5">
        <h2 className="section-title">Action Detail</h2>
        <p className="mt-3 text-sm text-subtle">Select an action to inspect evidence and gates.</p>
      </section>
    )
  }

  return (
    <section className="theme-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="label-text">{action.ticker}</p>
          <h2 className="mt-1 text-xl font-semibold text-app">{action.action}</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone={toneForAction(action.action)}>{action.conviction_band}</Badge>
          <Badge tone={toneForGate(action.gate_status)}>{action.gate_status}</Badge>
        </div>
      </div>

      <p className="mt-4 text-sm leading-6 text-muted">{action.deterministic_rationale}</p>

      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryCard title="Scenario" value={formatScore(action.scenario_score)} />
        <SummaryCard title="Delta" value={formatScore(action.score_delta)} />
        <SummaryCard title="Confidence" value={formatPercent(action.confidence)} />
        <SummaryCard title="Coverage" value={formatPercent(action.data_coverage?.ratio)} detail={`${action.data_coverage?.available ?? 0}/${action.data_coverage?.applicable ?? 0} factors`} />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <div>
          <h3 className="card-title">Factor Evidence</h3>
          <div className="mt-3">
            <FactorBreakdown factors={action.factor_breakdown ?? []} />
          </div>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="card-title">Sizing Implication</h3>
            <p className="mt-2 text-sm text-muted">{action.sizing_implication?.implication ?? "review before sizing"}</p>
            <p className="mt-1 text-xs text-subtle">{action.sizing_implication?.note ?? "Analysis only."}</p>
          </div>

          {(action.gate_reasons?.length || action.warnings?.length) ? (
            <div>
              <h3 className="card-title">Gates And Watch-Outs</h3>
              <ul className="mt-2 space-y-2 text-sm text-muted">
                {[...(action.gate_reasons ?? []), ...(action.warnings ?? [])].map((warning, index) => (
                  <li key={`${warning}-${index}`} className="rounded-lg border border-app bg-card-muted px-3 py-2">{warning}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="rounded-lg border border-app bg-card-muted px-3 py-2 text-sm text-muted">No gates or warnings on this action.</p>
          )}

          {llmReady && (
            <div>
              <button
                type="button"
                onClick={onGenerateBrief}
                disabled={briefLoading}
                className="theme-button-base theme-button-secondary min-h-10 px-3 text-xs"
              >
                <Sparkles className="h-4 w-4" />
                {briefLoading ? "Writing brief..." : "AI Brief"}
              </button>
              {briefError && <p className="mt-2 text-sm text-negative">{briefError}</p>}
              {brief && <div className="mt-3 whitespace-pre-wrap rounded-lg border border-app bg-card-muted p-3 text-sm text-muted">{brief}</div>}
            </div>
          )}
        </div>
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
    normalizeScenarioState(cachedState?.scenario),
  )
  const [cachedResult, setCachedResult] = useState<AnalyzerResponse | null>(cachedState?.result ?? null)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null)
  const [briefTicker, setBriefTicker] = useState<string | null>(null)

  const llmSettings = useApiQuery<LLMSettings>(LLM_SETTINGS_QUERY_KEY, fetchLLMSettings, 30_000)
  const llmReady = Boolean(
    llmSettings.data?.available_providers.find(provider => provider.provider === llmSettings.data?.provider)?.configured,
  )

  const mutation = useMutation({
    mutationFn: (nextScenario: AnalyzerScenarioState) => runPortfolioAnalyzerAsync({ scenario: toScenarioRequest(nextScenario) }),
    onSuccess: result => setCachedResult((result as AnalyzerResponse) ?? null),
  })

  const briefMutation = useMutation({
    mutationFn: generatePortfolioAnalyzerBrief,
    onSuccess: () => undefined,
  })

  useEffect(() => {
    queryClient.setQueryData(ANALYZER_STATE_KEY, { result: cachedResult, scenario })
  }, [cachedResult, queryClient, scenario])

  const data = (mutation.data as AnalyzerResponse | undefined) ?? cachedResult
  const rows = toRows(data?.weights_df)
  const course = data?.course_of_action
  const actionQueue = useMemo(() => course?.action_queue ?? [], [course?.action_queue])
  const selectedAction = useMemo(
    () => actionQueue.find(action => action.ticker === selectedTicker) ?? actionQueue[0] ?? null,
    [actionQueue, selectedTicker],
  )
  const summary = course?.summary

  useEffect(() => {
    if (!selectedTicker && actionQueue.length > 0) setSelectedTicker(actionQueue[0].ticker)
    if (selectedTicker && actionQueue.length > 0 && !actionQueue.some(action => action.ticker === selectedTicker)) {
      setSelectedTicker(actionQueue[0].ticker)
    }
  }, [actionQueue, selectedTicker])

  const actionableCount = actionQueue.filter(action => !actionIsHold(action.action) && action.gate_status !== "review").length
  const reviewCount = actionQueue.filter(action => action.gate_status === "review" || action.action.includes("Review")).length
  const dataWarnings = Object.values(summary?.data_quality_counts ?? {}).reduce((sum, value) => sum + Number(value || 0), 0)
  const asOf = summary?.as_of ? new Date(summary.as_of).toLocaleString() : "Not run"
  const visibleBrief = briefTicker === selectedAction?.ticker ? briefMutation.data?.brief ?? null : null
  const briefError = briefTicker === selectedAction?.ticker && briefMutation.isError ? String(briefMutation.error) : null

  function applyPreset(preset: Exclude<ScenarioPreset, "custom">) {
    setScenario(cloneScenario(SCENARIO_PRESETS[preset]))
  }

  function setFactorWeight(key: keyof AnalyzerScenarioState["factor_weights"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", factor_weights: rebalanceWeightGroup(prev.factor_weights, key, value) }))
  }

  function setFundamentalWeight(key: keyof AnalyzerScenarioState["fundamental_momentum_weights"], value: number) {
    setScenario(prev => ({
      ...prev,
      preset: "custom",
      fundamental_momentum_weights: rebalanceWeightGroup(prev.fundamental_momentum_weights, key, value),
    }))
  }

  function setValuationWeight(key: keyof AnalyzerScenarioState["valuation_weights"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", valuation_weights: rebalanceWeightGroup(prev.valuation_weights, key, value) }))
  }

  function setBrake(key: keyof AnalyzerScenarioState["brakes"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", brakes: { ...prev.brakes, [key]: value } }))
  }

  function handleBrief() {
    if (!selectedAction) return
    setBriefTicker(selectedAction.ticker)
    briefMutation.mutate(selectedAction)
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Analyzer</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          Course-of-action recommender for current portfolio directions. Analysis only; sizing remains in Portfolio Sizer.
        </p>
      </div>

      <div className="theme-surface mb-6 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="section-title">Mission</h2>
            <p className="mt-1 text-sm text-subtle">Choose the operating objective, then inspect the recommended action queue.</p>
          </div>
          <ActionButton
            onClick={() => mutation.mutate(scenario)}
            loading={mutation.isPending}
            loadingText="Running mission..."
            className="w-auto px-5"
          >
            Run Mission
          </ActionButton>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-3 lg:grid-cols-5">
          {MISSION_OPTIONS.map(option => {
            const active = scenario.preset === option.value
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => applyPreset(option.value)}
                className={`min-h-28 rounded-lg border p-4 text-left transition-colors ${
                  active ? "border-blue-200 bg-blue-50" : "border-app bg-card-muted hover:bg-hover"
                }`}
              >
                <span className="text-sm font-semibold text-app">{option.label}</span>
                <span className="mt-2 block text-xs leading-5 text-subtle">{option.description}</span>
              </button>
            )
          })}
        </div>

        <button
          type="button"
          onClick={() => setAdvancedOpen(open => !open)}
          className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-muted hover:text-app"
        >
          {advancedOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Advanced metric weights
          {scenario.preset === "custom" && <Badge tone="info">Custom</Badge>}
        </button>

        {advancedOpen && (
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
        )}
      </div>

      {mutation.isPending && <LoadingSpinner message="Running portfolio analyzer..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && !mutation.isError && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <SummaryCard title="Mission" value={missionLabel(summary?.mission ?? scenario.preset)} detail={scenario.preset === "custom" ? "Custom weights" : "Preset weights"} />
            <SummaryCard title="Actionable" value={String(actionableCount)} detail="Non-held pass-gated actions" />
            <SummaryCard title="Reviews" value={String(reviewCount)} detail="Gated or review actions" />
            <SummaryCard title="Data Warnings" value={String(dataWarnings)} detail={asOf} />
          </div>

          <section className="theme-surface p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="section-title">Action Queue</h2>
                <p className="mt-1 text-sm text-subtle">Ranked courses of action from absolute score, direction, confidence, gates, and data quality.</p>
              </div>
              <Badge tone="neutral">Analysis only</Badge>
            </div>
            <div className="mt-4">
              <ActionQueue
                actions={actionQueue}
                selectedTicker={selectedAction?.ticker ?? null}
                onSelect={action => {
                  setSelectedTicker(action.ticker)
                  briefMutation.reset()
                  setBriefTicker(null)
                }}
              />
            </div>
          </section>

          <ActionDetail
            action={selectedAction}
            llmReady={llmReady}
            brief={visibleBrief}
            briefLoading={briefTicker === selectedAction?.ticker && briefMutation.isPending}
            briefError={briefError}
            onGenerateBrief={handleBrief}
          />

          <details className="theme-surface p-5">
            <summary className="cursor-pointer list-none">
              <span className="section-title">Diagnostics</span>
              <span className="ml-3 text-sm text-subtle">Raw scenario metrics and legacy signal table</span>
            </summary>
            <div className="mt-4">
              <DataTable
                label="Scenario Metrics"
                columns={buildColumns(rows)}
                rows={rows}
              />
            </div>
          </details>
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Choose a mission and run the analyzer to generate the action queue.</p>
      )}
    </div>
  )
}
