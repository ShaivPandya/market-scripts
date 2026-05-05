import { useEffect, useMemo, useState, type ReactNode } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { Bell, ChevronDown, ChevronRight, Play, Send, Sparkles } from "lucide-react"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton, SliderInput } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { useApiQuery } from "@/hooks/useApiQuery"
import { colorPositiveNegative } from "@/lib/colors"
import {
  fetchLLMSettings,
  fetchOptimizationAlerts,
  fetchOptimizationMissions,
  fetchOptimizationRuns,
  generatePortfolioAnalyzerBrief,
  createAction,
  dismissOptimizationAlert,
  runOptimizationMissionAsync,
  runPortfolioAnalyzerAsync,
  type AnalyzerCourseAction,
  type AnalyzerCourseOfAction,
  type AnalyzerFactorBreakdown,
  type AnalyzerScenarioRequest,
  type LLMSettings,
  type OptimizationAlert,
  type OptimizationMission,
  type OptimizationRun,
  type StagedMutationResponse,
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
  metric_scores: {
    quality: number
    price_momentum: number
    revenue: number
    eps: number
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
const OPTIMIZATION_MISSIONS_QUERY_KEY = ["optimization", "missions"] as const
const OPTIMIZATION_ALERTS_QUERY_KEY = ["optimization", "alerts", "open"] as const
const OPTIMIZATION_RUNS_QUERY_KEY = ["optimization", "runs"] as const

const SCENARIO_PRESETS: Record<Exclude<ScenarioPreset, "custom">, AnalyzerScenarioState> = {
  balanced: {
    preset: "balanced",
    metric_scores: {
      quality: 30,
      price_momentum: 40,
      revenue: 20,
      eps: 10,
      price_sales: 0,
      price_operating_income: 0,
      price_fcf: 0,
      price_earnings: 0,
    },
    brakes: { drawdown_sensitivity: 0, contrarian_penalty: 0, short_squeeze_brake: 0 },
  },
  capital_preservation: {
    preset: "capital_preservation",
    metric_scores: {
      quality: 70,
      price_momentum: 30,
      revenue: 20,
      eps: 20,
      price_sales: 10,
      price_operating_income: 20,
      price_fcf: 20,
      price_earnings: 10,
    },
    brakes: { drawdown_sensitivity: 60, contrarian_penalty: 50, short_squeeze_brake: 60 },
  },
  momentum_exploitation: {
    preset: "momentum_exploitation",
    metric_scores: {
      quality: 30,
      price_momentum: 100,
      revenue: 30,
      eps: 30,
      price_sales: 0,
      price_operating_income: 0,
      price_fcf: 0,
      price_earnings: 0,
    },
    brakes: { drawdown_sensitivity: 10, contrarian_penalty: 10, short_squeeze_brake: 20 },
  },
  value_dislocation: {
    preset: "value_dislocation",
    metric_scores: {
      quality: 40,
      price_momentum: 20,
      revenue: 20,
      eps: 20,
      price_sales: 30,
      price_operating_income: 30,
      price_fcf: 30,
      price_earnings: 20,
    },
    brakes: { drawdown_sensitivity: 20, contrarian_penalty: 20, short_squeeze_brake: 30 },
  },
  short_defense: {
    preset: "short_defense",
    metric_scores: {
      quality: 60,
      price_momentum: 70,
      revenue: 30,
      eps: 20,
      price_sales: 0,
      price_operating_income: 0,
      price_fcf: 10,
      price_earnings: 0,
    },
    brakes: { drawdown_sensitivity: 30, contrarian_penalty: 20, short_squeeze_brake: 70 },
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

const SCORE_MIN = 0
const SCORE_MAX = 100
const SCORE_STEP = 10

type MetricScores = AnalyzerScenarioState["metric_scores"]
type BrakeScores = AnalyzerScenarioState["brakes"]
type LegacyScenarioState = Partial<AnalyzerScenarioState> & {
  factor_weights?: {
    quality?: number
    price_momentum?: number
    fundamental_momentum?: number
    valuation?: number
  }
  fundamental_momentum_weights?: {
    revenue?: number
    eps?: number
  }
  valuation_weights?: {
    price_sales?: number
    price_operating_income?: number
    price_fcf?: number
    price_earnings?: number
  }
}

function clampScore(value: number) {
  if (!Number.isFinite(value)) return SCORE_MIN
  const stepped = Math.round(value / SCORE_STEP) * SCORE_STEP
  return Math.min(SCORE_MAX, Math.max(SCORE_MIN, stepped))
}

function normalizeScoreMap<T extends Record<string, number>>(values: Partial<T> | undefined, defaults: T): T {
  const raw = (values ?? {}) as Partial<Record<keyof T, number>>
  return Object.fromEntries(
    Object.entries(defaults).map(([key, fallback]) => {
      const value = Number(raw[key as keyof T] ?? fallback)
      return [key, clampScore(value)]
    }),
  ) as T
}

function normalizeBrakeScores(values: Partial<BrakeScores> | undefined, defaults: BrakeScores): BrakeScores {
  const raw = values ?? {}
  return Object.fromEntries(
    Object.entries(defaults).map(([key, fallback]) => {
      const value = Number(raw[key as keyof BrakeScores] ?? fallback)
      const scoreValue = value > 0 && value <= 1 ? value * SCORE_MAX : value
      return [key, clampScore(scoreValue)]
    }),
  ) as BrakeScores
}

function cloneScenario(scenario: AnalyzerScenarioState): AnalyzerScenarioState {
  return {
    preset: scenario.preset,
    metric_scores: normalizeScoreMap(scenario.metric_scores, scenario.metric_scores),
    brakes: normalizeBrakeScores(scenario.brakes, scenario.brakes),
  }
}

function normalizeScenarioState(value: AnalyzerScenarioState | undefined): AnalyzerScenarioState {
  if (!value) return cloneScenario(SCENARIO_PRESETS.balanced)
  const rawPreset = value.preset
  const preset: ScenarioPreset = rawPreset === "custom" || rawPreset in SCENARIO_PRESETS ? rawPreset : "balanced"
  const base = preset === "custom" ? SCENARIO_PRESETS.balanced : SCENARIO_PRESETS[preset]
  const legacyValue = value as LegacyScenarioState
  return {
    preset: value.preset === "custom" ? "custom" : preset as ScenarioPreset,
    metric_scores: value.metric_scores
      ? normalizeScoreMap(value.metric_scores, base.metric_scores)
      : legacyWeightsToMetricScores(legacyValue, base.metric_scores),
    brakes: normalizeBrakeScores(value.brakes, base.brakes),
  }
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

function legacyWeightsToMetricScores(value: LegacyScenarioState, defaults: MetricScores): MetricScores {
  if (!value.factor_weights && !value.fundamental_momentum_weights && !value.valuation_weights) {
    return normalizeScoreMap(defaults, defaults)
  }

  const defaultValuationTotal =
    defaults.price_sales + defaults.price_operating_income + defaults.price_fcf + defaults.price_earnings
  const factorWeights = normalizeWeightGroup({
    quality: defaults.quality,
    price_momentum: defaults.price_momentum,
    fundamental_momentum: defaults.revenue + defaults.eps,
    valuation: defaultValuationTotal,
    ...(value.factor_weights ?? {}),
  })
  const fundamentalWeights = normalizeWeightGroup({
    revenue: defaults.revenue,
    eps: defaults.eps,
    ...(value.fundamental_momentum_weights ?? {}),
  })
  const valuationWeights = normalizeWeightGroup({
    price_sales: defaults.price_sales,
    price_operating_income: defaults.price_operating_income,
    price_fcf: defaults.price_fcf,
    price_earnings: defaults.price_earnings,
    ...(value.valuation_weights ?? {}),
  })

  return {
    quality: clampScore(factorWeights.quality * SCORE_MAX),
    price_momentum: clampScore(factorWeights.price_momentum * SCORE_MAX),
    revenue: clampScore(factorWeights.fundamental_momentum * fundamentalWeights.revenue * SCORE_MAX),
    eps: clampScore(factorWeights.fundamental_momentum * fundamentalWeights.eps * SCORE_MAX),
    price_sales: clampScore(factorWeights.valuation * valuationWeights.price_sales * SCORE_MAX),
    price_operating_income: clampScore(factorWeights.valuation * valuationWeights.price_operating_income * SCORE_MAX),
    price_fcf: clampScore(factorWeights.valuation * valuationWeights.price_fcf * SCORE_MAX),
    price_earnings: clampScore(factorWeights.valuation * valuationWeights.price_earnings * SCORE_MAX),
  }
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
    metric_scores: normalizeScoreMap(scenario.metric_scores, scenario.metric_scores),
    brakes: normalizeBrakeScores(scenario.brakes, scenario.brakes),
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

function toneForSeverity(severity: string | undefined | null) {
  if (severity === "urgent" || severity === "high") return "error"
  if (severity === "normal") return "warning"
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

function workspaceActionType(action: AnalyzerCourseAction) {
  if (["Research Long", "Research Short"].includes(action.action)) return "research"
  if (["Review", "Squeeze Review"].includes(action.action)) return "review"
  if (action.action === "Exit Review") return "exit"
  if (["Increase Long", "Trim Long", "Press Short", "Cover Short"].includes(action.action)) return "resize"
  return "review"
}

function workspaceActionVerb(action: AnalyzerCourseAction) {
  switch (action.action) {
    case "Increase Long":
      return "Increase long exposure"
    case "Trim Long":
      return "Trim long exposure"
    case "Press Short":
      return "Press short exposure"
    case "Cover Short":
      return "Cover short exposure"
    case "Research Long":
      return "Research long setup"
    case "Research Short":
      return "Research short setup"
    case "Exit Review":
      return "Review exit"
    case "Squeeze Review":
      return "Review short squeeze risk"
    case "Review":
      return "Review position"
    default:
      return action.action || "Review position"
  }
}

function workspaceActionUrgency(action: AnalyzerCourseAction) {
  const band = String(action.conviction_band || "").toLowerCase()
  if (band === "large") return "high"
  if (band === "small" || band === "none" || action.gate_status === "watch") return "low"
  return "normal"
}

function workspaceActionDescription(action: AnalyzerCourseAction) {
  const lines = [
    `${workspaceActionVerb(action)} for ${action.ticker}.`,
    `Analyzer result: ${action.action}; direction ${action.direction || "n/a"}; conviction ${action.conviction_band}; gate ${action.gate_status}; confidence ${formatPercent(action.confidence)}.`,
    `Scenario ${formatScore(action.scenario_score)}, delta ${formatScore(action.score_delta)}.`,
  ]
  if (action.sizing_implication?.implication) {
    lines.push(`Sizing implication: ${action.sizing_implication.implication}.`)
  }
  if (action.deterministic_rationale) {
    lines.push(`Evidence: ${action.deterministic_rationale}`)
  }
  return lines.join("\n")
}

function toWorkspaceActionRequest(action: AnalyzerCourseAction) {
  return {
    ticker: action.ticker,
    action_type: workspaceActionType(action),
    urgency: workspaceActionUrgency(action),
    description: workspaceActionDescription(action),
    reason: `Stage portfolio analyzer result for ${action.ticker}: ${action.action}`,
  }
}

function formatDateTime(value: string | undefined | null) {
  if (!value) return "Not run"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
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
  workspaceProposal,
  workspaceLoading,
  workspaceError,
  onGenerateBrief,
  onStageWorkspaceAction,
}: {
  action: AnalyzerCourseAction | null
  llmReady: boolean
  brief: string | null
  briefLoading: boolean
  briefError: string | null
  workspaceProposal: StagedMutationResponse | null
  workspaceLoading: boolean
  workspaceError: string | null
  onGenerateBrief: () => void
  onStageWorkspaceAction: () => void
}) {
  if (!action) {
    return (
      <section className="theme-surface p-5">
        <h2 className="section-title">Action Detail</h2>
        <p className="mt-3 text-sm text-subtle">Select an action to inspect evidence and gates.</p>
      </section>
    )
  }

  const canStageWorkspaceAction = !actionIsHold(action.action)

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
            <h3 className="card-title">Workspace Action</h3>
            <p className="mt-2 text-sm text-muted">
              Stage this analyzer result as an internal action item. It will show in Workspace as a proposal before anything is applied.
            </p>
            <ActionButton
              onClick={onStageWorkspaceAction}
              loading={workspaceLoading}
              loadingText="Staging..."
              disabled={!canStageWorkspaceAction}
              className="mt-3 w-auto px-3 text-xs"
            >
              <Send className="h-4 w-4" />
              Stage Workspace Action
            </ActionButton>
            {!canStageWorkspaceAction && (
              <p className="mt-2 text-xs text-subtle">Hold and watch rows are not staged as action items.</p>
            )}
            {workspaceError && <p className="mt-2 text-sm text-negative">{workspaceError}</p>}
            {workspaceProposal && (
              <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
                Proposal #{workspaceProposal.approval_id} staged for {workspaceProposal.action_id.replace(/_/g, " ")}.{" "}
                <Link to="/workspace" className="font-semibold underline underline-offset-2">Review in Workspace</Link>.
              </div>
            )}
          </div>

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
              {brief && (
                <div className="mt-3 rounded-lg border border-app bg-card-muted p-3 text-sm text-muted">
                  <MarkdownRenderer content={brief} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function ContinuousOptimizationPanel({
  mission,
  latestRun,
  alerts,
  running,
  dismissingId,
  onRun,
  onDismiss,
}: {
  mission: OptimizationMission | null
  latestRun: OptimizationRun | null
  alerts: OptimizationAlert[]
  running: boolean
  dismissingId: number | null
  onRun: () => void
  onDismiss: (alert: OptimizationAlert) => void
}) {
  const statusTone = mission?.status === "active" ? "success" : mission?.status === "paused" ? "warning" : "neutral"

  return (
    <section className="theme-surface mb-6 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="section-title">Continuous Optimization</h2>
            {mission && <Badge tone={statusTone}>{mission.status}</Badge>}
          </div>
          <p className="mt-1 text-sm text-subtle">
            Mission alerts stage only material decision-state changes for human review.
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!mission || running}
          className="theme-button-base theme-button-primary min-h-10 px-4 text-sm"
        >
          <Play className="h-4 w-4" />
          {running ? "Starting..." : "Run Now"}
        </button>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-4">
        <SummaryCard title="Mission" value={mission?.name ?? "Loading"} detail={mission?.schedule_label ?? "Weekdays at 10:15 ET"} />
        <SummaryCard title="Last Run" value={latestRun?.status ?? "None"} detail={formatDateTime(latestRun?.completed_at ?? latestRun?.started_at)} />
        <SummaryCard title="Open Alerts" value={String(alerts.length)} detail="Material action, gate, band, confidence, or risk changes" />
        <SummaryCard title="Mode" value="Stage" detail="No orders or position mutations" />
      </div>

      <div className="mt-5">
        <div className="mb-3 flex items-center gap-2">
          <Bell className="h-4 w-4 text-muted" />
          <h3 className="card-title">Changed Actions Queue</h3>
        </div>
        {alerts.length === 0 ? (
          <p className="rounded-lg border border-app bg-card-muted px-3 py-3 text-sm text-subtle">
            No open optimizer alerts. Repeated runs with unchanged action state are suppressed.
          </p>
        ) : (
          <div className="space-y-3">
            {alerts.map(alert => {
              const previous = alert.previous_snapshot
              const current = alert.current_snapshot
              return (
                <article key={alert.id} className="rounded-lg border border-app bg-card-muted p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="mono-text text-sm font-semibold text-app">{alert.ticker ?? "PORTFOLIO"}</span>
                        <Badge tone={toneForSeverity(alert.severity)}>{alert.severity}</Badge>
                        <Badge tone="neutral">{alert.alert_type.replace(/_/g, " ")}</Badge>
                      </div>
                      <p className="mt-2 text-sm text-muted">{alert.change_summary}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => onDismiss(alert)}
                      disabled={dismissingId === alert.id}
                      className="theme-button-base theme-button-secondary min-h-9 px-3 text-xs"
                    >
                      {dismissingId === alert.id ? "Dismissing..." : "Dismiss"}
                    </button>
                  </div>
                  <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                    <div className="rounded-lg border border-app bg-card px-3 py-2">
                      <p className="label-text">Prior</p>
                      <p className="mt-1 text-sm text-muted">
                        {previous ? `${previous.action} · ${previous.conviction_band ?? "none"} · ${previous.gate_status ?? "gate n/a"}` : "No prior state"}
                      </p>
                    </div>
                    <div className="rounded-lg border border-app bg-card px-3 py-2">
                      <p className="label-text">Current</p>
                      <p className="mt-1 text-sm text-app">
                        {current ? `${current.action} · ${current.conviction_band ?? "none"} · ${current.gate_status ?? "gate n/a"}` : "Missing current state"}
                      </p>
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-subtle">
                    Approval: {alert.action_item_approval_id ? `#${alert.action_item_approval_id}` : "not staged"}
                  </p>
                </article>
              )
            })}
          </div>
        )}
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
  const [workspaceProposal, setWorkspaceProposal] = useState<{ ticker: string; response: StagedMutationResponse } | null>(null)

  const llmSettings = useApiQuery<LLMSettings>(LLM_SETTINGS_QUERY_KEY, fetchLLMSettings, 30_000)
  const llmReady = Boolean(
    llmSettings.data?.available_providers.find(provider => provider.provider === llmSettings.data?.provider)?.configured,
  )
  const optimizerMissions = useApiQuery<{ missions: OptimizationMission[]; count: number }>(
    OPTIMIZATION_MISSIONS_QUERY_KEY,
    fetchOptimizationMissions,
    60_000,
  )
  const optimizerAlerts = useApiQuery<{ alerts: OptimizationAlert[]; count: number }>(
    OPTIMIZATION_ALERTS_QUERY_KEY,
    () => fetchOptimizationAlerts({ status: "open", limit: 10 }),
    30_000,
  )
  const optimizerRuns = useApiQuery<{ runs: OptimizationRun[]; count: number }>(
    OPTIMIZATION_RUNS_QUERY_KEY,
    () => fetchOptimizationRuns({ limit: 5 }),
    30_000,
  )

  const mutation = useMutation({
    mutationFn: (nextScenario: AnalyzerScenarioState) => runPortfolioAnalyzerAsync({ scenario: toScenarioRequest(nextScenario) }),
    onSuccess: result => setCachedResult((result as AnalyzerResponse) ?? null),
  })

  const briefMutation = useMutation({
    mutationFn: generatePortfolioAnalyzerBrief,
    onSuccess: () => undefined,
  })

  const workspaceActionMutation = useMutation({
    mutationFn: (action: AnalyzerCourseAction) => createAction(toWorkspaceActionRequest(action)),
    onSuccess: (response, action) => {
      setWorkspaceProposal({ ticker: action.ticker, response })
      void queryClient.invalidateQueries({ queryKey: ["workspace"] })
      void queryClient.invalidateQueries({ queryKey: ["actions"] })
    },
  })

  const runOptimizerMutation = useMutation({
    mutationFn: (missionId: number) => runOptimizationMissionAsync(missionId, { source: "manual" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: OPTIMIZATION_ALERTS_QUERY_KEY })
      void queryClient.invalidateQueries({ queryKey: OPTIMIZATION_RUNS_QUERY_KEY })
    },
  })

  const dismissAlertMutation = useMutation({
    mutationFn: (alert: OptimizationAlert) => dismissOptimizationAlert(alert.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: OPTIMIZATION_ALERTS_QUERY_KEY })
      void queryClient.invalidateQueries({ queryKey: OPTIMIZATION_RUNS_QUERY_KEY })
    },
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
  const optimizationMission = optimizerMissions.data?.missions?.[0] ?? null
  const latestOptimizationRun = optimizerRuns.data?.runs?.[0] ?? null
  const openOptimizerAlerts = optimizerAlerts.data?.alerts ?? []

  const actionableCount = actionQueue.filter(action => !actionIsHold(action.action) && action.gate_status !== "review").length
  const reviewCount = actionQueue.filter(action => action.gate_status === "review" || action.action.includes("Review")).length
  const dataWarnings = Object.values(summary?.data_quality_counts ?? {}).reduce((sum, value) => sum + Number(value || 0), 0)
  const asOf = summary?.as_of ? new Date(summary.as_of).toLocaleString() : "Not run"
  const visibleBrief = briefTicker === selectedAction?.ticker ? briefMutation.data?.brief ?? null : null
  const briefError = briefTicker === selectedAction?.ticker && briefMutation.isError ? String(briefMutation.error) : null
  const selectedActionTicker = selectedAction?.ticker ?? null
  const visibleWorkspaceProposal =
    workspaceProposal && selectedActionTicker != null && workspaceProposal.ticker === selectedActionTicker
      ? workspaceProposal.response
      : null
  const workspaceActionError =
    selectedActionTicker != null &&
    workspaceActionMutation.variables?.ticker === selectedActionTicker &&
    workspaceActionMutation.isError
      ? String(workspaceActionMutation.error)
      : null
  const workspaceActionLoading =
    selectedActionTicker != null &&
    workspaceActionMutation.variables?.ticker === selectedActionTicker &&
    workspaceActionMutation.isPending

  function applyPreset(preset: Exclude<ScenarioPreset, "custom">) {
    setScenario(cloneScenario(SCENARIO_PRESETS[preset]))
  }

  function setMetricScore(key: keyof AnalyzerScenarioState["metric_scores"], value: number) {
    setScenario(prev => ({
      ...prev,
      preset: "custom",
      metric_scores: { ...prev.metric_scores, [key]: clampScore(value) },
    }))
  }

  function setBrake(key: keyof AnalyzerScenarioState["brakes"], value: number) {
    setScenario(prev => ({ ...prev, preset: "custom", brakes: { ...prev.brakes, [key]: clampScore(value) } }))
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

      <ContinuousOptimizationPanel
        mission={optimizationMission}
        latestRun={latestOptimizationRun}
        alerts={openOptimizerAlerts}
        running={runOptimizerMutation.isPending}
        dismissingId={dismissAlertMutation.variables?.id ?? null}
        onRun={() => {
          if (optimizationMission) runOptimizerMutation.mutate(optimizationMission.id)
        }}
        onDismiss={alert => dismissAlertMutation.mutate(alert)}
      />
      {runOptimizerMutation.isError && <ErrorMessage message={String(runOptimizerMutation.error)} />}
      {dismissAlertMutation.isError && <ErrorMessage message={String(dismissAlertMutation.error)} />}

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
          Advanced metric scores
          {scenario.preset === "custom" && <Badge tone="info">Custom</Badge>}
        </button>

        {advancedOpen && (
          <div className="mt-5 grid grid-cols-1 gap-6 xl:grid-cols-4">
            <section className="space-y-4">
              <h3 className="text-sm font-semibold text-app">Signal Scores</h3>
              <SliderInput label="Quality" value={scenario.metric_scores.quality} onChange={v => setMetricScore("quality", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="Price Momentum" value={scenario.metric_scores.price_momentum} onChange={v => setMetricScore("price_momentum", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
            </section>

            <section className="space-y-4">
              <h3 className="text-sm font-semibold text-app">Fundamental Momentum</h3>
              <SliderInput label="Revenue" value={scenario.metric_scores.revenue} onChange={v => setMetricScore("revenue", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="EPS" value={scenario.metric_scores.eps} onChange={v => setMetricScore("eps", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
            </section>

            <section className="space-y-4">
              <h3 className="text-sm font-semibold text-app">Valuation</h3>
              <SliderInput label="P/S" value={scenario.metric_scores.price_sales} onChange={v => setMetricScore("price_sales", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="P/Operating Income" value={scenario.metric_scores.price_operating_income} onChange={v => setMetricScore("price_operating_income", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="P/FCF" value={scenario.metric_scores.price_fcf} onChange={v => setMetricScore("price_fcf", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="P/E" value={scenario.metric_scores.price_earnings} onChange={v => setMetricScore("price_earnings", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
            </section>

            <section className="space-y-4">
              <h3 className="text-sm font-semibold text-app">Risk Brakes</h3>
              <SliderInput label="Drawdown" value={scenario.brakes.drawdown_sensitivity} onChange={v => setBrake("drawdown_sensitivity", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="Contrarian" value={scenario.brakes.contrarian_penalty} onChange={v => setBrake("contrarian_penalty", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
              <SliderInput label="Short Squeeze" value={scenario.brakes.short_squeeze_brake} onChange={v => setBrake("short_squeeze_brake", v)} min={SCORE_MIN} max={SCORE_MAX} step={SCORE_STEP} />
            </section>
          </div>
        )}
      </div>

      {mutation.isPending && <LoadingSpinner message="Running portfolio analyzer..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && !mutation.isError && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <SummaryCard title="Mission" value={missionLabel(summary?.mission ?? scenario.preset)} detail={scenario.preset === "custom" ? "Custom scores" : "Preset scores"} />
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
            workspaceProposal={visibleWorkspaceProposal}
            workspaceLoading={workspaceActionLoading}
            workspaceError={workspaceActionError}
            onGenerateBrief={handleBrief}
            onStageWorkspaceAction={() => {
              if (selectedAction) workspaceActionMutation.mutate(selectedAction)
            }}
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
