import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import { useApiQuery } from "@/hooks/useApiQuery"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { ActionButton, SegmentedControl, SelectInput, TextInput } from "@/components/shared/FormControls"
import { cn } from "@/lib/utils"
import {
  deletePositionValueRange,
  fetchPositionValuation,
  updatePositionValuationProfileOverride,
  updatePositionValueRange,
  type PositionValuation,
  type PositionValueRangeAssumption,
  type PositionValueRangeRequest,
  type PositionValueRangeScenario,
} from "@/lib/api"

const VALUATION_METRIC_ORDER = [
  "price_sales",
  "price_ebitda",
  "price_operating_income",
  "price_fcf",
  "price_earnings",
  "price_book",
] as const
type ValuationMetricKey = typeof VALUATION_METRIC_ORDER[number]

const DCF_VALUE_RANGE_METHODS = [
  "dcf_gordon_growth",
  "dcf_ev_ebitda",
  "dcf_ev_revenue",
] as const

const VALUE_RANGE_METHOD_ORDER = [...VALUATION_METRIC_ORDER, ...DCF_VALUE_RANGE_METHODS] as const
type ValueRangeMetricKey = typeof VALUE_RANGE_METHOD_ORDER[number]

const VALUE_RANGE_SCENARIOS = ["bear", "base", "bull"] as const
type ValueRangeScenarioKey = typeof VALUE_RANGE_SCENARIOS[number]

const VALUE_RANGE_SCENARIO_LABELS: Record<ValueRangeScenarioKey, string> = {
  bear: "Bear",
  base: "Base",
  bull: "Bull",
}

type ValuationTabView = "inputs" | "football_field"

const ENTERPRISE_VALUE_RANGE_METHODS = new Set<ValueRangeMetricKey>([
  "price_sales",
  "price_ebitda",
  "price_operating_income",
  "price_fcf",
  "dcf_ev_ebitda",
  "dcf_ev_revenue",
  "dcf_gordon_growth",
])
const PER_SHARE_VALUE_RANGE_METRICS = new Set<ValuationMetricKey>(["price_earnings"])
const DCF_GORDON_GROWTH_METHOD = "dcf_gordon_growth" as const

interface ValueRangeDraft {
  scenarios: Record<ValueRangeScenarioKey, { multiple: string; denominator: string }>
}

type ValueRangeDraftMap = Partial<Record<ValueRangeMetricKey, ValueRangeDraft["scenarios"]>>

interface ComputedValueRangeScenario {
  multiple: number | null
  terminalGrowth: number | null
  wacc: number | null
  denominator: number | null
  denominatorConverted: number | null
  expectedPrice: number | null
  percentChange: number | null
  status: string
  reason: string | null
}

function formatMultipleValue(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  if (Math.abs(value) >= 100) return `${value.toFixed(0)}x`
  return `${value.toFixed(1)}x`
}

function cleanCurrency(value: unknown): string | null {
  const text = String(value ?? "").trim()
  return text || null
}

function sameCurrency(a: unknown, b: unknown): boolean {
  const left = cleanCurrency(a)
  const right = cleanCurrency(b)
  return Boolean(left && right && left.toUpperCase() === right.toUpperCase())
}

function currencyPrefix(currency: unknown): string {
  const code = cleanCurrency(currency)
  return code ? `${code} ` : ""
}

function priceCurrency(data: PositionValuation): string | null {
  return cleanCurrency(data.currency_context?.price_currency ?? data.market_data?.price_currency ?? data.market_data?.currency ?? data.value_range?.output_currency)
}

function financialCurrency(data: PositionValuation): string | null {
  return cleanCurrency(data.currency_context?.financial_currency ?? data.market_data?.financial_currency)
}

function valueRangeOutputCurrency(data: PositionValuation): string | null {
  return cleanCurrency(data.value_range?.output_currency ?? data.value_range?.currency ?? priceCurrency(data))
}

function formatValuationMoney(value: unknown, currency?: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  const abs = Math.abs(value)
  const prefix = currencyPrefix(currency)
  if (abs >= 1e12) return `${prefix}${(value / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${prefix}${(value / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${prefix}${(value / 1e6).toFixed(1)}M`
  return `${prefix}${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function formatValuationDenominator(metric: ValuationMetricKey, value: unknown, currency?: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  if (PER_SHARE_VALUE_RANGE_METRICS.has(metric)) {
    return `${currencyPrefix(currency)}${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }
  return formatValuationMoney(value, currency)
}

function formatValuationPct(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return `${Math.round(value)}%`
}

function formatWeight(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "0%"
  return `${Math.round(value * 100)}%`
}

function isValuationMetricKey(value: unknown): value is ValuationMetricKey {
  return typeof value === "string" && (VALUATION_METRIC_ORDER as readonly string[]).includes(value)
}

function isValueRangeMetricKey(value: unknown): value is ValueRangeMetricKey {
  return typeof value === "string" && (VALUE_RANGE_METHOD_ORDER as readonly string[]).includes(value)
}

function isDcfGordonGrowthMetric(metric: ValueRangeMetricKey): metric is typeof DCF_GORDON_GROWTH_METHOD {
  return metric === DCF_GORDON_GROWTH_METHOD
}

function valueRangeMetricLabel(data: PositionValuation, metric: ValueRangeMetricKey): string {
  if (isValuationMetricKey(metric)) return data.metrics[metric]?.label ?? metric
  if (metric === "dcf_gordon_growth") return "DCF (Gordon Growth)"
  if (metric === "dcf_ev_ebitda") return "DCF (EV/EBITDA)"
  if (metric === "dcf_ev_revenue") return "DCF (EV/Revenue)"
  return metric
}

function valueRangeDenominatorLabel(metric: ValueRangeMetricKey, currency: string | null): string {
  const suffix = currency ? ` (${currency})` : ""
  if (metric === "dcf_gordon_growth") return `Terminal UFCF${suffix}`
  if (metric === "dcf_ev_ebitda") return `Terminal EBITDA${suffix}`
  if (metric === "dcf_ev_revenue") return `Terminal Revenue${suffix}`
  return currency ? `Denominator (${currency})` : "Denominator"
}

function valueRangeFirstInputLabel(metric: ValueRangeMetricKey): string {
  return isDcfGordonGrowthMetric(metric) ? "Terminal Growth (%)" : "Multiple"
}

function finiteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  return value
}

function positiveNumber(value: unknown): number | null {
  const parsed = finiteNumber(value)
  return parsed != null && parsed > 0 ? parsed : null
}

function trimNumberText(value: string): string {
  return value.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1")
}

function formatInputNumber(value: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return ""
  const abs = Math.abs(parsed)
  if (abs >= 1e12) return `${trimNumberText((parsed / 1e12).toFixed(2))}T`
  if (abs >= 1e9) return `${trimNumberText((parsed / 1e9).toFixed(2))}B`
  if (abs >= 1e6) return `${trimNumberText((parsed / 1e6).toFixed(2))}M`
  return trimNumberText(parsed.toFixed(abs >= 100 ? 0 : 2))
}

function formatMultipleInput(value: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return ""
  return trimNumberText(parsed.toFixed(2))
}

function formatPercentInput(value: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return ""
  return trimNumberText((parsed * 100).toFixed(2))
}

function parseMultipleInput(value: string): number | null {
  const cleaned = value.trim().replace(/x$/i, "").replace(/,/g, "")
  if (!cleaned) return null
  const parsed = Number(cleaned)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function parseTerminalGrowthInput(value: string): number | null {
  const cleaned = value.trim().replace(/%$/i, "").replace(/,/g, "")
  if (!cleaned) return null
  const parsed = Number(cleaned)
  return Number.isFinite(parsed) && parsed > -100 ? parsed / 100 : null
}

function parseWaccInput(value: string): number | null {
  const cleaned = value.trim().replace(/%$/i, "").replace(/,/g, "")
  if (!cleaned) return null
  const parsed = Number(cleaned)
  return Number.isFinite(parsed) && parsed > 0 && parsed < 100 ? parsed / 100 : null
}

function parseScaledNumberInput(value: string): number | null {
  const cleaned = value.trim().replace(/[$,\s]/g, "").toUpperCase()
  if (!cleaned) return null
  const match = cleaned.match(/^([+-]?(?:\d+\.?\d*|\.\d+))([MBT])?$/)
  if (!match) return null
  const base = Number(match[1])
  if (!Number.isFinite(base) || base <= 0) return null
  const suffix = match[2]
  const multiplier = suffix === "T" ? 1e12 : suffix === "B" ? 1e9 : suffix === "M" ? 1e6 : 1
  return base * multiplier
}

function formatSharePrice(value: unknown, currency?: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return "N/A"
  return `${currencyPrefix(currency)}${parsed.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatScenarioChange(value: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return "N/A"
  const sign = parsed > 0 ? "+" : ""
  return `${sign}${parsed.toFixed(1)}%`
}

function scenarioChangeClass(value: unknown): string {
  const parsed = finiteNumber(value)
  if (parsed == null) return "text-subtle"
  if (parsed > 0) return "text-green-600 dark:text-green-300"
  if (parsed < 0) return "text-red-600 dark:text-red-300"
  return "text-subtle"
}

function valueRangeDefaultMetric(data: PositionValuation): ValueRangeMetricKey {
  let selected: ValuationMetricKey | null = null
  let selectedWeight = 0
  for (const metric of VALUATION_METRIC_ORDER) {
    const weight = positiveNumber(data.profile.effective_weights?.[metric])
    const denominator = positiveNumber(data.metrics[metric]?.denominator)
    const multiple = positiveNumber(data.metrics[metric]?.value)
    if (weight != null && denominator != null && multiple != null && weight > selectedWeight) {
      selected = metric
      selectedWeight = weight
    }
  }
  if (selected) return selected

  for (const metric of VALUATION_METRIC_ORDER) {
    const denominator = positiveNumber(data.metrics[metric]?.denominator)
    const multiple = positiveNumber(data.metrics[metric]?.value)
    if (denominator != null && multiple != null) return metric
  }
  return "price_sales"
}

function blankValueRangeScenarioDrafts(): ValueRangeDraft["scenarios"] {
  return {
    bear: { multiple: "", denominator: "" },
    base: { multiple: "", denominator: "" },
    bull: { multiple: "", denominator: "" },
  }
}

function valueRangeSelectedMetric(data: PositionValuation): ValueRangeMetricKey {
  const selected = data.value_range?.selected_metric ?? data.value_range?.metric
  return isValueRangeMetricKey(selected) ? selected : valueRangeDefaultMetric(data)
}

function valueRangeAssumptions(data: PositionValuation): Partial<Record<ValueRangeMetricKey, PositionValueRangeAssumption>> {
  const raw = data.value_range?.metric_assumptions
  const assumptions: Partial<Record<ValueRangeMetricKey, PositionValueRangeAssumption>> = {}
  if (raw && typeof raw === "object") {
    for (const [metric, value] of Object.entries(raw)) {
      if (isValueRangeMetricKey(metric) && value && typeof value === "object") {
        assumptions[metric] = value
      }
    }
  }
  if (Object.keys(assumptions).length === 0 && data.value_range?.saved && isValueRangeMetricKey(data.value_range.metric)) {
    assumptions[data.value_range.metric] = {
      denominator_currency: data.value_range.stored_denominator_currency ?? data.value_range.denominator_currency ?? null,
      source_denominator_currency: Boolean(data.value_range.source_denominator_currency),
      wacc: data.value_range.wacc ?? null,
      scenarios: data.value_range.scenarios ?? {},
    }
  }
  return assumptions
}

function valueRangeCurrencyConversionRate(data: PositionValuation, from: unknown, to: unknown): number | null {
  if (sameCurrency(from, to)) return 1
  const explicit = positiveNumber(data.value_range?.denominator_to_price_fx_rate)
  const explicitDenominator = cleanCurrency(data.value_range?.denominator_currency ?? financialCurrency(data))
  const outputCurrency = valueRangeOutputCurrency(data)
  if (explicit != null && sameCurrency(from, explicitDenominator) && sameCurrency(to, outputCurrency)) {
    return explicit
  }
  if (explicit != null && sameCurrency(from, outputCurrency) && sameCurrency(to, explicitDenominator)) {
    return 1 / explicit
  }
  const contextRate = positiveNumber(data.currency_context?.financial_to_price_fx_rate)
  if (contextRate != null && sameCurrency(from, financialCurrency(data)) && sameCurrency(to, priceCurrency(data))) {
    return contextRate
  }
  if (contextRate != null && sameCurrency(from, priceCurrency(data)) && sameCurrency(to, financialCurrency(data))) {
    return 1 / contextRate
  }
  return null
}

function valueRangeDisplayContext(data: PositionValuation, assumption?: PositionValueRangeAssumption | null) {
  const outputCurrency = valueRangeOutputCurrency(data)
  const financial = financialCurrency(data) ?? outputCurrency
  const storedCurrency = cleanCurrency(assumption?.denominator_currency) ?? (assumption?.source_denominator_currency ? priceCurrency(data) : financial)
  let denominatorCurrency = financial ?? storedCurrency
  let displayRate = valueRangeCurrencyConversionRate(data, storedCurrency, denominatorCurrency)
  if (displayRate == null) {
    denominatorCurrency = storedCurrency
    displayRate = 1
  }
  const denominatorToPriceRate = valueRangeCurrencyConversionRate(data, denominatorCurrency, outputCurrency)
  return { denominatorCurrency, denominatorToPriceRate, displayRate }
}

function valueRangeDraftFromAssumption(
  data: PositionValuation,
  metric: ValueRangeMetricKey,
  assumption: PositionValueRangeAssumption,
): ValueRangeDraft["scenarios"] {
  const { displayRate } = valueRangeDisplayContext(data, assumption)
  return VALUE_RANGE_SCENARIOS.reduce((acc, scenario) => {
    const row = assumption.scenarios?.[scenario]
    const denominator = positiveNumber(row?.denominator)
    acc[scenario] = {
      multiple: isDcfGordonGrowthMetric(metric) ? formatPercentInput(row?.terminal_growth) : formatMultipleInput(row?.multiple),
      denominator: formatInputNumber(denominator != null ? denominator * displayRate : null),
    }
    return acc
  }, {} as ValueRangeDraft["scenarios"])
}

function valueRangeInitialDrafts(data: PositionValuation): ValueRangeDraftMap {
  const assumptions = valueRangeAssumptions(data)
  const drafts = VALUE_RANGE_METHOD_ORDER.reduce((acc, metric) => {
    acc[metric] = assumptions[metric] ? valueRangeDraftFromAssumption(data, metric, assumptions[metric]) : blankValueRangeScenarioDrafts()
    return acc
  }, {} as ValueRangeDraftMap)
  return drafts
}

function valueRangeInitialWaccDrafts(data: PositionValuation): Partial<Record<ValueRangeMetricKey, string>> {
  const assumptions = valueRangeAssumptions(data)
  return VALUE_RANGE_METHOD_ORDER.reduce((acc, metric) => {
    const wacc = assumptions[metric]?.wacc
    acc[metric] = formatPercentInput(wacc ?? (isDcfGordonGrowthMetric(metric) ? 0.1 : null))
    return acc
  }, {} as Partial<Record<ValueRangeMetricKey, string>>)
}

function inferredShareCount(data: PositionValuation): number | null {
  const direct = positiveNumber(data.value_range?.shares ?? data.market_data?.shares_outstanding)
  if (direct != null) return direct
  const marketCap = positiveNumber(data.market_data?.market_cap)
  const price = positiveNumber(data.market_data?.current_price)
  return marketCap != null && price != null ? marketCap / price : null
}

function computeDraftValueRangeScenario(
  data: PositionValuation,
  metric: ValueRangeMetricKey,
  draft: ValueRangeDraft["scenarios"][ValueRangeScenarioKey],
  denominatorToPriceRate: number | null,
  waccDraft: string,
): ComputedValueRangeScenario {
  const terminalGrowth = isDcfGordonGrowthMetric(metric) ? parseTerminalGrowthInput(draft.multiple) : null
  const multiple = isDcfGordonGrowthMetric(metric) ? null : parseMultipleInput(draft.multiple)
  const wacc = isDcfGordonGrowthMetric(metric) ? parseWaccInput(waccDraft) : null
  const denominator = parseScaledNumberInput(draft.denominator)
  const denominatorConverted = denominator != null && denominatorToPriceRate != null ? denominator * denominatorToPriceRate : null
  const shares = inferredShareCount(data)
  const currentPrice = positiveNumber(data.market_data?.current_price)
  const netDebt = finiteNumber(data.value_range?.net_debt ?? data.market_data?.net_debt)

  if (isDcfGordonGrowthMetric(metric)) {
    if (terminalGrowth == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing terminal growth" }
    if (wacc == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing WACC" }
    if (denominator == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing denominator" }
    if (denominatorToPriceRate == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing fx rate" }
    if (shares == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing shares" }
    if (netDebt == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing net debt" }
    if (wacc <= terminalGrowth) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "not_meaningful", reason: "WACC must exceed growth" }
    const grossValue = (denominatorConverted ?? 0) * (1 + terminalGrowth) / (wacc - terminalGrowth)
    const equityValue = grossValue - netDebt
    if (!Number.isFinite(equityValue) || equityValue <= 0) {
      return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "not_meaningful", reason: "non-positive equity value" }
    }
    const expectedPrice = equityValue / shares
    const percentChange = currentPrice != null ? (expectedPrice / currentPrice - 1) * 100 : null
    return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice, percentChange, status: "ok", reason: null }
  }

  if (multiple == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing multiple" }
  if (denominator == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing denominator" }
  if (denominatorToPriceRate == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing fx rate" }
  if (!PER_SHARE_VALUE_RANGE_METRICS.has(metric as ValuationMetricKey) && shares == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing shares" }
  if (ENTERPRISE_VALUE_RANGE_METHODS.has(metric) && netDebt == null) {
    return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing net debt" }
  }

  if (PER_SHARE_VALUE_RANGE_METRICS.has(metric as ValuationMetricKey)) {
    const expectedPrice = multiple * (denominatorConverted ?? 0)
    if (!Number.isFinite(expectedPrice) || expectedPrice <= 0) {
      return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "not_meaningful", reason: "non-positive expected price" }
    }
    const percentChange = currentPrice != null ? (expectedPrice / currentPrice - 1) * 100 : null
    return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice, percentChange, status: "ok", reason: null }
  }
  if (shares == null) return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "missing", reason: "missing shares" }

  const grossValue = multiple * (denominatorConverted ?? 0)
  const equityValue = ENTERPRISE_VALUE_RANGE_METHODS.has(metric) ? grossValue - (netDebt ?? 0) : grossValue
  if (!Number.isFinite(equityValue) || equityValue <= 0) {
    return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice: null, percentChange: null, status: "not_meaningful", reason: "non-positive equity value" }
  }
  const expectedPrice = equityValue / shares
  const percentChange = currentPrice != null ? (expectedPrice / currentPrice - 1) * 100 : null
  return { multiple, terminalGrowth, wacc, denominator, denominatorConverted, expectedPrice, percentChange, status: "ok", reason: null }
}

function valueRangeRequestFromDraft(
  metric: ValueRangeMetricKey,
  draft: ValueRangeDraft["scenarios"],
  denominatorCurrency: string | null,
  waccDraft: string,
): PositionValueRangeRequest {
  const scenarios: PositionValueRangeRequest["scenarios"] = {}
  const wacc = isDcfGordonGrowthMetric(metric) ? parseWaccInput(waccDraft) : null
  if (isDcfGordonGrowthMetric(metric) && wacc == null) {
    throw new Error("DCF (Gordon Growth) requires WACC greater than 0% and below 100%.")
  }
  for (const scenario of VALUE_RANGE_SCENARIOS) {
    const row = draft[scenario]
    const denominator = parseScaledNumberInput(row.denominator)
    if (isDcfGordonGrowthMetric(metric)) {
      const terminalGrowth = parseTerminalGrowthInput(row.multiple)
      if (terminalGrowth == null || denominator == null) {
        throw new Error(`${VALUE_RANGE_SCENARIO_LABELS[scenario]} requires terminal growth above -100% and a positive denominator.`)
      }
      scenarios[scenario] = { terminal_growth: terminalGrowth, denominator }
    } else {
      const multiple = parseMultipleInput(row.multiple)
      if (multiple == null || denominator == null) {
        throw new Error(`${VALUE_RANGE_SCENARIO_LABELS[scenario]} requires a positive multiple and denominator.`)
      }
      scenarios[scenario] = { multiple, denominator }
    }
  }
  return { metric, denominator_currency: denominatorCurrency, ...(wacc != null ? { wacc } : {}), scenarios }
}

function valueRangeDraftFromRequest(payload: PositionValueRangeRequest): ValueRangeDraft["scenarios"] {
  const metric = isValueRangeMetricKey(payload.metric) ? payload.metric : "price_sales"
  return VALUE_RANGE_SCENARIOS.reduce((acc, scenario) => {
    const row = payload.scenarios[scenario]
    acc[scenario] = {
      multiple: isDcfGordonGrowthMetric(metric) ? formatPercentInput(row?.terminal_growth) : formatMultipleInput(row?.multiple),
      denominator: formatInputNumber(row?.denominator),
    }
    return acc
  }, {} as ValueRangeDraft["scenarios"])
}

function valueRangeDraftsMatch(a: ValueRangeDraft["scenarios"], b: ValueRangeDraft["scenarios"]): boolean {
  return VALUE_RANGE_SCENARIOS.every(
    scenario =>
      a[scenario].multiple === b[scenario].multiple &&
      a[scenario].denominator === b[scenario].denominator,
  )
}

function valueRangeDraftHasAnyValue(draft: ValueRangeDraft["scenarios"]): boolean {
  return VALUE_RANGE_SCENARIOS.some(scenario => draft[scenario].multiple.trim() || draft[scenario].denominator.trim())
}

function savedWaccDraft(assumption: PositionValueRangeAssumption | null): string {
  return formatPercentInput(assumption?.wacc)
}

function valuationStatusClass(status?: string | null): string {
  if (status === "ok") return "border-green-200 bg-green-50 text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-300"
  if (status === "degraded") return "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
  return "border-app bg-[hsl(var(--muted-2))] text-muted"
}

function ValueRangePanel({
  valuation,
  isSaving,
  isClearing,
  saveError,
  clearError,
  onSave,
  onClear,
}: {
  valuation: PositionValuation
  isSaving: boolean
  isClearing: boolean
  saveError: unknown
  clearError: unknown
  onSave: (payload: PositionValueRangeRequest) => void
  onClear: (metric: ValueRangeMetricKey) => void
}) {
  const [activeMetric, setActiveMetric] = useState<ValueRangeMetricKey>(() => valueRangeSelectedMetric(valuation))
  const [drafts, setDrafts] = useState<ValueRangeDraftMap>(() => valueRangeInitialDrafts(valuation))
  const [waccDrafts, setWaccDrafts] = useState<Partial<Record<ValueRangeMetricKey, string>>>(() => valueRangeInitialWaccDrafts(valuation))
  const [validationError, setValidationError] = useState<string | null>(null)

  function updateScenario(scenario: ValueRangeScenarioKey, patch: Partial<ValueRangeDraft["scenarios"][ValueRangeScenarioKey]>) {
    setValidationError(null)
    setDrafts(prev => {
      const current = prev[activeMetric] ?? blankValueRangeScenarioDrafts()
      return {
        ...prev,
        [activeMetric]: {
          ...current,
          [scenario]: { ...current[scenario], ...patch },
        },
      }
    })
  }

  function handleMetricChange(value: string) {
    if (!isValueRangeMetricKey(value)) return
    setValidationError(null)
    setDrafts(prev => (prev[value] ? prev : { ...prev, [value]: blankValueRangeScenarioDrafts() }))
    setWaccDrafts(prev => (prev[value] != null ? prev : { ...prev, [value]: isDcfGordonGrowthMetric(value) ? "10" : "" }))
    setActiveMetric(value)
  }

  function handleWaccChange(value: string) {
    setValidationError(null)
    setWaccDrafts(prev => ({ ...prev, [activeMetric]: value }))
  }

  function handleSave() {
    try {
      setValidationError(null)
      const payload = valueRangeRequestFromDraft(activeMetric, activeDraft, activeContext.denominatorCurrency, activeWaccDraft)
      setDrafts(prev => ({ ...prev, [activeMetric]: valueRangeDraftFromRequest(payload) }))
      setWaccDrafts(prev => ({ ...prev, [activeMetric]: formatPercentInput(payload.wacc) }))
      onSave(payload)
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : "Invalid value range.")
    }
  }

  function handleClear() {
    const fallbackMetric = VALUE_RANGE_METHOD_ORDER.find(metric => metric !== activeMetric && savedAssumptions[metric]) ?? valueRangeDefaultMetric(valuation)
    setValidationError(null)
    setDrafts(prev => ({ ...prev, [activeMetric]: blankValueRangeScenarioDrafts() }))
    setWaccDrafts(prev => ({ ...prev, [activeMetric]: isDcfGordonGrowthMetric(activeMetric) ? "10" : "" }))
    setActiveMetric(fallbackMetric)
    onClear(activeMetric)
  }

  const savedAssumptions = valueRangeAssumptions(valuation)
  const activeAssumption = savedAssumptions[activeMetric] ?? null
  const activeDraft = drafts[activeMetric] ?? blankValueRangeScenarioDrafts()
  const activeSavedDraft = activeAssumption ? valueRangeDraftFromAssumption(valuation, activeMetric, activeAssumption) : null
  const activeWaccDraft = waccDrafts[activeMetric] ?? (isDcfGordonGrowthMetric(activeMetric) ? "10" : "")
  const activeContext = valueRangeDisplayContext(valuation, activeAssumption)
  const hasSavedMetric = Boolean(activeAssumption)
  const metricLabel = valueRangeMetricLabel(valuation, activeMetric)
  const outputCurrency = valueRangeOutputCurrency(valuation)
  const denominatorCurrency = activeContext.denominatorCurrency
  const currentPrice = formatSharePrice(valuation.market_data?.current_price, outputCurrency)
  const saveErrorText = saveError instanceof Error ? saveError.message : saveError ? String(saveError) : null
  const clearErrorText = clearError instanceof Error ? clearError.message : clearError ? String(clearError) : null
  const hasUnsavedChanges = activeSavedDraft
    ? !valueRangeDraftsMatch(activeDraft, activeSavedDraft) || (isDcfGordonGrowthMetric(activeMetric) && activeWaccDraft !== savedWaccDraft(activeAssumption))
    : valueRangeDraftHasAnyValue(activeDraft)

  return (
    <section className="space-y-3">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-app">Value Range</h3>
            <span className="rounded border border-app px-2 py-0.5 text-xs font-semibold text-muted">
              {hasSavedMetric ? "saved" : "blank"}
            </span>
          </div>
          <p className="mt-1 text-xs text-muted">
            {metricLabel} scenarios from user assumptions. Current price {currentPrice}.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(220px,1fr)_auto_auto] lg:w-[520px]">
          <SelectInput
            label="Metric"
            value={activeMetric}
            onChange={handleMetricChange}
            options={VALUE_RANGE_METHOD_ORDER.map(metric => ({
              value: metric,
              label: `${savedAssumptions[metric] ? "• " : ""}${valueRangeMetricLabel(valuation, metric)}`,
            }))}
          />
          <button
            type="button"
            onClick={handleClear}
            disabled={!hasSavedMetric || isSaving || isClearing}
            className="theme-button-base theme-button-secondary min-h-10 px-4 text-sm disabled:pointer-events-none disabled:opacity-50 sm:self-end"
          >
            {isClearing ? "Clearing..." : "Clear"}
          </button>
          <ActionButton
            onClick={handleSave}
            disabled={!hasUnsavedChanges || isClearing}
            loading={isSaving}
            loadingText="Saving..."
            className="min-h-10 px-4 sm:self-end sm:w-auto"
          >
            Save
          </ActionButton>
        </div>
      </div>

      {isDcfGordonGrowthMetric(activeMetric) && (
        <div className="max-w-xs">
          <TextInput
            label="WACC (%)"
            value={activeWaccDraft}
            onChange={handleWaccChange}
            placeholder="10"
          />
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {VALUE_RANGE_SCENARIOS.map(scenario => {
          const computed = computeDraftValueRangeScenario(valuation, activeMetric, activeDraft[scenario], activeContext.denominatorToPriceRate, activeWaccDraft)
          return (
            <article key={scenario} className="rounded-lg border border-app bg-card px-3 py-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="text-sm font-semibold text-app">{VALUE_RANGE_SCENARIO_LABELS[scenario]}</h4>
                  <p className="text-xs text-subtle">{metricLabel}</p>
                </div>
                <span className={cn("rounded border px-2 py-0.5 text-xs font-semibold", valuationStatusClass(computed.status))}>
                  {computed.status.replace(/_/g, " ")}
                </span>
              </div>
              <p className="mt-3 text-2xl font-semibold text-app">{formatSharePrice(computed.expectedPrice, outputCurrency)}</p>
              <p className={cn("mt-0.5 text-xs font-medium", scenarioChangeClass(computed.percentChange))}>
                {formatScenarioChange(computed.percentChange)} from current
              </p>
              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                <TextInput
                  label={valueRangeFirstInputLabel(activeMetric)}
                  value={activeDraft[scenario].multiple}
                  onChange={value => updateScenario(scenario, { multiple: value })}
                  placeholder={isDcfGordonGrowthMetric(activeMetric) ? "3" : "10x"}
                />
                <TextInput
                  label={valueRangeDenominatorLabel(activeMetric, denominatorCurrency)}
                  value={activeDraft[scenario].denominator}
                  onChange={value => updateScenario(scenario, { denominator: value })}
                  placeholder={isValuationMetricKey(activeMetric) && PER_SHARE_VALUE_RANGE_METRICS.has(activeMetric) ? "5.25" : "1.5B"}
                />
              </div>
              {computed.reason && <p className="mt-2 text-xs text-subtle">{computed.reason}</p>}
            </article>
          )
        })}
      </div>

      {(validationError || saveErrorText || clearErrorText) && (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {validationError || saveErrorText || clearErrorText}
        </p>
      )}
    </section>
  )
}

interface ValuationFootballFieldRow {
  id: string
  label: string
  subtitle: string
  low: number
  high: number
  base: number | null
  tone: "valuation" | "reference"
}

function scenarioExpectedPrice(
  scenarios: Record<string, PositionValueRangeScenario> | undefined,
  scenario: ValueRangeScenarioKey,
): number | null {
  return positiveNumber(scenarios?.[scenario]?.expected_price)
}

function valuationFootballFieldRows(data: PositionValuation): {
  rows: ValuationFootballFieldRow[]
  savedMetricCount: number
  skippedMetricCount: number
} {
  const assumptions = valueRangeAssumptions(data)
  const rows: ValuationFootballFieldRow[] = []
  let savedMetricCount = 0
  let skippedMetricCount = 0

  for (const metric of VALUE_RANGE_METHOD_ORDER) {
    const assumption = assumptions[metric]
    if (!assumption) continue
    savedMetricCount += 1
    const scenarios = assumption.computed_scenarios
    const bear = scenarioExpectedPrice(scenarios, "bear")
    const base = scenarioExpectedPrice(scenarios, "base")
    const bull = scenarioExpectedPrice(scenarios, "bull")
    if (bear == null || base == null || bull == null) {
      skippedMetricCount += 1
      continue
    }
    rows.push({
      id: metric,
      label: valueRangeMetricLabel(data, metric),
      subtitle: isValuationMetricKey(metric) ? data.metrics[metric]?.denominator_label ?? data.value_range?.denominator_label ?? "Saved range" : valueRangeDenominatorLabel(metric, null),
      low: Math.min(bear, base, bull),
      high: Math.max(bear, base, bull),
      base,
      tone: "valuation",
    })
  }

  const weekLow = positiveNumber(data.market_data?.fifty_two_week_low)
  const weekHigh = positiveNumber(data.market_data?.fifty_two_week_high)
  if (savedMetricCount > 0 && weekLow != null && weekHigh != null) {
    rows.push({
      id: "52-week",
      label: "52W High/Low",
      subtitle: "Market reference",
      low: Math.min(weekLow, weekHigh),
      high: Math.max(weekLow, weekHigh),
      base: null,
      tone: "reference",
    })
  }

  return { rows, savedMetricCount, skippedMetricCount }
}

function ValuationFootballField({ valuation }: { valuation: PositionValuation }) {
  const { rows, savedMetricCount, skippedMetricCount } = valuationFootballFieldRows(valuation)
  const outputCurrency = valueRangeOutputCurrency(valuation)
  const currentPrice = positiveNumber(valuation.market_data?.current_price)

  if (savedMetricCount === 0) {
    return (
      <section className="rounded-lg border border-app px-4 py-5">
        <h3 className="text-sm font-semibold text-app">Football Field</h3>
        <p className="mt-1 text-sm text-muted">Save at least one metric range to view the football field.</p>
      </section>
    )
  }

  if (rows.length === 0) {
    return (
      <section className="rounded-lg border border-app px-4 py-5">
        <h3 className="text-sm font-semibold text-app">Football Field</h3>
        <p className="mt-1 text-sm text-muted">No saved ranges can be plotted with the current valuation inputs.</p>
      </section>
    )
  }

  const values = rows.flatMap(row => [row.low, row.high, ...(row.base != null ? [row.base] : [])])
  if (currentPrice != null) values.push(currentPrice)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const span = maxValue - minValue
  const padding = span > 0 ? span * 0.06 : Math.max(maxValue * 0.1, 1)
  const axisMin = Math.max(0, minValue - padding)
  const axisMax = maxValue + padding
  const axisSpan = axisMax - axisMin || 1
  const positionPct = (value: number) => Math.max(0, Math.min(100, ((value - axisMin) / axisSpan) * 100))
  const currentPct = currentPrice != null ? positionPct(currentPrice) : null

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-app">Football Field</h3>
          <p className="mt-1 text-xs text-muted">{savedMetricCount} saved metric{savedMetricCount === 1 ? "" : "s"}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-5 rounded-full bg-[hsl(var(--accent))]" />
            Valuation range
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-4 w-px bg-app" />
            Base
          </span>
          {currentPct != null && (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-4 w-px border-l border-dashed border-[hsl(var(--foreground-tertiary))]" />
              Current
            </span>
          )}
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-app">
        <div className="min-w-[680px] px-4 py-3">
          <div className="grid grid-cols-[11rem_minmax(0,1fr)] gap-4 border-b border-app pb-2 text-xs text-subtle">
            <span>Metric</span>
            <div className="flex justify-between">
              <span>{formatSharePrice(axisMin, outputCurrency)}</span>
              {currentPrice != null && <span>Current {formatSharePrice(currentPrice, outputCurrency)}</span>}
              <span>{formatSharePrice(axisMax, outputCurrency)}</span>
            </div>
          </div>
          <div className="divide-y divide-[hsl(var(--border))]">
            {rows.map(row => {
              const left = positionPct(row.low)
              const right = positionPct(row.high)
              const width = Math.max(right - left, 0.7)
              const basePct = row.base != null ? positionPct(row.base) : null
              return (
                <div key={row.id} className="grid grid-cols-[11rem_minmax(0,1fr)] gap-4 py-3">
                  <div>
                    <p className="text-sm font-medium text-app">{row.label}</p>
                    <p className="text-xs text-subtle">{row.subtitle}</p>
                  </div>
                  <div className="space-y-1.5">
                    <div className="relative h-8">
                      <div className="absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-[hsl(var(--border))]" />
                      <div
                        className={cn(
                          "absolute top-1/2 h-3 -translate-y-1/2 rounded-full",
                          row.tone === "reference" ? "bg-[hsl(var(--foreground-quaternary))]" : "bg-[hsl(var(--accent))]",
                        )}
                        style={{ left: `${left}%`, width: `${width}%` }}
                      />
                      {basePct != null && (
                        <div
                          className="absolute top-1/2 h-7 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[hsl(var(--foreground))]"
                          style={{ left: `${basePct}%` }}
                          title={`Base ${formatSharePrice(row.base, outputCurrency)}`}
                        />
                      )}
                      {currentPct != null && (
                        <div
                          className="absolute top-1/2 h-8 -translate-x-1/2 -translate-y-1/2 border-l border-dashed border-[hsl(var(--foreground-tertiary))]"
                          style={{ left: `${currentPct}%` }}
                          title={`Current ${formatSharePrice(currentPrice, outputCurrency)}`}
                        />
                      )}
                    </div>
                    <div className="flex justify-between text-xs text-subtle">
                      <span>{formatSharePrice(row.low, outputCurrency)}</span>
                      {row.base != null && <span>Base {formatSharePrice(row.base, outputCurrency)}</span>}
                      <span>{formatSharePrice(row.high, outputCurrency)}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {skippedMetricCount > 0 && (
        <p className="text-xs text-muted">
          {skippedMetricCount} saved metric{skippedMetricCount === 1 ? "" : "s"} could not be plotted because one or more scenarios did not compute.
        </p>
      )}
    </section>
  )
}

export function PositionValuationTab({ ticker }: { ticker: string }) {
  const qc = useQueryClient()
  const [valuationView, setValuationView] = useState<ValuationTabView>("inputs")
  const { data, isLoading, error } = useApiQuery<PositionValuation>(
    ["valuation", ticker],
    () => fetchPositionValuation(ticker),
    300_000,
  )

  const profileMutation = useMutation({
    mutationFn: (profileId: string | null) => updatePositionValuationProfileOverride(ticker, profileId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["valuation", ticker] })
    },
  })

  const valueRangeMutation = useMutation({
    mutationFn: (body: PositionValueRangeRequest) => updatePositionValueRange(ticker, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["valuation", ticker] })
    },
  })

  const clearValueRangeMutation = useMutation({
    mutationFn: (metric: ValueRangeMetricKey) => deletePositionValueRange(ticker, metric),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["valuation", ticker] })
    },
  })

  if (isLoading) return <LoadingSpinner message="Loading valuation..." />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const profileValue = data.profile.override_profile_id ?? "auto"
  const profileOptions = [
    { value: "auto", label: `Auto (${data.profile.label})` },
    ...data.profile.options.map(option => ({ value: option.id, label: option.label })),
  ]
  const warnings = data.data_quality?.warnings ?? []
  const priceCcy = priceCurrency(data)
  const financialCcy = financialCurrency(data)
  const mixedCurrencies = Boolean(priceCcy && financialCcy && !sameCurrency(priceCcy, financialCcy))
  const fxRate = positiveNumber(data.currency_context?.financial_to_price_fx_rate)

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold text-app">Valuation</h2>
            <span className={cn("rounded border px-2 py-0.5 text-xs font-semibold", valuationStatusClass(data.data_quality?.status))}>
              {data.data_quality?.status?.replace(/_/g, " ") ?? "unknown"}
            </span>
            {priceCcy && <span className="rounded border border-app px-2 py-0.5 text-xs font-semibold text-muted">Price: {priceCcy}</span>}
            {financialCcy && <span className="rounded border border-app px-2 py-0.5 text-xs font-semibold text-muted">Financials: {financialCcy}</span>}
            {mixedCurrencies && (
              <span className="rounded border border-app px-2 py-0.5 text-xs font-semibold text-muted">
                FX: {financialCcy} -&gt; {priceCcy}{fxRate != null ? ` ${fxRate.toFixed(4)}` : ""}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-muted">
            {data.company_name || data.ticker}
            {data.market_data?.sector ? ` - ${data.market_data.sector}` : ""}
            {data.market_data?.industry ? ` / ${data.market_data.industry}` : ""}
          </p>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:w-[420px]">
          <div className="rounded-lg border border-app px-3 py-2">
            <p className="text-xs uppercase text-subtle">Composite</p>
            <p className="mt-1 text-xl font-semibold text-app">{formatValuationPct(data.composite_score?.value)}</p>
            <p className="text-xs text-muted">{data.peer_context.peer_count} peers - {data.peer_context.source.replace(/_/g, " ")}</p>
          </div>
          <SelectInput
            label="Profile"
            value={profileValue}
            onChange={value => profileMutation.mutate(value === "auto" ? null : value)}
            options={profileOptions}
            disabled={profileMutation.isPending}
            helperText={profileMutation.isPending ? "Saving profile override..." : data.profile.selection_mode === "override" ? "Manual profile override" : "Auto profile"}
          />
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
          {warnings.join(" ")}
        </div>
      )}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-app">Scenario View</h3>
          <SegmentedControl<ValuationTabView>
            value={valuationView}
            onChange={setValuationView}
            size="sm"
            options={[
              { value: "inputs", label: "Inputs" },
              { value: "football_field", label: "Football Field" },
            ]}
          />
        </div>
        {valuationView === "inputs" ? (
          <ValueRangePanel
            key={`${data.ticker}-${data.profile.override_profile_id ?? "auto"}`}
            valuation={data}
            isSaving={valueRangeMutation.isPending}
            isClearing={clearValueRangeMutation.isPending}
            saveError={valueRangeMutation.error}
            clearError={clearValueRangeMutation.error}
            onSave={payload => valueRangeMutation.mutate(payload)}
            onClear={metric => clearValueRangeMutation.mutate(metric)}
          />
        ) : (
          <ValuationFootballField valuation={data} />
        )}
      </section>

      <section className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-app">Profile Weights</h3>
          <p className="text-xs text-muted">{data.profile.rationale}</p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
          {VALUATION_METRIC_ORDER.map(key => (
            <div key={key} className="rounded-lg border border-app px-3 py-2">
              <p className="text-xs text-subtle">{data.metrics[key]?.label ?? key}</p>
              <p className="text-sm font-semibold text-app">{formatWeight(data.profile.effective_weights[key])}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="overflow-x-auto rounded-lg border border-app">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-app text-xs uppercase text-subtle">
            <tr>
              <th className="px-3 py-2 font-semibold">Metric</th>
              <th className="px-3 py-2 font-semibold">Multiple</th>
              <th className="px-3 py-2 font-semibold">Denominator</th>
              <th className="px-3 py-2 font-semibold">Peer Percentile</th>
              <th className="px-3 py-2 font-semibold">Peer Median</th>
              <th className="px-3 py-2 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[hsl(var(--border))]">
            {VALUATION_METRIC_ORDER.map(key => {
              const metric = data.metrics[key]
              const peer = data.peer_context.metric_stats[key]
              return (
                <tr key={key}>
                  <td className="px-3 py-2">
                    <div className="font-medium text-app">{metric?.label ?? key}</div>
                    <div className="text-xs text-subtle">{metric?.period ?? ""}</div>
                  </td>
                  <td className="px-3 py-2 font-mono text-app">{formatMultipleValue(metric?.value)}</td>
                  <td className="px-3 py-2">
                    <div className="text-app">{formatValuationDenominator(key, metric?.denominator, metric?.denominator_currency ?? financialCcy)}</div>
                    {metric?.denominator_converted != null && !sameCurrency(metric.denominator_currency, metric.denominator_converted_currency) && (
                      <div className="text-xs text-subtle">
                        {formatValuationDenominator(key, metric.denominator_converted, metric.denominator_converted_currency ?? priceCcy)}
                      </div>
                    )}
                    <div className="text-xs text-subtle">{metric?.denominator_label}</div>
                  </td>
                  <td className="px-3 py-2 text-app">{formatValuationPct(peer?.percentile)}</td>
                  <td className="px-3 py-2 text-app">{formatMultipleValue(peer?.median)}</td>
                  <td className="px-3 py-2">
                    <span className={cn("inline-flex rounded border px-2 py-0.5 text-xs font-semibold", valuationStatusClass(metric?.status))}>
                      {(metric?.status ?? "missing").replace(/_/g, " ")}
                    </span>
                    {metric?.reason && <div className="mt-1 text-xs text-subtle">{metric.reason.replace(/_/g, " ")}</div>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-app px-3 py-2">
          <h3 className="text-sm font-semibold text-app">Peer Set</h3>
          <p className="mt-1 text-sm text-muted">
            {data.peer_context.peer_count > 0 ? data.peer_context.peers.slice(0, 18).join(", ") : "No peer set available."}
            {data.peer_context.peer_count > 18 ? "..." : ""}
          </p>
        </div>
        <div className="rounded-lg border border-app px-3 py-2">
          <h3 className="text-sm font-semibold text-app">Market Data</h3>
          <p className="mt-1 text-sm text-muted">
            EV {formatValuationMoney(data.market_data?.enterprise_value, priceCcy)}
            {" - "}
            Market cap {formatValuationMoney(data.market_data?.market_cap, priceCcy)}
            {data.market_data?.current_price ? ` - Price ${formatSharePrice(data.market_data.current_price, priceCcy)}` : ""}
          </p>
        </div>
      </div>
    </div>
  )
}
