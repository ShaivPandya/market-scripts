import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ChevronDown, Layers, Plus, Trash2, Upload } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput } from "@/components/shared/FormControls"
import { StagedProposalNotice } from "@/components/shared/StagedProposalNotice"
import {
  fetchHedgePositions,
  fetchPortfolioSettings,
  fetchPortfolioPositions,
  importIbkrFlexPortfolioPositions,
  saveHedgePositions,
  savePortfolioPositions,
  updatePortfolioSettings,
  type HedgePosition,
  type PortfolioPosition,
  type IbkrFlexImportResponse,
  type StagedMutationResponse,
} from "@/lib/api"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"
import {
  ASSET_OPTIONS,
  INSTRUMENT_TYPE_OPTIONS,
  OPTION_TYPE_OPTIONS,
  applyOptionPaste,
  assetLabel,
  buildOptionContractSymbol,
  canonicalSpotFxSymbol,
  displayTicker,
  effectivePriceSymbol,
  exposureDirection,
  inferInstrumentType,
  instrumentTypeLabel,
  nextContractMultiplier,
  normalizedSymbol,
  positionRowId,
  spotFxCurrencies,
} from "@/lib/instruments"
import { groupKey, normalizeGroupConviction, normalizeGroupName } from "@/lib/positionGroups"

interface EditorRow extends PortfolioPosition {
  _id: string
  _isNew: boolean
  _contractMultiplierTouched: boolean
}

interface HedgeEditorRow extends HedgePosition {
  _id: string
  _contractMultiplierTouched: boolean
}

type AnyRow = EditorRow | HedgeEditorRow
type EditorTab = "Positions" | "Hedges"
type InstrumentType = NonNullable<PortfolioPosition["instrument_type"]>

const DIRECTION_OPTIONS = [
  { value: "long", label: "Long" },
  { value: "short", label: "Short" },
]

const MIN_BOOK_SIZE = 10_000
const MAX_BOOK_SIZE = 10_000_000
const DEFAULT_BOOK_SIZE = 100_000
const SIZER_STATE_QUERY_KEY = ["portfolio-sizer", "state"] as const

const CONVICTION_LABELS: Record<number, string> = {
  1: "Very Low",
  2: "Low",
  3: "Medium",
  4: "High",
  5: "Very High",
}

/* ── Visual ramps ──────────────────────────────────────────────────────── */
const CONV_HSL: Record<number, string> = {
  1: "215 16% 52%",
  2: "212 48% 52%",
  3: "211 78% 52%",
  4: "224 74% 57%",
  5: "250 72% 61%",
}
const convColor = (c: number) => `hsl(${CONV_HSL[c] ?? CONV_HSL[3]})`
const convTint = (c: number, a = 0.14) => `hsl(${CONV_HSL[c] ?? CONV_HSL[3]} / ${a})`

const GROUP_HUES = [248, 30, 168, 211, 330, 96, 280, 12]
function groupHue(key: string) {
  let hash = 0
  for (let i = 0; i < key.length; i += 1) hash = (hash * 31 + key.charCodeAt(i)) >>> 0
  return GROUP_HUES[hash % GROUP_HUES.length]
}
const groupColor = (key: string) => `hsl(${groupHue(key)} 64% 55%)`
const groupTint = (key: string, a = 0.12) => `hsl(${groupHue(key)} 64% 55% / ${a})`

/* ── Formatters ────────────────────────────────────────────────────────── */
const fmtUSD0 = (v: number) =>
  (v < 0 ? "−" : "") +
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(Math.abs(v))
const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`
const fmtSignedPct = (v: number) => `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(1)}%`

function makeId() {
  return Math.random().toString(36).slice(2, 10)
}

function proposalSubjectLabel(entityType?: string | null): string {
  return String(entityType || "proposal").replace(/_/g, " ")
}

function rowQuantity(row: { quantity?: number | null; shares?: number | null }) {
  return row.quantity ?? row.shares ?? null
}

function formatBaseCurrency(value: number, currency?: string | null) {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: 0,
    }).format(value)
  } catch {
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value)
  }
}

function parseBookSizeInput(value: string) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.round(parsed * 100) / 100 : null
}

function rowInstrumentType(row: { ticker: string; instrument_type?: InstrumentType | null }) {
  return inferInstrumentType(row.ticker, row.instrument_type)
}

function rowMultiplier(row: AnyRow) {
  const type = rowInstrumentType(row)
  if (type === "future") return row.contract_multiplier ?? 1
  if (type === "option") return row.contract_multiplier ?? 100
  return 1
}

/* Notional uses entry cost (cost_basis) as the best locally-known price. */
function grossNotional(row: AnyRow) {
  const qty = rowQuantity(row) ?? 0
  const price = row.cost_basis ?? 0
  return Math.abs(qty * price * rowMultiplier(row))
}
function netNotional(row: AnyRow) {
  return grossNotional(row) * (exposureDirection(row) === "short" ? -1 : 1)
}

interface BookSummary {
  gross: number
  net: number
  longN: number
  shortN: number
  byAsset: Record<string, number>
  leverage: number
  netPct: number
}

function summarize(rows: AnyRow[], bookSize: number): BookSummary {
  let gross = 0
  let net = 0
  let longN = 0
  let shortN = 0
  const byAsset: Record<string, number> = {}
  for (const row of rows) {
    const g = grossNotional(row)
    const n = netNotional(row)
    gross += g
    net += n
    if (n >= 0) longN += g
    else shortN += g
    const asset = row.asset ?? "equity"
    byAsset[asset] = (byAsset[asset] ?? 0) + n
  }
  return { gross, net, longN, shortN, byAsset, leverage: bookSize > 0 ? gross / bookSize : 0, netPct: bookSize > 0 ? net / bookSize : 0 }
}

function optionContractSymbolForRow(row: {
  ticker: string
  underlying_ticker?: string | null
  option_contract_symbol?: string | null
  option_expiration?: string | null
  option_strike?: number | null
  option_type?: PortfolioPosition["option_type"] | null
}) {
  const underlying = normalizedSymbol(row.underlying_ticker || row.ticker)
  if (row.option_contract_symbol?.trim()) return normalizedSymbol(row.option_contract_symbol)
  if (!underlying || row.option_expiration == null || row.option_strike == null || !row.option_type) return null
  return buildOptionContractSymbol(underlying, row.option_expiration, row.option_type, row.option_strike)
}

function serializeInstrumentRow<T extends PortfolioPosition | HedgePosition>(
  row: EditorRow | HedgeEditorRow,
  extras?: Partial<PortfolioPosition>,
): T {
  const instrumentType = rowInstrumentType(row)
  const quantity = rowQuantity(row)

  if (instrumentType === "option") {
    const underlying = normalizedSymbol(row.underlying_ticker || row.ticker)
    const contractSymbol = optionContractSymbolForRow(row)
    if (!underlying || !contractSymbol) {
      throw new Error("Option rows require underlying, expiration, strike, type, or a valid OCC contract symbol.")
    }
    const positionId = positionRowId({
      ticker: underlying,
      position_id: row.position_id,
      option_contract_symbol: contractSymbol,
      price_symbol: contractSymbol,
      instrument_type: "option",
    })
    return {
      ticker: underlying,
      asset: row.asset ?? "equity",
      direction: row.direction,
      cost_basis: row.cost_basis,
      shares: quantity,
      quantity,
      instrument_type: "option",
      price_symbol: contractSymbol,
      contract_multiplier: row.contract_multiplier ?? 100,
      position_id: positionId,
      underlying_ticker: underlying,
      option_contract_symbol: contractSymbol,
      option_expiration: row.option_expiration ?? null,
      option_strike: row.option_strike ?? null,
      option_type: row.option_type ?? null,
      currency: row.currency ?? null,
      country: row.country ?? null,
      exchange: row.exchange ?? null,
      base_currency: row.base_currency ?? null,
      fx_rate_to_base: row.fx_rate_to_base ?? null,
      fx_rate_as_of: row.fx_rate_as_of ?? null,
      cost_basis_base: row.cost_basis_base ?? null,
      notional_base: row.notional_base ?? null,
      valuation_status: row.valuation_status ?? null,
      ...extras,
    } as T
  }

  const ticker = instrumentType === "spot_fx"
    ? canonicalSpotFxSymbol(row.price_symbol || row.ticker) ?? row.ticker.trim().toUpperCase()
    : row.ticker.trim().toUpperCase()
  const priceSymbol = instrumentType === "spot_fx" ? ticker : (row.price_symbol?.trim() || row.ticker).toUpperCase()
  const fxCurrencies = instrumentType === "spot_fx"
    ? spotFxCurrencies(priceSymbol)
    : { fx_base_currency: row.fx_base_currency ?? null, fx_quote_currency: row.fx_quote_currency ?? null }

  return {
    ticker,
    asset: instrumentType === "spot_fx" ? "fx" : row.asset ?? "equity",
    direction: row.direction,
    cost_basis: row.cost_basis,
    shares: quantity,
    quantity,
    instrument_type: instrumentType,
    price_symbol: priceSymbol,
    contract_multiplier: instrumentType === "future" ? row.contract_multiplier ?? null : 1,
    position_id: positionRowId({ ticker, instrument_type: instrumentType }),
    fx_base_currency: fxCurrencies.fx_base_currency,
    fx_quote_currency: fxCurrencies.fx_quote_currency,
    currency: instrumentType === "spot_fx" ? fxCurrencies.fx_quote_currency : row.currency ?? null,
    country: row.country ?? null,
    exchange: instrumentType === "spot_fx" ? row.exchange ?? "FX" : row.exchange ?? null,
    base_currency: row.base_currency ?? null,
    fx_rate_to_base: row.fx_rate_to_base ?? null,
    fx_rate_as_of: row.fx_rate_as_of ?? null,
    cost_basis_base: row.cost_basis_base ?? null,
    notional_base: row.notional_base ?? null,
    valuation_status: row.valuation_status ?? null,
    ...extras,
  } as T
}

/* Backend-fetched descriptor shown read-only under the ticker (only when it adds info). */
function valuationSummary(row: PortfolioPosition | HedgePosition) {
  const parts: string[] = []
  const market = [row.country, row.exchange].filter(Boolean).join(" / ")
  if (row.instrument_type === "option" && row.option_contract_symbol) {
    parts.push(row.option_contract_symbol)
    if (row.option_type && row.option_strike != null && row.option_expiration) {
      parts.push(`${String(row.option_type).toUpperCase()} ${row.option_strike} exp ${row.option_expiration}`)
    }
  }
  if (row.instrument_type === "spot_fx" && row.fx_base_currency && row.fx_quote_currency) {
    parts.push(`${row.fx_base_currency}/${row.fx_quote_currency} spot`)
  }
  if (market) parts.push(market)
  if (row.currency) parts.push(`${row.currency}${row.base_currency ? ` to ${row.base_currency}` : ""}`)
  if (typeof row.notional_base === "number" && Number.isFinite(row.notional_base)) {
    parts.push(`${formatBaseCurrency(row.notional_base, row.base_currency)} base notional`)
  }
  if (row.valuation_status && row.valuation_status !== "ok") {
    parts.push(row.valuation_status.replace(/_/g, " "))
  }
  return parts.join(" · ")
}

interface GroupState {
  key: string
  name: string
  conviction: number
  direction: PortfolioPosition["direction"]
  ids: string[]
  tickers: string[]
}

interface UnderlyingCluster {
  key: string
  ticker: string
  legs: EditorRow[]
  conviction: number
  groupName: string | null
  direction: PortfolioPosition["direction"]
  gross: number
  net: number
}

function underlyingClusterKey(row: EditorRow) {
  const ticker = displayTicker(row) || normalizedSymbol(row.ticker)
  return ticker || row._id
}

function clusterConviction(legs: EditorRow[]) {
  if (legs.length === 0) return 3
  return Math.max(...legs.map(leg => leg.conviction ?? 3))
}

function clusterGroupName(legs: EditorRow[]) {
  for (const leg of legs) {
    const name = normalizeGroupName(leg.group_name)
    if (name) return name
  }
  return null
}

function inferClusterDirection(legs: EditorRow[]): PortfolioPosition["direction"] {
  const securityLeg = legs.find(leg => rowInstrumentType(leg) !== "option")
  if (securityLeg) {
    return exposureDirection(securityLeg) ?? securityLeg.direction
  }
  const net = legs.reduce((sum, leg) => sum + netNotional(leg), 0)
  if (Math.abs(net) < 1e-6) return legs[0]?.direction ?? "long"
  return net >= 0 ? "long" : "short"
}

function buildUnderlyingClusters(rows: EditorRow[]): UnderlyingCluster[] {
  const map = new Map<string, EditorRow[]>()
  const order: string[] = []
  for (const row of rows) {
    const key = underlyingClusterKey(row)
    if (!map.has(key)) {
      map.set(key, [])
      order.push(key)
    }
    map.get(key)?.push(row)
  }
  return order.map(key => {
    const legs = map.get(key) ?? []
    const ticker = displayTicker(legs[0]) || legs[0]?.ticker || "New position"
    return {
      key,
      ticker,
      legs,
      conviction: clusterConviction(legs),
      groupName: clusterGroupName(legs),
      direction: inferClusterDirection(legs),
      gross: legs.reduce((sum, leg) => sum + grossNotional(leg), 0),
      net: legs.reduce((sum, leg) => sum + netNotional(leg), 0),
    }
  })
}

function clusterHasGroup(cluster: UnderlyingCluster, groupKeyValue: string) {
  return cluster.legs.some(leg => groupKey(leg.group_name) === groupKeyValue)
}

function positionGroupState(rows: EditorRow[]) {
  const groups = new Map<string, GroupState>()
  const errors: string[] = []
  for (const row of rows) {
    const name = normalizeGroupName(row.group_name)
    const key = groupKey(name)
    if (!key || !name) continue
    const conviction = normalizeGroupConviction(row.group_conviction)
    if (conviction == null) {
      errors.push(`Group ${name} requires a conviction.`)
      continue
    }
    const direction = exposureDirection(row) ?? row.direction
    const existing = groups.get(key)
    if (!existing) {
      groups.set(key, {
        key,
        name,
        conviction,
        direction,
        ids: [row._id],
        tickers: [displayTicker(row) || row.ticker || "New position"],
      })
      continue
    }
    existing.ids.push(row._id)
    existing.tickers.push(displayTicker(row) || row.ticker || "New position")
    if (existing.conviction !== conviction) {
      errors.push(`Group ${existing.name} has inconsistent convictions.`)
    }
    if (existing.direction !== direction) {
      errors.push(`Group ${existing.name} cannot mix ${existing.direction} and ${direction} exposure.`)
    }
  }
  return { groups, errors: Array.from(new Set(errors)) }
}

function positionToRow(p: PortfolioPosition): EditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
  const defaultMultiplier = instrumentType === "future" ? null : instrumentType === "option" ? 100 : 1
  return {
    ...p,
    _id: makeId(),
    _isNew: false,
    _contractMultiplierTouched: false,
    quantity,
    shares: quantity,
    instrument_type: instrumentType,
    price_symbol: p.price_symbol ?? p.option_contract_symbol ?? p.ticker,
    underlying_ticker: p.underlying_ticker ?? (instrumentType === "option" ? p.ticker : null),
    contract_multiplier: p.contract_multiplier ?? defaultMultiplier,
    position_id: p.position_id ?? null,
  }
}

function newRow(): EditorRow {
  return {
    _id: makeId(),
    _isNew: true,
    ticker: "",
    asset: "equity",
    direction: "long",
    contrarian: false,
    conviction: 3,
    cost_basis: null,
    shares: null,
    quantity: null,
    instrument_type: "security",
    price_symbol: "",
    contract_multiplier: null,
    group_name: null,
    group_conviction: null,
    _contractMultiplierTouched: false,
  }
}

function hedgeToRow(p: HedgePosition): HedgeEditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
  const defaultMultiplier = instrumentType === "future" ? null : instrumentType === "option" ? 100 : 1
  return {
    ...p,
    _id: makeId(),
    _contractMultiplierTouched: false,
    ticker: p.ticker,
    asset: p.asset ?? "equity",
    direction: p.direction,
    cost_basis: p.cost_basis,
    shares: quantity,
    quantity,
    instrument_type: instrumentType,
    price_symbol: p.price_symbol ?? p.option_contract_symbol ?? p.ticker,
    underlying_ticker: p.underlying_ticker ?? (instrumentType === "option" ? p.ticker : null),
    contract_multiplier: p.contract_multiplier ?? defaultMultiplier,
    position_id: p.position_id ?? null,
  }
}

function newHedgeRow(): HedgeEditorRow {
  return {
    _id: makeId(),
    ticker: "",
    asset: "equity",
    direction: "short",
    cost_basis: null,
    shares: null,
    quantity: null,
    instrument_type: "security",
    price_symbol: "",
    contract_multiplier: null,
    _contractMultiplierTouched: false,
  }
}

/* ── Patch builders (shared by position & hedge rows) ──────────────────── */
function tickerChangePatch<T extends AnyRow>(row: T, rawValue: string): Partial<T> {
  const nextTicker = rawValue.toUpperCase()
  if (rowInstrumentType(row) === "option") {
    return applyOptionPaste({ ...row, underlying_ticker: nextTicker, ticker: nextTicker }) as unknown as Partial<T>
  }
  const currentPriceSymbol = row.price_symbol?.trim().toUpperCase()
  const nextPriceSymbol = !currentPriceSymbol || currentPriceSymbol === row.ticker.toUpperCase()
    ? nextTicker
    : row.price_symbol
  const nextInstrumentType = inferInstrumentType(nextTicker, row.instrument_type)
  const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol) : { fx_base_currency: null, fx_quote_currency: null }
  return {
    ticker: nextTicker,
    price_symbol: nextPriceSymbol,
    instrument_type: nextInstrumentType,
    asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
    fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
    fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
    contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol)),
  } as Partial<T>
}

function typeChangePatch<T extends AnyRow>(row: T, value: string): Partial<T> {
  const nextInstrumentType = value as InstrumentType
  const nextPriceSymbol = nextInstrumentType === "spot_fx"
    ? canonicalSpotFxSymbol(effectivePriceSymbol(row)) ?? row.price_symbol
    : row.price_symbol
  const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol || row.ticker) : { fx_base_currency: null, fx_quote_currency: null }
  const nextUnderlying = normalizedSymbol(row.underlying_ticker || row.ticker)
  return {
    instrument_type: nextInstrumentType,
    price_symbol: nextPriceSymbol,
    asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
    fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
    fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
    underlying_ticker: nextInstrumentType === "option" ? nextUnderlying : row.underlying_ticker,
    ticker: nextInstrumentType === "option" ? nextUnderlying : row.ticker,
    contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol ?? "")),
    _contractMultiplierTouched: nextInstrumentType === "future" || nextInstrumentType === "option"
      ? row._contractMultiplierTouched
      : false,
  } as Partial<T>
}

/* ════════════════════════════════════════════════════════════════════════
   Presentational pieces
   ════════════════════════════════════════════════════════════════════════ */

function ConvictionChips({ value, onChange }: { value: number; onChange: (c: number) => void }) {
  return (
    <div className="inline-flex items-center" style={{ gap: 3 }} title={`Conviction: ${CONVICTION_LABELS[value] ?? value}`}>
      {[1, 2, 3, 4, 5].map(c => {
        const active = c <= value
        return (
          <button
            key={c}
            type="button"
            onClick={() => onChange(c)}
            aria-label={`Conviction ${c} — ${CONVICTION_LABELS[c]}`}
            style={{
              width: 27,
              height: 27,
              borderRadius: 7,
              cursor: "pointer",
              padding: 0,
              fontSize: 12,
              fontWeight: 700,
              border: active ? "none" : "1px solid hsl(var(--border))",
              background: active ? convColor(value) : "hsl(var(--background-card-muted))",
              color: active ? "#fff" : "hsl(var(--foreground-quaternary))",
              boxShadow: active ? "0 1px 2px hsl(var(--shadow-color) / 0.14)" : "none",
              transition: "background 120ms ease, border-color 120ms ease",
            }}
          >
            {c}
          </button>
        )
      })}
    </div>
  )
}

function DirectionTag({ direction, exposure }: { direction: string; exposure?: string | null }) {
  const isLong = direction === "long"
  const mismatch = exposure && exposure !== direction
  return (
    <span
      className="theme-badge"
      title={mismatch ? `Economic exposure: ${exposure}` : undefined}
      style={{
        backgroundColor: isLong ? "hsl(var(--success-muted))" : "hsl(var(--destructive-muted))",
        color: isLong ? "hsl(var(--success))" : "hsl(var(--destructive))",
        borderColor: isLong ? "hsl(var(--success) / 0.18)" : "hsl(var(--destructive) / 0.18)",
      }}
    >
      <svg width="10" height="10" viewBox="0 0 10 10" style={{ transform: isLong ? "none" : "scaleY(-1)" }} aria-hidden="true">
        <path d="M1 7 L5 3 L9 7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {isLong ? "Long" : "Short"}
      {mismatch ? <span style={{ opacity: 0.6, fontWeight: 600 }}>≈{exposure![0].toUpperCase()}</span> : null}
    </span>
  )
}

const ASSET_BADGE_STYLE: Record<string, string> = {
  equity: "info",
  commodity: "warning",
  fx: "success",
  bond: "neutral",
}
function AssetBadge({ asset }: { asset?: string | null }) {
  const key = asset ?? "equity"
  return <span className={`theme-badge theme-badge-${ASSET_BADGE_STYLE[key] ?? "neutral"}`}>{assetLabel(key)}</span>
}

const INSTRUMENT_ICON: Record<string, string> = { security: "▣", future: "⬡", spot_fx: "⇄", option: "◇" }
function InstrumentBadge({ type }: { type?: string | null }) {
  return (
    <span className="theme-badge theme-badge-neutral">
      <span style={{ opacity: 0.7 }}>{INSTRUMENT_ICON[type ?? "security"] ?? "▣"}</span>
      {instrumentTypeLabel(type)}
    </span>
  )
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "pos" | "neg" }) {
  const color = tone === "pos" ? "hsl(var(--positive))" : tone === "neg" ? "hsl(var(--negative))" : "hsl(var(--foreground))"
  return (
    <div>
      <div className="label-text" style={{ marginBottom: 4 }}>{label}</div>
      <div className="mono-text" style={{ fontSize: "1.05rem", fontWeight: 700, color, lineHeight: 1.1 }}>{value}</div>
      {sub ? <div className="text-subtle" style={{ fontSize: "0.72rem", marginTop: 2 }}>{sub}</div> : null}
    </div>
  )
}

const ASSET_EXPOSURE_HSL: Record<string, string> = {
  equity: "211 90% 54%",
  commodity: "30 84% 50%",
  fx: "142 60% 42%",
  bond: "260 50% 58%",
}
function ExposureBar({ summary }: { summary: BookSummary }) {
  const entries = Object.entries(summary.byAsset).filter(([, v]) => Math.abs(v) > 0)
  const total = entries.reduce((s, [, v]) => s + Math.abs(v), 0) || 1
  if (entries.length === 0) {
    return <div className="text-subtle" style={{ fontSize: "0.78rem" }}>No exposure yet — add cost basis and quantity to size positions.</div>
  }
  return (
    <div>
      <div style={{ height: 10, borderRadius: 999, overflow: "hidden", display: "flex", background: "hsl(var(--separator))" }}>
        {entries.map(([a, v]) => (
          <div key={a} title={`${assetLabel(a)} ${fmtUSD0(v)}`} style={{ width: `${(Math.abs(v) / total) * 100}%`, background: `hsl(${ASSET_EXPOSURE_HSL[a] ?? "211 50% 54%"})` }} />
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px", marginTop: 8 }}>
        {entries.map(([a, v]) => (
          <span key={a} style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: "0.72rem", color: "hsl(var(--foreground-secondary))" }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: `hsl(${ASSET_EXPOSURE_HSL[a] ?? "211 50% 54%"})` }} />
            {assetLabel(a)} <span className="mono-text" style={{ fontWeight: 600 }}>{fmtUSD0(v)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/* Grid templates: positions carry a conviction column; hedges do not. */
const GRID_POSITION = "14px minmax(190px,1fr) 116px 168px 188px 38px"
const GRID_HEDGE = "14px minmax(190px,1fr) 116px 168px 38px"
const GRID_UNDERLYING = "14px minmax(170px,1fr) 100px 120px 150px minmax(130px,1fr) 38px"

interface RowCallbacks {
  onUpdate: (id: string, patch: Partial<AnyRow>) => void
  onRemove: (id: string) => void
}

interface EditorRowViewProps extends RowCallbacks {
  row: AnyRow
  bookSize: number
  isHedge: boolean
  spine?: string | null
  groups?: GroupState[]
  onConviction?: (id: string, c: number) => void
  onAssignGroup?: (id: string, name: string | null) => void
  suppressMetadata?: boolean
  hideSummary?: boolean
}

function EditorRowView({ row, bookSize, isHedge, spine, groups, onUpdate, onRemove, onConviction, onAssignGroup, suppressMetadata, hideSummary }: EditorRowViewProps) {
  const [open, setOpen] = useState(Boolean(hideSummary))
  const type = rowInstrumentType(row)
  const isOption = type === "option"
  const expo = exposureDirection(row) ?? row.direction
  const gross = grossNotional(row)
  const pct = bookSize > 0 ? gross / bookSize : 0
  const subtext = valuationSummary(row)
  const grid = isHedge || suppressMetadata ? GRID_HEDGE : GRID_POSITION
  const editorRow = row as EditorRow
  const showMetadata = !isHedge && !suppressMetadata

  const toggle = () => setOpen(o => !o)

  return (
    <div style={{ borderTop: "1px solid hsl(var(--separator))", background: open ? "hsl(var(--background-card-muted) / 0.5)" : "transparent" }}>
      {!hideSummary ? (
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggle}
        onKeyDown={e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            toggle()
          }
        }}
        style={{ display: "grid", gridTemplateColumns: grid, alignItems: "center", gap: 8, padding: "6px 12px", minHeight: 54, cursor: "pointer" }}
      >
        <div style={{ alignSelf: "stretch", borderRadius: 999, background: spine || "transparent", width: 4 }} />

        {/* Instrument */}
        <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
            <span className="theme-icon-button" style={{ width: 22, height: 22, flex: "0 0 auto" }} aria-hidden="true">
              <ChevronDown size={14} style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 140ms" }} />
            </span>
            <span className="mono-text" style={{ fontWeight: 700, fontSize: "0.95rem", flex: "0 0 auto" }}>{displayTicker(row) || "—"}</span>
            <div style={{ display: "flex", gap: 5, alignItems: "center", minWidth: 0 }}>
              <InstrumentBadge type={type} />
              <AssetBadge asset={row.asset} />
            </div>
          </div>
          {subtext ? (
            <div className="text-subtle" style={{ fontSize: "0.74rem", paddingLeft: 29, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{subtext}</div>
          ) : null}
        </div>

        {/* Direction — read-only color-coded tag */}
        <div><DirectionTag direction={row.direction} exposure={expo} /></div>

        {/* Notional + % of book */}
        <div style={{ textAlign: "right" }}>
          <div className="mono-text" style={{ fontSize: "0.92rem", fontWeight: 700, color: expo === "short" ? "hsl(var(--negative))" : "hsl(var(--foreground))" }}>
            {expo === "short" ? "−" : ""}{fmtUSD0(gross)}
          </div>
          <div className="text-subtle" style={{ fontSize: "0.72rem" }}>{fmtPct(pct)} of book</div>
        </div>

        {/* Conviction — positions only, editable in-row */}
        {showMetadata ? (
          <div style={{ display: "flex", justifyContent: "center" }} onClick={e => e.stopPropagation()}>
            <ConvictionChips value={editorRow.conviction} onChange={c => onConviction?.(row._id, c)} />
          </div>
        ) : null}

        {/* Delete */}
        <button
          type="button"
          className="theme-icon-button"
          style={{ width: 32, height: 32 }}
          onClick={e => {
            e.stopPropagation()
            onRemove(row._id)
          }}
          aria-label="Remove position"
          title="Remove"
        >
          <Trash2 size={15} />
        </button>
      </div>
      ) : null}

      {open ? (
        <div style={{ padding: hideSummary ? "6px 18px 20px 18px" : "6px 18px 20px 42px", display: "grid", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 16, alignItems: "start" }}>
            <div>
              <div className="label-text" style={{ marginBottom: 8 }}>Direction</div>
              {isHedge || editorRow._isNew ? (
                <SegmentedControl
                  options={DIRECTION_OPTIONS}
                  value={row.direction}
                  onChange={v => onUpdate(row._id, { direction: v as PortfolioPosition["direction"] })}
                  size="sm"
                />
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <DirectionTag direction={row.direction} exposure={expo} />
                  <span className="text-subtle" style={{ fontSize: "0.72rem" }}>Direction is fixed for existing positions.</span>
                </div>
              )}
            </div>
            {showMetadata ? (
              <label style={{ display: "block" }}>
                <span className="theme-field-label">Group</span>
                <select
                  className="theme-input"
                  value={editorRow.group_name ?? ""}
                  onChange={e => onAssignGroup?.(row._id, e.target.value || null)}
                >
                  <option value="">— Ungrouped —</option>
                  {(groups ?? []).map(g => (
                    <option key={g.key} value={g.name}>{g.name}</option>
                  ))}
                </select>
              </label>
            ) : <div />}
          </div>

          <div>
            <div className="label-text" style={{ marginBottom: 10 }}>Instrument Detail</div>
            <InstrumentDetailFields row={row} onUpdate={onUpdate} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <label>
              <span className="theme-field-label">Quantity</span>
              <input
                className="theme-input mono-text"
                type="number"
                value={rowQuantity(row) ?? ""}
                placeholder={type === "spot_fx" ? "Base units" : type === "future" || isOption ? "Contracts" : "Shares"}
                onChange={e => {
                  const v = e.target.value
                  const quantity = v === "" ? null : Number(v)
                  onUpdate(row._id, { shares: quantity, quantity })
                }}
              />
            </label>
            <label>
              <span className="theme-field-label">{type === "spot_fx" ? "Entry Rate" : "Cost Basis"}</span>
              <input
                className="theme-input mono-text"
                type="number"
                value={row.cost_basis ?? ""}
                placeholder="Price"
                onChange={e => {
                  const v = e.target.value
                  onUpdate(row._id, { cost_basis: v === "" ? null : Number(v) })
                }}
              />
            </label>
          </div>

          {!isHedge ? (
            <label className="flex items-center gap-3 select-none" style={{ cursor: "pointer" }}>
              <button
                type="button"
                role="switch"
                aria-checked={Boolean(editorRow.contrarian)}
                onClick={() => onUpdate(row._id, { contrarian: !editorRow.contrarian } as Partial<AnyRow>)}
                className="relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200"
                style={{ backgroundColor: editorRow.contrarian ? "hsl(var(--accent))" : "hsl(var(--separator))" }}
              >
                <span
                  className={`pointer-events-none inline-block h-[18px] w-[18px] rounded-full shadow-sm transition-transform duration-200 mt-[2px] ${editorRow.contrarian ? "translate-x-[20px]" : "translate-x-[2px]"}`}
                  style={{ backgroundColor: "hsl(var(--background-elevated))" }}
                />
              </button>
              <span>
                <span className="text-sm font-medium text-app">Contrarian</span>
                <span className="mt-0.5 block text-xs text-subtle">Flag positions that run against consensus.</span>
              </span>
            </label>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function InstrumentDetailFields({ row, onUpdate }: { row: AnyRow; onUpdate: (id: string, patch: Partial<AnyRow>) => void }) {
  const type = rowInstrumentType(row)
  const isOption = type === "option"
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <label style={{ display: "block" }}>
          <span className="theme-field-label">Instrument Type</span>
          <SelectInput
            value={type}
            onChange={v => onUpdate(row._id, typeChangePatch(row, v))}
            options={INSTRUMENT_TYPE_OPTIONS}
          />
        </label>
        <label style={{ display: "block" }}>
          <span className="theme-field-label">Asset Class</span>
          <SelectInput
            value={row.asset ?? "equity"}
            onChange={v => onUpdate(row._id, { asset: v as PortfolioPosition["asset"] })}
            options={ASSET_OPTIONS}
            disabled={type === "spot_fx"}
          />
        </label>
      </div>

      {!isOption ? (
        <label style={{ display: "block" }}>
          <span className="theme-field-label">{type === "spot_fx" ? "FX Pair" : type === "future" ? "Price Symbol" : "Ticker"}</span>
          <input
            className="theme-input mono-text"
            value={row.ticker}
            placeholder={type === "spot_fx" ? "EURUSD=X" : type === "future" ? "CL=F" : "AAPL"}
            onChange={e => onUpdate(row._id, tickerChangePatch(row, e.target.value))}
          />
        </label>
      ) : null}

      {isOption ? (
        <div style={{ display: "grid", gap: 12 }}>
          <label style={{ display: "block" }}>
            <span className="theme-field-label">OCC Contract Symbol</span>
            <input
              className="theme-input mono-text"
              value={row.option_contract_symbol ?? ""}
              placeholder="NVDA251219C00150000"
              onChange={e => onUpdate(row._id, applyOptionPaste({ ...row, option_contract_symbol: e.target.value.toUpperCase() }) as Partial<AnyRow>)}
            />
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <label style={{ display: "block" }}>
              <span className="theme-field-label">Underlying</span>
              <input
                className="theme-input mono-text"
                value={row.underlying_ticker ?? ""}
                onChange={e => onUpdate(row._id, applyOptionPaste({ ...row, underlying_ticker: e.target.value.toUpperCase(), ticker: e.target.value.toUpperCase() }) as Partial<AnyRow>)}
              />
            </label>
            <label style={{ display: "block" }}>
              <span className="theme-field-label">Call / Put</span>
              <div style={{ paddingTop: 2 }}>
                <SegmentedControl
                  options={[...OPTION_TYPE_OPTIONS]}
                  value={row.option_type ?? "call"}
                  onChange={v => onUpdate(row._id, { option_type: v as PortfolioPosition["option_type"] })}
                  size="sm"
                />
              </div>
            </label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
            <label style={{ display: "block" }}>
              <span className="theme-field-label">Strike</span>
              <input
                className="theme-input mono-text"
                type="number"
                value={row.option_strike ?? ""}
                onChange={e => {
                  const v = e.target.value
                  onUpdate(row._id, { option_strike: v === "" ? null : Number(v) })
                }}
              />
            </label>
            <label style={{ display: "block" }}>
              <span className="theme-field-label">Expiration</span>
              <input
                className="theme-input"
                type="date"
                value={row.option_expiration ?? ""}
                onChange={e => onUpdate(row._id, { option_expiration: e.target.value || null })}
              />
            </label>
            <label style={{ display: "block" }}>
              <span className="theme-field-label">Multiplier</span>
              <input
                className="theme-input mono-text"
                type="number"
                value={row.contract_multiplier ?? ""}
                placeholder="100"
                onChange={e => {
                  const v = e.target.value
                  onUpdate(row._id, { contract_multiplier: v === "" ? null : Number(v), _contractMultiplierTouched: true })
                }}
              />
            </label>
          </div>
          <p className="text-subtle" style={{ fontSize: "0.74rem" }}>
            {optionContractSymbolForRow(row) ?? "Enter fields or paste an OCC symbol"}
          </p>
        </div>
      ) : null}

      {type === "future" ? (
        <label style={{ display: "block", maxWidth: 240 }}>
          <span className="theme-field-label">Contract Multiplier</span>
          <input
            className="theme-input mono-text"
            type="number"
            value={row.contract_multiplier ?? ""}
            placeholder="Auto"
            onChange={e => {
              const v = e.target.value
              onUpdate(row._id, { contract_multiplier: v === "" ? null : Number(v), _contractMultiplierTouched: true })
            }}
          />
        </label>
      ) : null}

      {type === "spot_fx" && row.fx_base_currency && row.fx_quote_currency ? (
        <p className="text-subtle" style={{ fontSize: "0.74rem" }}>{row.fx_base_currency} / {row.fx_quote_currency} spot pair</p>
      ) : null}
    </div>
  )
}

interface UnderlyingBandProps {
  cluster: UnderlyingCluster
  bookSize: number
  spine?: string | null
  groups: GroupState[]
  rowCallbacks: RowCallbacks
  onConviction: (ids: string[], conviction: number) => void
  onAssignGroup: (ids: string[], name: string | null) => void
}

function UnderlyingBand({ cluster, bookSize, spine, groups, rowCallbacks, onConviction, onAssignGroup }: UnderlyingBandProps) {
  const [open, setOpen] = useState(cluster.legs.length > 1)
  const legIds = cluster.legs.map(leg => leg._id)
  const pct = bookSize > 0 ? cluster.gross / bookSize : 0
  const hasOptions = cluster.legs.some(leg => rowInstrumentType(leg) === "option")
  const hasSecurity = cluster.legs.some(leg => rowInstrumentType(leg) !== "option")
  const primaryType = hasOptions && hasSecurity ? "option" : hasOptions ? "option" : rowInstrumentType(cluster.legs[0])
  const primaryAsset = cluster.legs[0]?.asset ?? "equity"
  const subtext = cluster.legs.length === 1 ? valuationSummary(cluster.legs[0]) : `${cluster.legs.length} legs · expand to edit`

  return (
    <div style={{ borderTop: "1px solid hsl(var(--separator))", background: open ? "hsl(var(--background-card-muted) / 0.35)" : "transparent" }}>
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
        onKeyDown={e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            setOpen(value => !value)
          }
        }}
        style={{ display: "grid", gridTemplateColumns: GRID_UNDERLYING, alignItems: "center", gap: 8, padding: "6px 12px", minHeight: 54, cursor: cluster.legs.length > 1 ? "pointer" : "default" }}
      >
        <div style={{ alignSelf: "stretch", borderRadius: 999, background: spine || "transparent", width: 4 }} />

        <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
            {cluster.legs.length > 1 ? (
              <span className="theme-icon-button" style={{ width: 22, height: 22, flex: "0 0 auto" }} aria-hidden="true">
                <ChevronDown size={14} style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 140ms" }} />
              </span>
            ) : (
              <span style={{ width: 22, flex: "0 0 auto" }} />
            )}
            <span className="mono-text" style={{ fontWeight: 700, fontSize: "0.95rem", flex: "0 0 auto" }}>{cluster.ticker || "—"}</span>
            <div style={{ display: "flex", gap: 5, alignItems: "center", minWidth: 0 }}>
              <InstrumentBadge type={primaryType} />
              <AssetBadge asset={primaryAsset} />
              {cluster.legs.length > 1 ? (
                <span className="theme-badge theme-badge-neutral">{cluster.legs.length} legs</span>
              ) : null}
            </div>
          </div>
          {subtext ? (
            <div className="text-subtle" style={{ fontSize: "0.74rem", paddingLeft: 29, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{subtext}</div>
          ) : null}
        </div>

        <div><DirectionTag direction={cluster.direction} /></div>

        <div style={{ display: "flex", justifyContent: "center" }} onClick={e => e.stopPropagation()}>
          <ConvictionChips value={cluster.conviction} onChange={c => onConviction(legIds, c)} />
        </div>

        <div style={{ textAlign: "right" }}>
          <div className="mono-text" style={{ fontSize: "0.92rem", fontWeight: 700, color: cluster.net < 0 ? "hsl(var(--negative))" : "hsl(var(--foreground))" }}>
            {cluster.net < 0 ? "−" : ""}{fmtUSD0(cluster.gross)}
          </div>
          <div className="text-subtle" style={{ fontSize: "0.72rem" }}>{fmtPct(pct)} of book</div>
        </div>

        <label style={{ display: "block", minWidth: 0 }} onClick={e => e.stopPropagation()}>
          <span className="sr-only">Group</span>
          <select
            className="theme-input w-full text-sm"
            value={cluster.groupName ?? ""}
            onChange={e => onAssignGroup(legIds, e.target.value || null)}
          >
            <option value="">— Ungrouped —</option>
            {groups.map(group => (
              <option key={group.key} value={group.name}>{group.name}</option>
            ))}
          </select>
        </label>

        <div />
      </div>

      {cluster.legs.length === 1 ? (
        <EditorRowView
          row={cluster.legs[0]}
          bookSize={bookSize}
          isHedge={false}
          spine={spine}
          groups={groups}
          onUpdate={rowCallbacks.onUpdate}
          onRemove={rowCallbacks.onRemove}
          suppressMetadata
          hideSummary
        />
      ) : open ? (
        <div style={{ background: "hsl(var(--background-card) / 0.45)" }}>
          {cluster.legs.map(leg => (
            <EditorRowView
              key={leg._id}
              row={leg}
              bookSize={bookSize}
              isHedge={false}
              spine={spine}
              groups={groups}
              onUpdate={rowCallbacks.onUpdate}
              onRemove={rowCallbacks.onRemove}
              suppressMetadata
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

interface GroupBandProps {
  group: GroupState
  clusters: UnderlyingCluster[]
  bookSize: number
  onRename: (key: string, name: string) => void
  onConvictionAll: (name: string, c: number) => void
  onAddToGroup: (name: string, conviction: number) => void
  onDisband: (key: string) => void
  rowCallbacks: RowCallbacks
  groups: GroupState[]
  onClusterConviction: (ids: string[], c: number) => void
  onAssignClusterGroup: (ids: string[], name: string | null) => void
}

function GroupBand({ group, clusters, bookSize, onRename, onConvictionAll, onAddToGroup, onDisband, rowCallbacks, groups, onClusterConviction, onAssignClusterGroup }: GroupBandProps) {
  const [draftName, setDraftName] = useState(group.name)
  const members = clusters.flatMap(cluster => cluster.legs)
  if (members.length === 0) return null
  const col = groupColor(group.key)
  const gross = members.reduce((s, m) => s + grossNotional(m), 0)
  const net = members.reduce((s, m) => s + netNotional(m), 0)
  const wConv = gross > 0 ? members.reduce((s, m) => s + m.conviction * grossNotional(m), 0) / gross : group.conviction
  const directions = new Set(members.map(m => exposureDirection(m) ?? m.direction))
  const mixed = directions.size > 1
  const pct = bookSize > 0 ? Math.abs(net) / bookSize : 0

  function commitName() {
    const nextName = normalizeGroupName(draftName) ?? ""
    setDraftName(nextName)
    onRename(group.key, nextName)
  }

  return (
    <div style={{ marginTop: 14, borderRadius: "var(--radius-lg)", overflow: "hidden", border: "1px solid hsl(var(--border))", boxShadow: "var(--shadow-soft)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "12px 14px", background: groupTint(group.key, 0.1), borderLeft: `4px solid ${col}`, flexWrap: "wrap" }}>
        <span style={{ color: col, display: "inline-flex" }}><Layers size={15} /></span>
        <input
          value={draftName}
          onChange={e => setDraftName(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter") {
              e.preventDefault()
              e.currentTarget.blur()
            }
          }}
          style={{
            border: "1px solid transparent",
            borderRadius: 8,
            background: "transparent",
            padding: "0.25rem 0.4rem",
            fontSize: "0.98rem",
            fontWeight: 650,
            color: "hsl(var(--foreground))",
            outline: "none",
            minWidth: 120,
            width: `${Math.max(8, draftName.length + 1)}ch`,
          }}
          onFocus={e => (e.target.style.background = "hsl(var(--background-input))")}
          onBlur={e => {
            e.target.style.background = "transparent"
            commitName()
          }}
          aria-label={`Group name for ${group.name}`}
        />
        <span className="theme-badge theme-badge-neutral">{members.length} {members.length === 1 ? "position" : "positions"}</span>
        {mixed ? <span className="theme-badge theme-badge-warning" title="Group positions should share one direction">⚠ Mixed direction</span> : null}
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="label-text" style={{ whiteSpace: "nowrap" }}>Group Conviction</span>
          <ConvictionChips value={group.conviction} onChange={c => onConvictionAll(group.name, c)} />
        </div>
        <div style={{ textAlign: "right", minWidth: 120 }}>
          <div className="label-text">Net Notional</div>
          <div className="mono-text" style={{ fontSize: "0.92rem", fontWeight: 700, color: net < 0 ? "hsl(var(--negative))" : "hsl(var(--foreground))" }}>{fmtUSD0(net)}</div>
          <div className="text-subtle" style={{ fontSize: "0.7rem" }}>{fmtPct(pct)} of book</div>
        </div>
        <button
          type="button"
          className="theme-icon-button"
          onClick={() => onAddToGroup(group.name, group.conviction)}
          title="Add to group"
          style={{ background: "hsl(var(--background-elevated))", border: "1px solid hsl(var(--border))" }}
        >
          <Plus size={16} />
        </button>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", background: "hsl(var(--background-card-muted) / 0.5)" }}>
        <span className="text-subtle" style={{ fontSize: "0.74rem", whiteSpace: "nowrap" }}>Weighted conviction</span>
        <div style={{ flex: 1, height: 5, borderRadius: 999, background: "hsl(var(--separator))", maxWidth: 220 }}>
          <div style={{ height: "100%", borderRadius: 999, width: `${(wConv / 5) * 100}%`, background: convColor(Math.round(wConv)) }} />
        </div>
        <span className="mono-text" style={{ fontSize: "0.74rem", fontWeight: 700 }}>{wConv.toFixed(2)} / 5</span>
        <button
          type="button"
          className="theme-button-base theme-button-ghost"
          style={{ minHeight: 30, fontSize: "0.76rem", paddingInline: 12, marginLeft: "auto" }}
          onClick={() => onDisband(group.key)}
        >
          Disband
        </button>
      </div>

      <div style={{ background: "hsl(var(--background-card) / 0.6)" }}>
        {clusters.map(cluster => (
          <UnderlyingBand
            key={cluster.key}
            cluster={cluster}
            bookSize={bookSize}
            spine={col}
            groups={groups}
            rowCallbacks={rowCallbacks}
            onConviction={onClusterConviction}
            onAssignGroup={onAssignClusterGroup}
          />
        ))}
      </div>
    </div>
  )
}

function SectionCard({ title, icon, count, children, footer }: { title: string; icon: ReactNode; count: number; children: ReactNode; footer?: ReactNode }) {
  return (
    <div className="theme-surface" style={{ marginTop: 14, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 14px" }}>
        <span style={{ color: "hsl(var(--foreground-tertiary))", display: "inline-flex" }}>{icon}</span>
        <span style={{ fontWeight: 650, fontSize: "0.95rem", whiteSpace: "nowrap" }}>{title}</span>
        <span className="theme-badge theme-badge-neutral">{count}</span>
      </div>
      {children}
      {footer ? <div style={{ borderTop: "1px solid hsl(var(--separator))", padding: 10, display: "flex", gap: 10 }}>{footer}</div> : null}
    </div>
  )
}

export interface PortfolioEditorPanelProps {
  onCancel?: () => void
}

export function PortfolioEditorPanel({ onCancel }: PortfolioEditorPanelProps = {} as PortfolioEditorPanelProps) {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<EditorTab>("Positions")
  const [positionRows, setPositionRows] = useState<EditorRow[]>([])
  const [hedgeRows, setHedgeRows] = useState<HedgeEditorRow[]>([])
  const [bookSizeInput, setBookSizeInput] = useState(String(DEFAULT_BOOK_SIZE))
  const [loadError, setLoadError] = useState<string | null>(null)
  const [settingsValidationError, setSettingsValidationError] = useState<string | null>(null)
  const [settingsSavedMessage, setSettingsSavedMessage] = useState<string | null>(null)
  const [positionValidationError, setPositionValidationError] = useState<string | null>(null)
  const [hedgeValidationError, setHedgeValidationError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastProposals, setLastProposals] = useState<StagedMutationResponse[]>([])
  const [ibkrImportSummary, setIbkrImportSummary] = useState<string | null>(null)
  const [ibkrImportError, setIbkrImportError] = useState<string | null>(null)
  const ibkrImportInputRef = useRef<HTMLInputElement | null>(null)

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    setTab("Positions")
    setLoadError(null)
    setSettingsValidationError(null)
    setSettingsSavedMessage(null)
    setPositionValidationError(null)
    setHedgeValidationError(null)
    setIbkrImportError(null)
    setIbkrImportSummary(null)
    setLastProposals([])
    setIsLoading(true)
    Promise.all([
      fetchPortfolioPositions(),
      fetchHedgePositions(),
      fetchPortfolioSettings().catch(err => {
        setSettingsValidationError(`Failed to load book size: ${String(err)}`)
        return null
      }),
    ])
      .then(([portfolioData, hedgeData, settingsData]) => {
        setPositionRows(portfolioData.positions.map(positionToRow))
        setHedgeRows(hedgeData.positions.map(hedgeToRow))
        if (settingsData) setBookSizeInput(String(settingsData.book_size ?? DEFAULT_BOOK_SIZE))
      })
      .catch(err => setLoadError(String(err)))
      .finally(() => setIsLoading(false))
  }, [])
  /* eslint-enable react-hooks/set-state-in-effect */

  function handleSaved(result: StagedMutationResponse | StagedMutationResponse[]) {
    const proposals = Array.isArray(result) ? result : [result]
    setLastProposals(proposals)
    void invalidateApprovalSummaries(queryClient)
  }

  function handleIbkrImportSuccess(result: IbkrFlexImportResponse) {
    setIbkrImportError(null)
    const proposals = result.staged_proposals?.length ? result.staged_proposals : [result]
    handleSaved(proposals)
    const summary = result.import_summary
    if (summary) {
      const hedgeCount = summary.hedge_imported_count ?? 0
      const hedgeTickers = summary.hedge_tickers?.length ? summary.hedge_tickers.join(", ") : null
      const parts = [
        `${summary.imported_count} row(s) parsed`,
        `${summary.portfolio_imported_count ?? summary.imported_count - hedgeCount} portfolio`,
      ]
      if (hedgeCount > 0) parts.push(`${hedgeCount} hedge${hedgeTickers ? ` (${hedgeTickers})` : ""}`)
      setIbkrImportSummary(parts.join(" · "))
    } else {
      setIbkrImportSummary(null)
    }
  }

  const positionMutation = useMutation({
    mutationFn: (positions: PortfolioPosition[]) => savePortfolioPositions(positions),
    onSuccess: handleSaved,
  })

  const hedgeMutation = useMutation({
    mutationFn: (positions: HedgePosition[]) => saveHedgePositions(positions),
    onSuccess: handleSaved,
  })

  const ibkrImportMutation = useMutation({
    mutationFn: (file: File) => importIbkrFlexPortfolioPositions(file),
    onSuccess: handleIbkrImportSuccess,
    onError: err => setIbkrImportError(String(err)),
  })

  const settingsMutation = useMutation({
    mutationFn: (bookSize: number) => updatePortfolioSettings({ book_size: bookSize }),
    onSuccess: settings => {
      setBookSizeInput(String(settings.book_size))
      setSettingsValidationError(null)
      setSettingsSavedMessage(`Book size saved at ${formatBaseCurrency(settings.book_size)}.`)
      queryClient.removeQueries({ queryKey: SIZER_STATE_QUERY_KEY, exact: false })
    },
  })

  function updatePositionRow(id: string, patch: Partial<AnyRow>) {
    setPositionRows(prev => prev.map(r => (r._id === id ? { ...r, ...(patch as Partial<EditorRow>) } : r)))
  }

  function updateHedgeRow(id: string, patch: Partial<AnyRow>) {
    setHedgeRows(prev => prev.map(r => (r._id === id ? { ...r, ...(patch as Partial<HedgeEditorRow>) } : r)))
  }

  function removePositionRow(id: string) {
    setPositionRows(prev => prev.filter(r => r._id !== id))
  }

  function removeHedgeRow(id: string) {
    setHedgeRows(prev => prev.filter(r => r._id !== id))
  }

  function addPositionRow() {
    setPositionRows(prev => [...prev, newRow()])
  }

  function addGroup() {
    setPositionRows(prev => {
      const existing = new Set(prev.map(r => groupKey(r.group_name)).filter(Boolean) as string[])
      const base = "New Group"
      let name = base
      let n = 2
      while (existing.has(groupKey(name) as string)) {
        name = `${base} ${n}`
        n += 1
      }
      return [...prev, { ...newRow(), group_name: name, group_conviction: 3 }]
    })
  }

  function addPositionToGroup(name: string, conviction: number) {
    setPositionRows(prev => [...prev, { ...newRow(), group_name: name, group_conviction: conviction }])
  }

  function renameGroup(key: string, value: string) {
    const nextName = normalizeGroupName(value)
    setPositionRows(prev => prev.map(item => (
      groupKey(item.group_name) === key
        ? { ...item, group_name: nextName, group_conviction: nextName ? item.group_conviction : null }
        : item
    )))
  }

  function setGroupConviction(name: string | null | undefined, conviction: number) {
    const key = groupKey(name)
    if (!key) return
    setPositionRows(prev => prev.map(row => (groupKey(row.group_name) === key ? { ...row, group_conviction: conviction } : row)))
  }

  function disbandGroup(key: string) {
    setPositionRows(prev => prev.map(row => (groupKey(row.group_name) === key ? { ...row, group_name: null, group_conviction: null } : row)))
  }

  function setClusterConviction(ids: string[], conviction: number) {
    const idSet = new Set(ids)
    setPositionRows(prev => prev.map(r => (idSet.has(r._id) ? { ...r, conviction } : r)))
  }

  function assignClusterGroup(ids: string[], name: string | null) {
    setPositionRows(prev => {
      const normalized = normalizeGroupName(name)
      const idSet = new Set(ids)
      if (!normalized) {
        return prev.map(row => (idSet.has(row._id) ? { ...row, group_name: null, group_conviction: null } : row))
      }
      const key = groupKey(normalized)
      const existing = prev.find(row => !idSet.has(row._id) && groupKey(row.group_name) === key)
      const targetLegs = prev.filter(row => idSet.has(row._id))
      const groupConviction = normalizeGroupConviction(existing?.group_conviction)
        ?? normalizeGroupConviction(targetLegs[0]?.group_conviction)
        ?? clusterConviction(targetLegs)
        ?? 3
      return prev.map(row => (
        idSet.has(row._id)
          ? { ...row, group_name: normalized, group_conviction: groupConviction }
          : row
      ))
    })
  }

  function addHedgeRow() {
    setHedgeRows(prev => [...prev, newHedgeRow()])
  }

  function handleIbkrFlexImport(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return
    setIbkrImportError(null)
    ibkrImportMutation.mutate(file)
  }

  function handleSaveBookSize() {
    setSettingsValidationError(null)
    setSettingsSavedMessage(null)
    const bookSize = parseBookSizeInput(bookSizeInput)
    if (bookSize == null) {
      setSettingsValidationError("Book size must be a number.")
      return
    }
    if (bookSize < MIN_BOOK_SIZE || bookSize > MAX_BOOK_SIZE) {
      setSettingsValidationError(`Book size must be between ${formatBaseCurrency(MIN_BOOK_SIZE)} and ${formatBaseCurrency(MAX_BOOK_SIZE)}.`)
      return
    }
    settingsMutation.mutate(bookSize)
  }

  function handleSavePositions() {
    setPositionValidationError(null)

    const positionIds = positionRows.map(row => positionRowId({
      ticker: row.ticker,
      position_id: row.position_id,
      option_contract_symbol: optionContractSymbolForRow(row) ?? undefined,
      price_symbol: row.price_symbol,
      instrument_type: rowInstrumentType(row),
    })).filter(Boolean)
    const unique = new Set(positionIds)
    if (unique.size !== positionIds.length) {
      setPositionValidationError("Duplicate position IDs detected. Each leg must be unique.")
      return
    }
    if (positionRows.some(r => rowInstrumentType(r) !== "option" && !r.ticker.trim())) {
      setPositionValidationError("All rows must have a ticker.")
      return
    }
    if (positionRows.some(r => rowInstrumentType(r) === "option" && !(r.underlying_ticker || r.ticker).trim())) {
      setPositionValidationError("Option rows must have an underlying ticker.")
      return
    }
    if (positionRows.some(r => rowInstrumentType(r) === "option" && !optionContractSymbolForRow(r))) {
      setPositionValidationError("Option rows require expiration, strike, type, or a valid OCC contract symbol.")
      return
    }
    if (positionRows.length === 0) {
      setPositionValidationError("At least one position is required.")
      return
    }
    if (positionRows.some(r => rowInstrumentType(r) === "spot_fx" && !canonicalSpotFxSymbol(r.price_symbol || r.ticker))) {
      setPositionValidationError("Spot FX rows must use a pair like EURUSD=X, EURUSD, EUR/USD, or EUR-USD.")
      return
    }
    const groupState = positionGroupState(positionRows)
    if (groupState.errors.length > 0) {
      setPositionValidationError(groupState.errors[0])
      return
    }

    try {
      const positions: PortfolioPosition[] = positionRows.map(r => {
        const rowGroupName = normalizeGroupName(r.group_name)
        const rowGroup = rowGroupName ? groupState.groups.get(groupKey(rowGroupName) ?? "") : null
        return serializeInstrumentRow<PortfolioPosition>(r, {
          contrarian: r.contrarian,
          conviction: r.conviction,
          group_name: rowGroup?.name ?? rowGroupName,
          group_conviction: rowGroupName ? rowGroup?.conviction ?? normalizeGroupConviction(r.group_conviction) : null,
        })
      })
      positionMutation.mutate(positions)
    } catch (err) {
      setPositionValidationError(String(err))
    }
  }

  function handleSaveHedges() {
    setHedgeValidationError(null)

    const positionIds = hedgeRows.map(row => positionRowId({
      ticker: row.ticker,
      position_id: row.position_id,
      option_contract_symbol: optionContractSymbolForRow(row) ?? undefined,
      price_symbol: row.price_symbol,
      instrument_type: rowInstrumentType(row),
    })).filter(Boolean)
    const unique = new Set(positionIds)
    if (unique.size !== positionIds.length) {
      setHedgeValidationError("Duplicate position IDs detected. Each leg must be unique.")
      return
    }
    if (hedgeRows.some(r => rowInstrumentType(r) !== "option" && !r.ticker.trim())) {
      setHedgeValidationError("All hedge rows must have a ticker.")
      return
    }
    if (hedgeRows.some(r => rowInstrumentType(r) === "option" && !(r.underlying_ticker || r.ticker).trim())) {
      setHedgeValidationError("Option hedge rows must have an underlying ticker.")
      return
    }
    if (hedgeRows.some(r => rowInstrumentType(r) === "option" && !optionContractSymbolForRow(r))) {
      setHedgeValidationError("Option hedge rows require expiration, strike, type, or a valid OCC contract symbol.")
      return
    }
    if (hedgeRows.some(r => rowInstrumentType(r) === "spot_fx" && !canonicalSpotFxSymbol(r.price_symbol || r.ticker))) {
      setHedgeValidationError("Spot FX rows must use a pair like EURUSD=X, EURUSD, EUR/USD, or EUR-USD.")
      return
    }

    try {
      const positions: HedgePosition[] = hedgeRows.map(r => serializeInstrumentRow<HedgePosition>(r))
      hedgeMutation.mutate(positions)
    } catch (err) {
      setHedgeValidationError(String(err))
    }
  }

  const currentValidationError = tab === "Positions" ? positionValidationError : hedgeValidationError
  const currentMutationError = tab === "Positions"
    ? (positionMutation.isError ? String(positionMutation.error) : null)
    : (hedgeMutation.isError ? String(hedgeMutation.error) : null)
  const currentLoading = tab === "Positions" ? positionMutation.isPending : hedgeMutation.isPending
  const currentLoadingText = tab === "Positions" ? "Proposing portfolio..." : "Proposing hedges..."
  const currentSaveLabel = tab === "Positions" ? "Propose Portfolio" : "Propose Hedges"
  const currentGroupState = positionGroupState(positionRows)
  const saveDisabled = tab === "Positions" && currentGroupState.errors.length > 0

  const bookSizeNum = (() => {
    const parsed = parseBookSizeInput(bookSizeInput)
    return parsed && parsed > 0 ? parsed : DEFAULT_BOOK_SIZE
  })()
  const activeRows: AnyRow[] = tab === "Positions" ? positionRows : hedgeRows
  const summary = summarize(activeRows, bookSizeNum)
  const orderedGroups = Array.from(currentGroupState.groups.values())
  const underlyingClusters = buildUnderlyingClusters(positionRows)
  const ungroupedClusters = underlyingClusters.filter(cluster => !clusterGroupName(cluster.legs))

  if (isLoading) {
    return <p className="text-sm text-muted py-4">Loading portfolio and hedge positions...</p>
  }

  if (loadError) {
    return (
      <div className="theme-notice theme-notice-error">{loadError}</div>
    )
  }

  const positionRowCallbacks: RowCallbacks = { onUpdate: updatePositionRow, onRemove: removePositionRow }

  return (
    <div className="tal-root">
      {/* Action header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, marginBottom: 18, flexWrap: "wrap" }}>
        <SegmentedControl
          options={[
            { value: "Positions", label: "Positions" },
            { value: "Hedges", label: "Hedges" },
          ]}
          value={tab}
          onChange={setTab}
        />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {tab === "Positions" ? (
            <>
              <input
                ref={ibkrImportInputRef}
                type="file"
                accept=".xml,application/xml,text/xml"
                className="hidden"
                onChange={handleIbkrFlexImport}
              />
              <button
                type="button"
                className="theme-button-base theme-button-secondary px-4"
                onClick={() => ibkrImportInputRef.current?.click()}
                disabled={ibkrImportMutation.isPending}
              >
                <Upload size={15} /> {ibkrImportMutation.isPending ? "Importing…" : "Import IBKR Flex"}
              </button>
            </>
          ) : null}
          {onCancel ? (
            <button
              type="button"
              onClick={onCancel}
              className="theme-button-base theme-button-ghost px-4"
            >
              Cancel
            </button>
          ) : null}
          <ActionButton
            onClick={tab === "Positions" ? handleSavePositions : handleSaveHedges}
            loading={currentLoading}
            loadingText={currentLoadingText}
            disabled={saveDisabled}
            className="w-auto px-5"
          >
            {currentSaveLabel}
          </ActionButton>
        </div>
      </div>

      {(ibkrImportSummary || ibkrImportError) && tab === "Positions" ? (
        <div className={`theme-notice ${ibkrImportError ? "theme-notice-error" : "theme-notice-success"}`} style={{ marginBottom: 14 }}>
          {ibkrImportError ?? ibkrImportSummary}
        </div>
      ) : null}

      {lastProposals.map((proposal, index) => (
        <StagedProposalNotice key={proposal.approval_id ?? `${proposal.entity_type}-${index}`} proposal={proposal} className="mb-4">
          staged for {proposalSubjectLabel(proposal.entity_type)}. Review it in Workspace before app state changes.
        </StagedProposalNotice>
      ))}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr)", gap: 22, alignItems: "start" }} className="portfolio-editor-grid">
        {/* Summary */}
        <div className="theme-surface portfolio-editor-summary" style={{ padding: 18, display: "grid", gap: 18 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 24, alignItems: "start" }}>
            <div>
              <div className="label-text" style={{ marginBottom: 4 }}>Book Size</div>
              <div style={{ display: "flex", alignItems: "center", border: "1px solid hsl(var(--border))", borderRadius: "var(--radius-md)", background: "hsl(var(--background-input))", height: "2.4rem", paddingInline: 10 }}>
                <span className="text-subtle" style={{ fontSize: "0.9rem" }}>$</span>
                <input
                  className="mono-text"
                  value={bookSizeInput}
                  onChange={e => {
                    setBookSizeInput(e.target.value)
                    setSettingsSavedMessage(null)
                  }}
                  inputMode="decimal"
                  style={{ width: "100%", border: "none", background: "transparent", color: "hsl(var(--foreground))", fontSize: "0.9rem", outline: "none", fontWeight: 600, paddingLeft: 6 }}
                  aria-label="Book size"
                />
              </div>
              <div style={{ marginTop: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, gap: 8 }}>
                  <span className="label-text" style={{ whiteSpace: "nowrap" }}>Gross Lev.</span>
                  <span className="mono-text" style={{ fontSize: "0.78rem", fontWeight: 700, color: summary.leverage > 1.5 ? "hsl(var(--warning))" : "hsl(var(--foreground))" }}>{summary.leverage.toFixed(2)}×</span>
                </div>
                <div style={{ height: 6, borderRadius: 999, background: "hsl(var(--separator))", overflow: "hidden", display: "flex" }}>
                  <div style={{ width: `${summary.gross > 0 ? Math.min(100, (summary.longN / summary.gross) * 100 * Math.min(1, summary.leverage / 2)) : 0}%`, background: "hsl(var(--success))" }} />
                  <div style={{ width: `${summary.gross > 0 ? Math.min(100, (summary.shortN / summary.gross) * 100 * Math.min(1, summary.leverage / 2)) : 0}%`, background: "hsl(var(--destructive))" }} />
                </div>
              </div>
              <button
                type="button"
                className="theme-button-base theme-button-ghost"
                style={{ minHeight: 32, fontSize: "0.78rem", paddingInline: 10, marginTop: 8 }}
                onClick={handleSaveBookSize}
                disabled={settingsMutation.isPending}
              >
                {settingsMutation.isPending ? "Saving…" : "Save book size"}
              </button>
              {(settingsValidationError || settingsMutation.isError || settingsSavedMessage) ? (
                <p
                  style={{ marginTop: 6, fontSize: "0.72rem" }}
                  className={settingsSavedMessage && !settingsValidationError && !settingsMutation.isError ? "text-positive" : "text-negative"}
                >
                  {settingsValidationError ?? (settingsMutation.isError ? String(settingsMutation.error) : settingsSavedMessage)}
                </p>
              ) : (
                <p className="text-subtle" style={{ marginTop: 6, fontSize: "0.7rem" }}>
                  {formatBaseCurrency(MIN_BOOK_SIZE)} – {formatBaseCurrency(MAX_BOOK_SIZE)}
                </p>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <Stat label="Net Exposure" value={fmtUSD0(summary.net)} sub={fmtSignedPct(summary.netPct)} tone={summary.net < 0 ? "neg" : "pos"} />
              <Stat label="Gross" value={fmtUSD0(summary.gross)} sub={`${summary.leverage.toFixed(2)}×`} />
              <Stat label="Long" value={fmtUSD0(summary.longN)} tone="pos" />
              <Stat label="Short" value={fmtUSD0(summary.shortN)} tone="neg" />
            </div>

            <div>
              <div className="label-text" style={{ marginBottom: 10 }}>Exposure by Asset</div>
              <ExposureBar summary={summary} />
            </div>

            {tab === "Positions" ? (
              <div>
                <div className="label-text" style={{ marginBottom: 10 }}>Group Rollups</div>
                {orderedGroups.length === 0 ? (
                  <p className="text-subtle" style={{ fontSize: "0.78rem" }}>No groups yet.</p>
                ) : (
                  <div style={{ display: "grid", gap: 9 }}>
                    {orderedGroups.map(group => {
                      const members = positionRows.filter(r => groupKey(r.group_name) === group.key)
                      const net = members.reduce((s, m) => s + netNotional(m), 0)
                      const gross = members.reduce((s, m) => s + grossNotional(m), 0)
                      const wConv = gross > 0 ? members.reduce((s, m) => s + m.conviction * grossNotional(m), 0) / gross : group.conviction
                      const rounded = Math.round(wConv)
                      return (
                        <div key={group.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ width: 8, height: 8, borderRadius: 2, background: groupColor(group.key), flex: "0 0 auto" }} />
                          <span style={{ fontSize: "0.8rem", flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{group.name}</span>
                          <span className="theme-badge" style={{ background: convTint(rounded), color: convColor(rounded), borderColor: convTint(rounded, 0.3), minHeight: "1.3rem", padding: "0.1rem 0.4rem" }}>{wConv.toFixed(1)}</span>
                          <span className="mono-text" style={{ fontSize: "0.76rem", color: "hsl(var(--foreground-secondary))", minWidth: 62, textAlign: "right" }}>{fmtUSD0(net)}</span>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>

        {/* Ledger */}
        <div style={{ minWidth: 0 }}>
          {tab === "Positions" ? (
            <>
              <div className="theme-surface" style={{ padding: 14, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, flexWrap: "wrap" }}>
                <button type="button" className="theme-button-base theme-button-ghost" style={{ minHeight: 38 }} onClick={addPositionRow}>
                  <Plus size={16} /> Position
                </button>
                <button type="button" className="theme-button-base theme-button-secondary" style={{ minHeight: 38 }} onClick={addGroup}>
                  <Layers size={15} /> Group
                </button>
              </div>

              {orderedGroups.map(group => (
                <GroupBand
                  key={group.key}
                  group={group}
                  clusters={underlyingClusters.filter(cluster => clusterHasGroup(cluster, group.key))}
                  bookSize={bookSizeNum}
                  onRename={renameGroup}
                  onConvictionAll={setGroupConviction}
                  onAddToGroup={addPositionToGroup}
                  onDisband={disbandGroup}
                  rowCallbacks={positionRowCallbacks}
                  groups={orderedGroups}
                  onClusterConviction={setClusterConviction}
                  onAssignClusterGroup={assignClusterGroup}
                />
              ))}

              <SectionCard
                title="Ungrouped positions"
                icon={<span>◇</span>}
                count={ungroupedClusters.length}
                footer={(
                  <>
                    <button type="button" className="theme-button-base theme-button-ghost" style={{ minHeight: 38 }} onClick={addPositionRow}>
                      <Plus size={16} /> Add position
                    </button>
                    <button type="button" className="theme-button-base theme-button-ghost" style={{ minHeight: 38 }} onClick={addGroup}>
                      <Layers size={15} /> New group
                    </button>
                  </>
                )}
              >
                {ungroupedClusters.map(cluster => (
                  <UnderlyingBand
                    key={cluster.key}
                    cluster={cluster}
                    bookSize={bookSizeNum}
                    spine={null}
                    groups={orderedGroups}
                    rowCallbacks={positionRowCallbacks}
                    onConviction={setClusterConviction}
                    onAssignGroup={assignClusterGroup}
                  />
                ))}
              </SectionCard>

              {currentGroupState.errors.length > 0 ? (
                <div className="theme-notice theme-notice-warning" style={{ marginTop: 14 }}>{currentGroupState.errors[0]}</div>
              ) : null}
            </>
          ) : (
            <SectionCard
              title="Hedge positions"
              icon={<span>◇</span>}
              count={hedgeRows.length}
              footer={(
                <button type="button" className="theme-button-base theme-button-ghost" style={{ minHeight: 38 }} onClick={addHedgeRow}>
                  <Plus size={16} /> Add hedge
                </button>
              )}
            >
              {hedgeRows.map(row => (
                <EditorRowView
                  key={row._id}
                  row={row}
                  bookSize={bookSizeNum}
                  isHedge
                  spine={null}
                  onUpdate={updateHedgeRow}
                  onRemove={removeHedgeRow}
                />
              ))}
            </SectionCard>
          )}

          {(currentValidationError || currentMutationError) ? (
            <div className="theme-notice theme-notice-error" style={{ marginTop: 14 }}>{currentValidationError ?? currentMutationError}</div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

interface PortfolioEditorProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function PortfolioEditor({ open, onOpenChange }: PortfolioEditorProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Edit Portfolio"
      description="Stage internal portfolio or hedge changes for approval. Nothing is applied until an approval is reviewed and applied."
      maxWidth="max-w-6xl"
    >
      {open ? <PortfolioEditorPanel onCancel={() => onOpenChange(false)} /> : null}
    </Dialog>
  )
}
