import type { InstrumentType, PortfolioAsset } from "@/lib/api"

export const ASSET_OPTIONS: { value: PortfolioAsset; label: string }[] = [
  { value: "equity", label: "Equity" },
  { value: "commodity", label: "Commodity" },
  { value: "fx", label: "FX" },
  { value: "bond", label: "Bond" },
]

export const INSTRUMENT_TYPE_OPTIONS: { value: InstrumentType; label: string }[] = [
  { value: "security", label: "Security" },
  { value: "future", label: "Future" },
  { value: "spot_fx", label: "Spot FX" },
  { value: "option", label: "Option" },
]

export const OPTION_TYPE_OPTIONS = [
  { value: "call", label: "Call" },
  { value: "put", label: "Put" },
] as const

type ExposureDirection = "long" | "short"

function normalizedDirection(value?: string | null): ExposureDirection | null {
  const direction = String(value ?? "").trim().toLowerCase()
  if (direction === "long" || direction === "short") return direction
  return null
}

export function exposureDirection(row: {
  direction?: string | null
  option_type?: string | null
}): ExposureDirection | null {
  const direction = normalizedDirection(row.direction)
  const optionType = String(row.option_type ?? "").trim().toLowerCase()
  if (optionType === "call" || optionType === "put") {
    const legSign = direction === "short" ? -1 : 1
    const optionSign = optionType === "call" ? 1 : -1
    return legSign * optionSign > 0 ? "long" : "short"
  }
  return direction
}

const OCC_SYMBOL_RE = /^([A-Z]{1,6})(\d{6})([CP])(\d{8})$/

export interface ParsedOptionContract {
  underlying_ticker: string
  option_expiration: string
  option_type: "call" | "put"
  option_strike: number
  option_contract_symbol: string
}

export function canonicalSpotFxSymbol(value?: string | null) {
  let symbol = (value ?? "").trim().toUpperCase()
  if (!symbol) return null
  symbol = symbol.replace(/[/-]/g, "")
  if (symbol.endsWith("=X")) symbol = symbol.slice(0, -2)
  if (!/^[A-Z]{6}$/.test(symbol)) return null
  if (symbol.slice(0, 3) === symbol.slice(3, 6)) return null
  return `${symbol}=X`
}

export function spotFxCurrencies(value?: string | null) {
  const symbol = canonicalSpotFxSymbol(value)
  if (!symbol) return { fx_base_currency: null, fx_quote_currency: null }
  return {
    fx_base_currency: symbol.slice(0, 3),
    fx_quote_currency: symbol.slice(3, 6),
  }
}

function occExpirationToIso(raw: string) {
  if (/^\d{6}$/.test(raw)) {
    const year = 2000 + Number(raw.slice(0, 2))
    const month = raw.slice(2, 4)
    const day = raw.slice(4, 6)
    return `${year}-${month}-${day}`
  }
  return raw
}

export function parseOccSymbol(value?: string | null): ParsedOptionContract | null {
  const symbol = (value ?? "").trim().toUpperCase().replace(/\s/g, "")
  if (!symbol) return null
  const match = OCC_SYMBOL_RE.exec(symbol)
  if (!match) return null
  const [, underlying, expRaw, cpFlag, strikeRaw] = match
  return {
    underlying_ticker: underlying,
    option_expiration: occExpirationToIso(expRaw),
    option_type: cpFlag === "C" ? "call" : "put",
    option_strike: Number(strikeRaw) / 1000,
    option_contract_symbol: symbol,
  }
}

export function buildOptionContractSymbol(
  underlyingTicker: string,
  optionExpiration: string,
  optionType: string,
  optionStrike: number,
) {
  const underlying = underlyingTicker.trim().toUpperCase()
  const normalizedType = optionType.trim().toLowerCase()
  if (!underlying || (normalizedType !== "call" && normalizedType !== "put")) {
    return null
  }
  let expRaw = optionExpiration.trim()
  if (/^\d{4}-\d{2}-\d{2}$/.test(expRaw)) {
    const [year, month, day] = expRaw.split("-")
    expRaw = `${Number(year) % 100}${month}${day}`
  }
  if (!/^\d{6}$/.test(expRaw)) return null
  const strikeRaw = String(Math.round(optionStrike * 1000)).padStart(8, "0")
  return `${underlying}${expRaw}${normalizedType === "call" ? "C" : "P"}${strikeRaw}`
}

export function inferInstrumentType(ticker: string, instrumentType?: InstrumentType | string | null): InstrumentType {
  if (instrumentType === "spot_fx") return "spot_fx"
  if (instrumentType === "option") return "option"
  if (ticker.trim().toUpperCase().endsWith("=X")) return "spot_fx"
  if (ticker.trim().toUpperCase().endsWith("=F")) return "future"
  if (parseOccSymbol(ticker)) return "option"
  return (instrumentType as InstrumentType) ?? "security"
}

export function normalizedSymbol(value?: string | null) {
  return (value ?? "").trim().toUpperCase()
}

/** Curated single-name leveraged / inverse ETFs (traded -> underlying, factor). */
export const STATIC_ECONOMIC_EXPOSURE: Record<string, { underlying: string; factor: number }> = {
  METU: { underlying: "META", factor: 2 },
  METD: { underlying: "META", factor: -1 },
  NVDU: { underlying: "NVDA", factor: 2 },
  NVDD: { underlying: "NVDA", factor: -1 },
  AMZU: { underlying: "AMZN", factor: 2 },
  AMZD: { underlying: "AMZN", factor: -1 },
  MSFU: { underlying: "MSFT", factor: 2 },
  MSFD: { underlying: "MSFT", factor: -1 },
  AAPU: { underlying: "AAPL", factor: 2 },
  AAPD: { underlying: "AAPL", factor: -1 },
  TSLU: { underlying: "TSLA", factor: 2 },
  TSLD: { underlying: "TSLA", factor: -1 },
  GOOU: { underlying: "GOOGL", factor: 2 },
  GOOD: { underlying: "GOOGL", factor: -1 },
}

export function resolveEconomicExposure(row: {
  ticker?: string | null
  underlying_ticker?: string | null
  instrument_type?: InstrumentType | string | null
  economic_underlying_ticker?: string | null
  exposure_multiplier?: number | null
}) {
  const traded = normalizedSymbol(row.ticker)
  if (!traded) {
    return { traded_ticker: "", underlying_ticker: "", factor: 1, source: "identity" as const }
  }
  if (inferInstrumentType(row.ticker ?? "", row.instrument_type) === "option") {
    const underlying = normalizedSymbol(row.underlying_ticker) || traded
    return { traded_ticker: traded, underlying_ticker: underlying, factor: 1, source: "identity" as const }
  }
  const mapped = STATIC_ECONOMIC_EXPOSURE[traded]
  if (mapped) {
    return {
      traded_ticker: traded,
      underlying_ticker: mapped.underlying,
      factor: mapped.factor,
      source: "static" as const,
    }
  }
  const explicitUnderlying = normalizedSymbol(row.economic_underlying_ticker)
  if (explicitUnderlying) {
    const factor = row.exposure_multiplier ?? 1
    return {
      traded_ticker: traded,
      underlying_ticker: explicitUnderlying,
      factor,
      source: "static" as const,
    }
  }
  return { traded_ticker: traded, underlying_ticker: traded, factor: 1, source: "identity" as const }
}

export function exposureGroupKey(row: {
  ticker?: string | null
  underlying_ticker?: string | null
  instrument_type?: InstrumentType | string | null
  economic_underlying_ticker?: string | null
  exposure_multiplier?: number | null
}) {
  if (inferInstrumentType(row.ticker ?? "", row.instrument_type) === "option") {
    return displayTicker(row)
  }
  return resolveEconomicExposure(row).underlying_ticker
}

export function displayTicker(row: {
  ticker?: string | null
  underlying_ticker?: string | null
  instrument_type?: InstrumentType | string | null
}) {
  if (inferInstrumentType(row.ticker ?? "", row.instrument_type) === "option") {
    return normalizedSymbol(row.underlying_ticker) || normalizedSymbol(row.ticker)
  }
  return normalizedSymbol(row.ticker)
}

export function positionRowId(row: {
  ticker: string
  position_id?: string | null
  option_contract_symbol?: string | null
  price_symbol?: string | null
  instrument_type?: InstrumentType | string | null
}) {
  if (row.position_id?.trim()) return row.position_id.trim().toUpperCase()
  if (inferInstrumentType(row.ticker, row.instrument_type) === "option") {
    return normalizedSymbol(row.option_contract_symbol) || normalizedSymbol(row.price_symbol)
  }
  return normalizedSymbol(row.ticker)
}

export function effectivePriceSymbol(row: {
  ticker: string
  price_symbol?: string | null
  option_contract_symbol?: string | null
  instrument_type?: InstrumentType | string | null
}) {
  if (inferInstrumentType(row.ticker, row.instrument_type) === "option") {
    return normalizedSymbol(row.price_symbol) || normalizedSymbol(row.option_contract_symbol)
  }
  return normalizedSymbol(row.price_symbol) || normalizedSymbol(row.ticker)
}

export function hasSeparatePriceSymbol(instrumentType?: InstrumentType | string | null) {
  return instrumentType === "future" || instrumentType === "spot_fx" || instrumentType === "option"
}

export function visiblePriceSymbol(row: {
  ticker?: string | null
  price_symbol?: string | null
  option_contract_symbol?: string | null
  instrument_type?: InstrumentType | string | null
}) {
  if (!hasSeparatePriceSymbol(row.instrument_type)) return ""
  if (row.instrument_type === "option") return normalizedSymbol(row.option_contract_symbol) || normalizedSymbol(row.price_symbol)
  return normalizedSymbol(row.price_symbol)
}

export function pricingSymbolLabel(instrumentType?: InstrumentType | string | null) {
  if (instrumentType === "spot_fx") return "FX Pair"
  if (instrumentType === "option") return "Contract"
  return "Price Symbol"
}

export function submissionSymbol(row: {
  ticker: string
  price_symbol?: string | null
  option_contract_symbol?: string | null
  instrument_type?: InstrumentType | string | null
}) {
  return positionRowId(row)
}

export function nextContractMultiplier(
  row: {
    ticker: string
    price_symbol?: string | null
    option_contract_symbol?: string | null
    instrument_type?: InstrumentType | string | null
    contract_multiplier?: number | null
    _contractMultiplierTouched: boolean
  },
  nextInstrumentType: InstrumentType,
  nextPriceSymbol = effectivePriceSymbol(row),
) {
  if (nextInstrumentType === "security" || nextInstrumentType === "spot_fx") return 1
  if (nextInstrumentType === "option") return row._contractMultiplierTouched ? row.contract_multiplier ?? 100 : 100
  if (row._contractMultiplierTouched) return row.contract_multiplier ?? null

  const currentInstrumentType = inferInstrumentType(row.ticker, row.instrument_type)
  const currentPriceSymbol = effectivePriceSymbol(row)
  const futureSymbolChanged = currentInstrumentType === "future" && nextPriceSymbol !== currentPriceSymbol
  if (currentInstrumentType !== "future" || futureSymbolChanged || row.contract_multiplier === 1) {
    return null
  }
  return row.contract_multiplier ?? null
}

export function applyOptionPaste(row: {
  ticker: string
  underlying_ticker?: string | null
  option_contract_symbol?: string | null
  option_expiration?: string | null
  option_strike?: number | null
  option_type?: "call" | "put" | null
  price_symbol?: string | null
  instrument_type?: InstrumentType | null
}) {
  const parsed = parseOccSymbol(row.option_contract_symbol || row.ticker || row.price_symbol)
  if (!parsed) return row
  return {
    ...row,
    instrument_type: "option" as InstrumentType,
    ticker: parsed.underlying_ticker,
    underlying_ticker: parsed.underlying_ticker,
    option_contract_symbol: parsed.option_contract_symbol,
    option_expiration: parsed.option_expiration,
    option_strike: parsed.option_strike,
    option_type: parsed.option_type,
    price_symbol: parsed.option_contract_symbol,
    position_id: parsed.option_contract_symbol,
  }
}

export function assetLabel(value?: string | null) {
  return ASSET_OPTIONS.find(option => option.value === value)?.label ?? (value ? value.replace(/_/g, " ") : "Equity")
}

export function instrumentTypeLabel(value?: string | null) {
  return INSTRUMENT_TYPE_OPTIONS.find(option => option.value === value)?.label ?? (value ? value.replace(/_/g, " ") : "Security")
}

export function isEquitySecurity(row: { asset?: string | null; instrument_type?: string | null }) {
  return (row.asset ?? "equity") === "equity" && (row.instrument_type ?? "security") === "security"
}
