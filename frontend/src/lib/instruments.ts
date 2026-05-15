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
]

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

export function inferInstrumentType(ticker: string, instrumentType?: InstrumentType | string | null): InstrumentType {
  if (instrumentType === "spot_fx") return "spot_fx"
  if (ticker.trim().toUpperCase().endsWith("=X")) return "spot_fx"
  if (ticker.trim().toUpperCase().endsWith("=F")) return "future"
  return (instrumentType as InstrumentType) ?? "security"
}

export function normalizedSymbol(value?: string | null) {
  return (value ?? "").trim().toUpperCase()
}

export function effectivePriceSymbol(row: { ticker: string; price_symbol?: string | null }) {
  return normalizedSymbol(row.price_symbol) || normalizedSymbol(row.ticker)
}

export function submissionSymbol(row: { ticker: string; price_symbol?: string | null; instrument_type?: InstrumentType | string | null }) {
  const instrumentType = inferInstrumentType(row.ticker, row.instrument_type)
  if (instrumentType === "spot_fx") {
    return canonicalSpotFxSymbol(row.price_symbol || row.ticker) || normalizedSymbol(row.price_symbol || row.ticker)
  }
  return normalizedSymbol(row.ticker)
}

export function nextContractMultiplier(
  row: {
    ticker: string
    price_symbol?: string | null
    instrument_type?: InstrumentType | string | null
    contract_multiplier?: number | null
    _contractMultiplierTouched: boolean
  },
  nextInstrumentType: InstrumentType,
  nextPriceSymbol = effectivePriceSymbol(row),
) {
  if (nextInstrumentType === "security" || nextInstrumentType === "spot_fx") return 1
  if (row._contractMultiplierTouched) return row.contract_multiplier ?? null

  const currentInstrumentType = inferInstrumentType(row.ticker, row.instrument_type)
  const currentPriceSymbol = effectivePriceSymbol(row)
  const futureSymbolChanged = currentInstrumentType === "future" && nextPriceSymbol !== currentPriceSymbol
  if (currentInstrumentType !== "future" || futureSymbolChanged || row.contract_multiplier === 1) {
    return null
  }
  return row.contract_multiplier ?? null
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
