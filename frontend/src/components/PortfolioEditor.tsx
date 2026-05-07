import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2 } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput } from "@/components/shared/FormControls"
import { DecisionStateBadge, EffectScopeBadge } from "@/components/shared/DecisionStateBadge"
import {
  fetchHedgePositions,
  fetchPortfolioSettings,
  fetchPortfolioPositions,
  saveHedgePositions,
  savePortfolioPositions,
  updatePortfolioSettings,
  type HedgePosition,
  type PortfolioPosition,
  type StagedMutationResponse,
} from "@/lib/api"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"

interface EditorRow extends PortfolioPosition {
  _id: string
  _isNew: boolean
  _contractMultiplierTouched: boolean
}

interface HedgeEditorRow extends HedgePosition {
  _id: string
  _contractMultiplierTouched: boolean
}

type EditorTab = "Positions" | "Hedges"
type InstrumentType = NonNullable<PortfolioPosition["instrument_type"]>

const ASSET_OPTIONS = [
  { value: "equity", label: "Equity" },
  { value: "commodity", label: "Commodity" },
  { value: "fx", label: "FX" },
  { value: "bond", label: "Bond" },
]

const INSTRUMENT_TYPE_OPTIONS = [
  { value: "security", label: "Security" },
  { value: "future", label: "Future" },
  { value: "spot_fx", label: "Spot FX" },
]

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

function makeId() {
  return Math.random().toString(36).slice(2, 10)
}

function proposalSubjectLabel(entityType?: string | null): string {
  return String(entityType || "proposal").replace(/_/g, " ")
}

function canonicalSpotFxSymbol(value?: string | null) {
  let symbol = (value ?? "").trim().toUpperCase()
  if (!symbol) return null
  symbol = symbol.replace(/[/-]/g, "")
  if (symbol.endsWith("=X")) symbol = symbol.slice(0, -2)
  if (!/^[A-Z]{6}$/.test(symbol)) return null
  if (symbol.slice(0, 3) === symbol.slice(3, 6)) return null
  return `${symbol}=X`
}

function spotFxCurrencies(value?: string | null) {
  const symbol = canonicalSpotFxSymbol(value)
  if (!symbol) return { fx_base_currency: null, fx_quote_currency: null }
  return {
    fx_base_currency: symbol.slice(0, 3),
    fx_quote_currency: symbol.slice(3, 6),
  }
}

function inferInstrumentType(ticker: string, instrumentType?: PortfolioPosition["instrument_type"] | null): InstrumentType {
  if (instrumentType === "spot_fx") return "spot_fx"
  if (ticker.trim().toUpperCase().endsWith("=X")) return "spot_fx"
  if (ticker.trim().toUpperCase().endsWith("=F")) return "future"
  return instrumentType ?? "security"
}

function normalizedSymbol(value?: string | null) {
  return (value ?? "").trim().toUpperCase()
}

function effectivePriceSymbol(row: { ticker: string; price_symbol?: string | null }) {
  return normalizedSymbol(row.price_symbol) || normalizedSymbol(row.ticker)
}

function submissionSymbol(row: { ticker: string; price_symbol?: string | null; instrument_type?: PortfolioPosition["instrument_type"] | null }) {
  const instrumentType = inferInstrumentType(row.ticker, row.instrument_type)
  if (instrumentType === "spot_fx") {
    return canonicalSpotFxSymbol(row.price_symbol || row.ticker) || normalizedSymbol(row.price_symbol || row.ticker)
  }
  return normalizedSymbol(row.ticker)
}

function nextContractMultiplier(
  row: {
    ticker: string
    price_symbol?: string | null
    instrument_type?: PortfolioPosition["instrument_type"] | null
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

function valuationSummary(row: PortfolioPosition | HedgePosition) {
  const parts: string[] = []
  const market = [row.country, row.exchange].filter(Boolean).join(" / ")
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
  return parts.join(" - ")
}

function positionToRow(p: PortfolioPosition): EditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
  return {
    ...p,
    _id: makeId(),
    _isNew: false,
    _contractMultiplierTouched: false,
    quantity,
    shares: quantity,
    instrument_type: instrumentType,
    price_symbol: p.price_symbol ?? p.ticker,
    contract_multiplier: p.contract_multiplier ?? (instrumentType === "future" ? null : 1),
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
    _contractMultiplierTouched: false,
  }
}

function hedgeToRow(p: HedgePosition): HedgeEditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
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
    price_symbol: p.price_symbol ?? p.ticker,
    contract_multiplier: p.contract_multiplier ?? (instrumentType === "future" ? null : 1),
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

interface PortfolioEditorProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function PortfolioEditor({ open, onOpenChange }: PortfolioEditorProps) {
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
  const [lastProposal, setLastProposal] = useState<StagedMutationResponse | null>(null)

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!open) return
    setTab("Positions")
    setLoadError(null)
    setSettingsValidationError(null)
    setSettingsSavedMessage(null)
    setPositionValidationError(null)
    setHedgeValidationError(null)
    setLastProposal(null)
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
  }, [open])
  /* eslint-enable react-hooks/set-state-in-effect */

  function handleSaved(result: StagedMutationResponse) {
    setLastProposal(result)
    void invalidateApprovalSummaries(queryClient)
  }

  const positionMutation = useMutation({
    mutationFn: (positions: PortfolioPosition[]) => savePortfolioPositions(positions),
    onSuccess: handleSaved,
  })

  const hedgeMutation = useMutation({
    mutationFn: (positions: HedgePosition[]) => saveHedgePositions(positions),
    onSuccess: handleSaved,
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

  function updatePositionRow(id: string, patch: Partial<EditorRow>) {
    setPositionRows(prev => prev.map(r => (r._id === id ? { ...r, ...patch } : r)))
  }

  function updateHedgeRow(id: string, patch: Partial<HedgeEditorRow>) {
    setHedgeRows(prev => prev.map(r => (r._id === id ? { ...r, ...patch } : r)))
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

  function addHedgeRow() {
    setHedgeRows(prev => [...prev, newHedgeRow()])
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

    const tickers = positionRows.map(submissionSymbol).filter(Boolean)
    const unique = new Set(tickers)
    if (unique.size !== tickers.length) {
      setPositionValidationError("Duplicate tickers detected. Each ticker must be unique.")
      return
    }
    if (positionRows.some(r => !r.ticker.trim())) {
      setPositionValidationError("All rows must have a ticker.")
      return
    }
    if (positionRows.length === 0) {
      setPositionValidationError("At least one position is required.")
      return
    }
    if (positionRows.some(r => inferInstrumentType(r.ticker, r.instrument_type) === "spot_fx" && !canonicalSpotFxSymbol(r.price_symbol || r.ticker))) {
      setPositionValidationError("Spot FX rows must use a pair like EURUSD=X, EURUSD, EUR/USD, or EUR-USD.")
      return
    }

    const positions: PortfolioPosition[] = positionRows.map(r => {
      const instrumentType = inferInstrumentType(r.ticker, r.instrument_type)
      const ticker = instrumentType === "spot_fx" ? canonicalSpotFxSymbol(r.price_symbol || r.ticker) ?? r.ticker.trim().toUpperCase() : r.ticker.trim().toUpperCase()
      const priceSymbol = instrumentType === "spot_fx" ? ticker : (r.price_symbol?.trim() || r.ticker).toUpperCase()
      const fxCurrencies = instrumentType === "spot_fx" ? spotFxCurrencies(priceSymbol) : { fx_base_currency: r.fx_base_currency ?? null, fx_quote_currency: r.fx_quote_currency ?? null }
      return {
        ticker,
        asset: instrumentType === "spot_fx" ? "fx" : r.asset,
        direction: r.direction,
        contrarian: r.contrarian,
        conviction: r.conviction,
        cost_basis: r.cost_basis,
        shares: rowQuantity(r),
        quantity: rowQuantity(r),
        instrument_type: instrumentType,
        price_symbol: priceSymbol,
        contract_multiplier: instrumentType === "future" ? r.contract_multiplier ?? null : 1,
        fx_base_currency: fxCurrencies.fx_base_currency,
        fx_quote_currency: fxCurrencies.fx_quote_currency,
        currency: instrumentType === "spot_fx" ? fxCurrencies.fx_quote_currency : r.currency ?? null,
        country: r.country ?? null,
        exchange: instrumentType === "spot_fx" ? r.exchange ?? "FX" : r.exchange ?? null,
        base_currency: r.base_currency ?? null,
        fx_rate_to_base: r.fx_rate_to_base ?? null,
        fx_rate_as_of: r.fx_rate_as_of ?? null,
        cost_basis_base: r.cost_basis_base ?? null,
        notional_base: r.notional_base ?? null,
        valuation_status: r.valuation_status ?? null,
      }
    })

    positionMutation.mutate(positions)
  }

  function handleSaveHedges() {
    setHedgeValidationError(null)

    const tickers = hedgeRows.map(submissionSymbol).filter(Boolean)
    const unique = new Set(tickers)
    if (unique.size !== tickers.length) {
      setHedgeValidationError("Duplicate tickers detected. Each ticker must be unique.")
      return
    }
    if (hedgeRows.some(r => !r.ticker.trim())) {
      setHedgeValidationError("All hedge rows must have a ticker.")
      return
    }
    if (hedgeRows.some(r => inferInstrumentType(r.ticker, r.instrument_type) === "spot_fx" && !canonicalSpotFxSymbol(r.price_symbol || r.ticker))) {
      setHedgeValidationError("Spot FX rows must use a pair like EURUSD=X, EURUSD, EUR/USD, or EUR-USD.")
      return
    }

    const positions: HedgePosition[] = hedgeRows.map(r => {
      const instrumentType = inferInstrumentType(r.ticker, r.instrument_type)
      const ticker = instrumentType === "spot_fx" ? canonicalSpotFxSymbol(r.price_symbol || r.ticker) ?? r.ticker.trim().toUpperCase() : r.ticker.trim().toUpperCase()
      const priceSymbol = instrumentType === "spot_fx" ? ticker : (r.price_symbol?.trim() || r.ticker).toUpperCase()
      const fxCurrencies = instrumentType === "spot_fx" ? spotFxCurrencies(priceSymbol) : { fx_base_currency: r.fx_base_currency ?? null, fx_quote_currency: r.fx_quote_currency ?? null }
      return {
        ticker,
        asset: instrumentType === "spot_fx" ? "fx" : r.asset ?? "equity",
        direction: r.direction,
        cost_basis: r.cost_basis,
        shares: rowQuantity(r),
        quantity: rowQuantity(r),
        instrument_type: instrumentType,
        price_symbol: priceSymbol,
        contract_multiplier: instrumentType === "future" ? r.contract_multiplier ?? null : 1,
        fx_base_currency: fxCurrencies.fx_base_currency,
        fx_quote_currency: fxCurrencies.fx_quote_currency,
        currency: instrumentType === "spot_fx" ? fxCurrencies.fx_quote_currency : r.currency ?? null,
        country: r.country ?? null,
        exchange: instrumentType === "spot_fx" ? r.exchange ?? "FX" : r.exchange ?? null,
        base_currency: r.base_currency ?? null,
        fx_rate_to_base: r.fx_rate_to_base ?? null,
        fx_rate_as_of: r.fx_rate_as_of ?? null,
        cost_basis_base: r.cost_basis_base ?? null,
        notional_base: r.notional_base ?? null,
        valuation_status: r.valuation_status ?? null,
      }
    })

    hedgeMutation.mutate(positions)
  }

  const currentValidationError = tab === "Positions" ? positionValidationError : hedgeValidationError
  const currentMutationError = tab === "Positions"
    ? (positionMutation.isError ? String(positionMutation.error) : null)
    : (hedgeMutation.isError ? String(hedgeMutation.error) : null)
  const currentLoading = tab === "Positions" ? positionMutation.isPending : hedgeMutation.isPending
  const currentLoadingText = tab === "Positions" ? "Proposing portfolio..." : "Proposing hedges..."
  const currentSaveLabel = tab === "Positions" ? "Propose Portfolio" : "Propose Hedges"

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Edit Portfolio"
      description="Stage internal portfolio or hedge changes for approval. Nothing is applied until an approval is reviewed and applied."
      maxWidth="max-w-6xl"
    >
      {isLoading && (
        <p className="text-sm text-gray-500 py-4">Loading portfolio and hedge positions...</p>
      )}

      {loadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 mb-4">
          {loadError}
        </div>
      )}

      {!isLoading && !loadError && (
        <>
          <div className="mb-4">
            <SegmentedControl
              options={[
                { value: "Positions", label: "Positions" },
                { value: "Hedges", label: "Hedges" },
              ]}
              value={tab}
              onChange={setTab}
              size="sm"
            />
          </div>

          <div className="mb-5 border-b border-gray-200 pb-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="w-full sm:max-w-xs">
                <label className="mb-1 block text-xs font-medium text-gray-500" htmlFor="portfolio-book-size">
                  Book Size
                </label>
                <input
                  id="portfolio-book-size"
                  type="number"
                  value={bookSizeInput}
                  onChange={e => {
                    setBookSizeInput(e.target.value)
                    setSettingsSavedMessage(null)
                  }}
                  placeholder={String(DEFAULT_BOOK_SIZE)}
                  className="theme-input w-full text-sm"
                  step="1000"
                  min={MIN_BOOK_SIZE}
                  max={MAX_BOOK_SIZE}
                />
              </div>
              <ActionButton
                onClick={handleSaveBookSize}
                loading={settingsMutation.isPending}
                loadingText="Saving book size..."
                className="w-auto px-4"
              >
                Save Book Size
              </ActionButton>
              <span className="pb-2 text-xs text-gray-400">
                {formatBaseCurrency(MIN_BOOK_SIZE)} - {formatBaseCurrency(MAX_BOOK_SIZE)}
              </span>
            </div>
            {(settingsValidationError || settingsMutation.isError || settingsSavedMessage) && (
              <p
                className={`mt-2 text-xs ${
                  settingsSavedMessage && !settingsValidationError && !settingsMutation.isError
                    ? "text-emerald-700"
                    : "text-red-700"
                }`}
              >
                {settingsValidationError ?? (settingsMutation.isError ? String(settingsMutation.error) : settingsSavedMessage)}
              </p>
            )}
          </div>

          {lastProposal && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
              <div className="flex flex-wrap items-center gap-2">
                <DecisionStateBadge state={lastProposal.decision_state ?? "pending_approval"} />
                <EffectScopeBadge scope={lastProposal.effect_scope ?? "internal_state"} />
                <span>
                  Proposal #{lastProposal.approval_id} staged for {proposalSubjectLabel(lastProposal.entity_type)}. Review it in Workspace before app state changes.
                </span>
              </div>
            </div>
          )}

          {tab === "Positions" ? (
            <>
              <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: "repeat(18, minmax(0, 1fr))" }}>
                <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Type</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Asset Class</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Conviction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost / Entry</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Qty / Base Units</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Multiplier</p>
                <p className="col-span-1 text-xs font-medium text-gray-500">Contrarian</p>
                <p className="col-span-1 text-xs font-medium text-gray-500"></p>
              </div>

              <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
                {positionRows.map(row => (
                  <div key={row._id} className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(18, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={row.ticker}
                        onChange={e => {
                          const nextTicker = e.target.value.toUpperCase()
                          const currentPriceSymbol = row.price_symbol?.trim().toUpperCase()
                          const nextPriceSymbol = !currentPriceSymbol || currentPriceSymbol === row.ticker.toUpperCase()
                            ? nextTicker
                            : row.price_symbol
                          const nextInstrumentType = inferInstrumentType(nextTicker, row.instrument_type)
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol) : { fx_base_currency: null, fx_quote_currency: null }
                          updatePositionRow(row._id, {
                            ticker: nextTicker,
                            price_symbol: nextPriceSymbol,
                            instrument_type: nextInstrumentType,
                            asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
                            fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
                            fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
                            contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol)),
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "EURUSD=X" : inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "ES=F" : "AAPL"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={inferInstrumentType(row.ticker, row.instrument_type)}
                        onChange={v => {
                          const nextInstrumentType = v as InstrumentType
                          const nextPriceSymbol = nextInstrumentType === "spot_fx"
                            ? canonicalSpotFxSymbol(effectivePriceSymbol(row)) ?? row.price_symbol
                            : row.price_symbol
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol || row.ticker) : { fx_base_currency: null, fx_quote_currency: null }
                          updatePositionRow(row._id, {
                            instrument_type: nextInstrumentType,
                            price_symbol: nextPriceSymbol,
                            asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
                            fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
                            fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
                            contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol)),
                            _contractMultiplierTouched: nextInstrumentType !== "future"
                              ? false
                              : row._contractMultiplierTouched,
                          })
                        }}
                        options={INSTRUMENT_TYPE_OPTIONS}
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={row.asset}
                        onChange={v => updatePositionRow(row._id, { asset: v as PortfolioPosition["asset"] })}
                        options={ASSET_OPTIONS}
                        disabled={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx"}
                      />
                    </div>

                    <div className="col-span-2">
                      {row._isNew ? (
                        <SelectInput
                          value={row.direction}
                          onChange={v => updatePositionRow(row._id, { direction: v as PortfolioPosition["direction"] })}
                          options={DIRECTION_OPTIONS}
                        />
                      ) : (
                        <span
                          className="inline-flex w-full items-center justify-center rounded-lg border px-2 py-1.5 text-center text-xs font-medium"
                          style={
                            row.direction === "long"
                              ? {
                                  backgroundColor: "hsl(var(--success-muted))",
                                  color: "hsl(var(--success))",
                                  borderColor: "hsl(var(--success) / 0.25)",
                                }
                              : {
                                  backgroundColor: "hsl(var(--destructive-muted))",
                                  color: "hsl(var(--destructive))",
                                  borderColor: "hsl(var(--destructive) / 0.25)",
                                }
                          }
                        >
                          {row.direction}
                        </span>
                      )}
                    </div>

                    <div className="col-span-2 min-w-0">
                      <div className="flex min-w-0 flex-col justify-center gap-1 px-1">
                        <input
                          type="range"
                          min={1}
                          max={5}
                          step={1}
                          value={row.conviction}
                          onChange={e => updatePositionRow(row._id, { conviction: Number(e.target.value) })}
                          aria-label={`Conviction for ${row.ticker || "position"}`}
                          aria-valuetext={`${row.conviction} ${CONVICTION_LABELS[row.conviction] ?? ""}`}
                          className="hig-slider w-full min-w-0 cursor-pointer"
                          style={{ accentColor: "hsl(var(--accent))" }}
                        />
                        <span
                          className="block truncate text-center text-[11px] leading-none text-gray-500"
                          title={`${row.conviction} · ${CONVICTION_LABELS[row.conviction] ?? ""}`}
                        >
                          {row.conviction} · {CONVICTION_LABELS[row.conviction]}
                        </span>
                      </div>
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={row.cost_basis ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          updatePositionRow(row._id, { cost_basis: v === "" ? null : Number(v) })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "Entry rate" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="0.01"
                        min="0"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={rowQuantity(row) ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          const quantity = v === "" ? null : Number(v)
                          updatePositionRow(row._id, { shares: quantity, quantity })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "Base units" : inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Contracts" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? row.contract_multiplier ?? "" : 1}
                        onChange={e => {
                          const v = e.target.value
                          updatePositionRow(row._id, {
                            contract_multiplier: v === "" ? null : Number(v),
                            _contractMultiplierTouched: true,
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Auto" : "1"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                        disabled={inferInstrumentType(row.ticker, row.instrument_type) !== "future"}
                      />
                    </div>

                    <div className="col-span-1 flex justify-center">
                      <button
                        type="button"
                        role="switch"
                        aria-checked={row.contrarian}
                        onClick={() => updatePositionRow(row._id, { contrarian: !row.contrarian })}
                        className="relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200"
                        style={{ backgroundColor: row.contrarian ? "hsl(var(--accent))" : "hsl(var(--separator))" }}
                      >
                        <span
                          className={`pointer-events-none inline-block h-[18px] w-[18px] rounded-full shadow-sm transition-transform duration-200 mt-[2px] ${row.contrarian ? "translate-x-[20px]" : "translate-x-[2px]"}`}
                          style={{ backgroundColor: "hsl(var(--background-elevated))" }}
                        />
                      </button>
                    </div>

                    <div className="col-span-1 flex justify-center">
                      <button
                        type="button"
                        onClick={() => removePositionRow(row._id)}
                        className="rounded-md p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors"
                        aria-label="Remove position"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    {valuationSummary(row) && (
                      <div className="-mt-1 text-[11px] text-muted" style={{ gridColumn: "1 / -1" }}>
                        {valuationSummary(row)}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <button
                type="button"
                onClick={addPositionRow}
                className="mt-4 flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-gray-800 transition-colors"
              >
                <Plus size={15} />
                Add Position
              </button>
            </>
          ) : (
            <>
              <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Type</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Asset Class</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost / Entry</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Qty / Base Units</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Multiplier</p>
                <p className="col-span-2 text-xs font-medium text-gray-500"></p>
              </div>

              <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
                {hedgeRows.map(row => (
                  <div key={row._id} className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={row.ticker}
                        onChange={e => {
                          const nextTicker = e.target.value.toUpperCase()
                          const currentPriceSymbol = row.price_symbol?.trim().toUpperCase()
                          const nextPriceSymbol = !currentPriceSymbol || currentPriceSymbol === row.ticker.toUpperCase()
                            ? nextTicker
                            : row.price_symbol
                          const nextInstrumentType = inferInstrumentType(nextTicker, row.instrument_type)
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol) : { fx_base_currency: null, fx_quote_currency: null }
                          updateHedgeRow(row._id, {
                            ticker: nextTicker,
                            price_symbol: nextPriceSymbol,
                            instrument_type: nextInstrumentType,
                            asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
                            fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
                            fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
                            contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol)),
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "EURUSD=X" : inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "ES=F" : "SPY"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={inferInstrumentType(row.ticker, row.instrument_type)}
                        onChange={v => {
                          const nextInstrumentType = v as InstrumentType
                          const nextPriceSymbol = nextInstrumentType === "spot_fx"
                            ? canonicalSpotFxSymbol(effectivePriceSymbol(row)) ?? row.price_symbol
                            : row.price_symbol
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol || row.ticker) : { fx_base_currency: null, fx_quote_currency: null }
                          updateHedgeRow(row._id, {
                            instrument_type: nextInstrumentType,
                            price_symbol: nextPriceSymbol,
                            asset: nextInstrumentType === "spot_fx" ? "fx" : row.asset,
                            fx_base_currency: nextFx.fx_base_currency ?? row.fx_base_currency,
                            fx_quote_currency: nextFx.fx_quote_currency ?? row.fx_quote_currency,
                            contract_multiplier: nextContractMultiplier(row, nextInstrumentType, normalizedSymbol(nextPriceSymbol)),
                            _contractMultiplierTouched: nextInstrumentType !== "future"
                              ? false
                              : row._contractMultiplierTouched,
                          })
                        }}
                        options={INSTRUMENT_TYPE_OPTIONS}
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={row.asset ?? "equity"}
                        onChange={v => updateHedgeRow(row._id, { asset: v as HedgePosition["asset"] })}
                        options={ASSET_OPTIONS}
                        disabled={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx"}
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={row.direction}
                        onChange={v => updateHedgeRow(row._id, { direction: v as HedgePosition["direction"] })}
                        options={DIRECTION_OPTIONS}
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={row.cost_basis ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          updateHedgeRow(row._id, { cost_basis: v === "" ? null : Number(v) })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "Entry rate" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="0.01"
                        min="0"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={rowQuantity(row) ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          const quantity = v === "" ? null : Number(v)
                          updateHedgeRow(row._id, { shares: quantity, quantity })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "spot_fx" ? "Base units" : inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Contracts" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="any"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? row.contract_multiplier ?? "" : 1}
                        onChange={e => {
                          const v = e.target.value
                          updateHedgeRow(row._id, {
                            contract_multiplier: v === "" ? null : Number(v),
                            _contractMultiplierTouched: true,
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Auto" : "1"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                        disabled={inferInstrumentType(row.ticker, row.instrument_type) !== "future"}
                      />
                    </div>

                    <div className="col-span-2 flex justify-center">
                      <button
                        type="button"
                        onClick={() => removeHedgeRow(row._id)}
                        className="rounded-md p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors"
                        aria-label="Remove hedge position"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    {valuationSummary(row) && (
                      <div className="-mt-1 text-[11px] text-muted" style={{ gridColumn: "1 / -1" }}>
                        {valuationSummary(row)}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <button
                type="button"
                onClick={addHedgeRow}
                className="mt-4 flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-gray-800 transition-colors"
              >
                <Plus size={15} />
                Add Hedge
              </button>
            </>
          )}

          {(currentValidationError || currentMutationError) && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {currentValidationError ?? currentMutationError}
            </div>
          )}

          <div className="mt-6 flex justify-end gap-3">
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <ActionButton
              onClick={tab === "Positions" ? handleSavePositions : handleSaveHedges}
              loading={currentLoading}
              loadingText={currentLoadingText}
              className="w-auto px-6"
            >
              {currentSaveLabel}
            </ActionButton>
          </div>
        </>
      )}
    </Dialog>
  )
}
