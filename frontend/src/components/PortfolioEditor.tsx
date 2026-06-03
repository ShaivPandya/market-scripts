import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2 } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput } from "@/components/shared/FormControls"
import { StagedProposalNotice } from "@/components/shared/StagedProposalNotice"
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
import {
  ASSET_OPTIONS,
  INSTRUMENT_TYPE_OPTIONS,
  OPTION_TYPE_OPTIONS,
  applyOptionPaste,
  buildOptionContractSymbol,
  canonicalSpotFxSymbol,
  displayTicker,
  effectivePriceSymbol,
  inferInstrumentType,
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
  return parts.join(" - ")
}

function positionGroupState(rows: EditorRow[]) {
  const groups = new Map<string, {
    key: string
    name: string
    conviction: number
    direction: PortfolioPosition["direction"]
    ids: string[]
    tickers: string[]
  }>()
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
    const existing = groups.get(key)
    if (!existing) {
      groups.set(key, {
        key,
        name,
        conviction,
        direction: row.direction,
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
    if (existing.direction !== row.direction) {
      errors.push(`Group ${existing.name} cannot mix ${existing.direction} and ${row.direction} positions.`)
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

  function updatePositionGroupName(id: string, value: string) {
    setPositionRows(prev => {
      const target = prev.find(row => row._id === id)
      const name = normalizeGroupName(value)
      if (!target || !name) {
        return prev.map(row => (row._id === id ? { ...row, group_name: null, group_conviction: null } : row))
      }
      const key = groupKey(name)
      const existing = prev.find(row => row._id !== id && groupKey(row.group_name) === key)
      const groupName = normalizeGroupName(existing?.group_name) ?? name
      const groupConviction = normalizeGroupConviction(existing?.group_conviction) ?? normalizeGroupConviction(target.group_conviction) ?? target.conviction
      return prev.map(row => (row._id === id ? { ...row, group_name: groupName, group_conviction: groupConviction } : row))
    })
  }

  function updatePositionGroupConviction(group: string | null | undefined, conviction: number) {
    const key = groupKey(group)
    if (!key) return
    setPositionRows(prev => prev.map(row => (
      groupKey(row.group_name) === key ? { ...row, group_conviction: conviction } : row
    )))
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
            <StagedProposalNotice proposal={lastProposal} className="mb-4">
              staged for {proposalSubjectLabel(lastProposal.entity_type)}. Review it in Workspace before app state changes.
            </StagedProposalNotice>
          )}

          {tab === "Positions" ? (
            <>
              <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: "repeat(22, minmax(0, 1fr))" }}>
                <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Type</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Asset Class</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Group</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Group Conv.</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Conviction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost / Entry</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Qty / Base Units</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Multiplier</p>
                <p className="col-span-1 text-xs font-medium text-gray-500">Contrarian</p>
                <p className="col-span-1 text-xs font-medium text-gray-500"></p>
              </div>

              <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
                {positionRows.map((row, idx) => {
                  const key = groupKey(row.group_name)
                  const previousKey = idx > 0 ? groupKey(positionRows[idx - 1]?.group_name) : null
                  const group = key ? currentGroupState.groups.get(key) : null
                  const showGroupHeader = Boolean(group && key !== previousKey)
                  const currentType = rowInstrumentType(row)
                  const isOption = currentType === "option"
                  return (
                  <div key={row._id} className="space-y-1">
                    {showGroupHeader && group && (
                      <div className="rounded-lg border border-app bg-card-muted px-3 py-2">
                        <div className="grid grid-cols-12 items-center gap-3">
                          <div className="col-span-4 min-w-0">
                            <input
                              type="text"
                              value={group.name}
                              onChange={e => {
                                const nextName = normalizeGroupName(e.target.value)
                                setPositionRows(prev => prev.map(item => (
                                  groupKey(item.group_name) === group.key
                                    ? { ...item, group_name: nextName, group_conviction: nextName ? group.conviction : null }
                                    : item
                                )))
                              }}
                              className="theme-input w-full text-sm font-medium"
                            />
                          </div>
                          <div className="col-span-4">
                            <input
                              type="range"
                              min={1}
                              max={5}
                              step={1}
                              value={group.conviction}
                              onChange={e => updatePositionGroupConviction(group.name, Number(e.target.value))}
                              className="hig-slider w-full cursor-pointer"
                              style={{ accentColor: "hsl(var(--accent))" }}
                              aria-label={`Group conviction for ${group.name}`}
                            />
                          </div>
                          <div className="col-span-4 truncate text-xs text-muted">
                            {group.conviction} · {CONVICTION_LABELS[group.conviction]} · {group.tickers.join(", ")}
                          </div>
                        </div>
                      </div>
                    )}
                  <div className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(22, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={isOption ? (row.underlying_ticker ?? row.ticker) : row.ticker}
                        onChange={e => {
                          const nextTicker = e.target.value.toUpperCase()
                          if (isOption) {
                            updatePositionRow(row._id, applyOptionPaste({
                              ...row,
                              underlying_ticker: nextTicker,
                              ticker: nextTicker,
                            }))
                            return
                          }
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
                        placeholder={isOption ? "META" : currentType === "spot_fx" ? "EURUSD=X" : currentType === "future" ? "ES=F" : "AAPL"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={currentType}
                        onChange={v => {
                          const nextInstrumentType = v as InstrumentType
                          const nextPriceSymbol = nextInstrumentType === "spot_fx"
                            ? canonicalSpotFxSymbol(effectivePriceSymbol(row)) ?? row.price_symbol
                            : row.price_symbol
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol || row.ticker) : { fx_base_currency: null, fx_quote_currency: null }
                          const nextUnderlying = normalizedSymbol(row.underlying_ticker || row.ticker)
                          updatePositionRow(row._id, {
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
                        disabled={currentType === "spot_fx"}
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

                    <div className="col-span-2">
                      <input
                        type="text"
                        value={row.group_name ?? ""}
                        onChange={e => updatePositionGroupName(row._id, e.target.value)}
                        placeholder="Optional"
                        className="theme-input w-full text-sm"
                      />
                    </div>

                    <div className="col-span-2 min-w-0">
                      <div className="flex min-w-0 flex-col justify-center gap-1 px-1">
                        <input
                          type="range"
                          min={1}
                          max={5}
                          step={1}
                          value={normalizeGroupConviction(row.group_conviction) ?? row.conviction}
                          onChange={e => updatePositionGroupConviction(row.group_name, Number(e.target.value))}
                          aria-label={`Group conviction for ${row.group_name || row.ticker || "position"}`}
                          className="hig-slider w-full min-w-0 cursor-pointer"
                          style={{ accentColor: "hsl(var(--accent))" }}
                          disabled={!normalizeGroupName(row.group_name)}
                        />
                        <span className="block truncate text-center text-[11px] leading-none text-gray-500">
                          {normalizeGroupName(row.group_name)
                            ? `${normalizeGroupConviction(row.group_conviction) ?? row.conviction} · ${CONVICTION_LABELS[normalizeGroupConviction(row.group_conviction) ?? row.conviction]}`
                            : "Ungrouped"}
                        </span>
                      </div>
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
                        placeholder={currentType === "spot_fx" ? "Entry rate" : "Optional"}
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
                        placeholder={currentType === "spot_fx" ? "Base units" : currentType === "future" || isOption ? "Contracts" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={currentType === "future" || isOption ? row.contract_multiplier ?? "" : 1}
                        onChange={e => {
                          const v = e.target.value
                          updatePositionRow(row._id, {
                            contract_multiplier: v === "" ? null : Number(v),
                            _contractMultiplierTouched: true,
                          })
                        }}
                        placeholder={currentType === "future" ? "Auto" : isOption ? "100" : "1"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                        disabled={currentType !== "future" && !isOption}
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
                    {isOption && (
                      <div
                        className="grid gap-2 items-center rounded-lg border border-app bg-card-muted px-2 py-2"
                        style={{ gridColumn: "1 / -1", gridTemplateColumns: "repeat(22, minmax(0, 1fr))" }}
                      >
                        <div className="col-span-6">
                          <label className="mb-1 block text-[11px] font-medium text-gray-500">OCC / Contract</label>
                          <input
                            type="text"
                            value={row.option_contract_symbol ?? ""}
                            onChange={e => updatePositionRow(row._id, applyOptionPaste({
                              ...row,
                              option_contract_symbol: e.target.value.toUpperCase(),
                            }))}
                            placeholder="META260116C00500000"
                            className="theme-input w-full font-mono text-sm"
                          />
                        </div>
                        <div className="col-span-4">
                          <label className="mb-1 block text-[11px] font-medium text-gray-500">Expiration</label>
                          <input
                            type="date"
                            value={row.option_expiration ?? ""}
                            onChange={e => updatePositionRow(row._id, { option_expiration: e.target.value || null })}
                            className="theme-input w-full text-sm"
                          />
                        </div>
                        <div className="col-span-3">
                          <label className="mb-1 block text-[11px] font-medium text-gray-500">Type</label>
                          <SelectInput
                            value={row.option_type ?? "call"}
                            onChange={v => updatePositionRow(row._id, { option_type: v as PortfolioPosition["option_type"] })}
                            options={[...OPTION_TYPE_OPTIONS]}
                          />
                        </div>
                        <div className="col-span-3">
                          <label className="mb-1 block text-[11px] font-medium text-gray-500">Strike</label>
                          <input
                            type="number"
                            value={row.option_strike ?? ""}
                            onChange={e => {
                              const v = e.target.value
                              updatePositionRow(row._id, { option_strike: v === "" ? null : Number(v) })
                            }}
                            placeholder="500"
                            className="theme-input w-full text-sm"
                            step="0.01"
                            min="0"
                          />
                        </div>
                        <div className="col-span-6 flex items-end">
                          <p className="pb-2 text-[11px] text-muted truncate">
                            {optionContractSymbolForRow(row) ?? "Enter fields or paste OCC symbol"}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                  </div>
                  )
                })}
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
                {hedgeRows.map(row => {
                  const hedgeType = rowInstrumentType(row)
                  const hedgeIsOption = hedgeType === "option"
                  return (
                  <div key={row._id} className="space-y-1">
                  <div className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={hedgeIsOption ? (row.underlying_ticker ?? row.ticker) : row.ticker}
                        onChange={e => {
                          const nextTicker = e.target.value.toUpperCase()
                          if (hedgeIsOption) {
                            updateHedgeRow(row._id, applyOptionPaste({
                              ...row,
                              underlying_ticker: nextTicker,
                              ticker: nextTicker,
                            }))
                            return
                          }
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
                        placeholder={hedgeIsOption ? "META" : hedgeType === "spot_fx" ? "EURUSD=X" : hedgeType === "future" ? "ES=F" : "SPY"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={hedgeType}
                        onChange={v => {
                          const nextInstrumentType = v as InstrumentType
                          const nextPriceSymbol = nextInstrumentType === "spot_fx"
                            ? canonicalSpotFxSymbol(effectivePriceSymbol(row)) ?? row.price_symbol
                            : row.price_symbol
                          const nextFx = nextInstrumentType === "spot_fx" ? spotFxCurrencies(nextPriceSymbol || row.ticker) : { fx_base_currency: null, fx_quote_currency: null }
                          const nextUnderlying = normalizedSymbol(row.underlying_ticker || row.ticker)
                          updateHedgeRow(row._id, {
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
                        disabled={hedgeType === "spot_fx"}
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
                        placeholder={hedgeType === "spot_fx" ? "Entry rate" : "Optional"}
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
                        placeholder={hedgeType === "spot_fx" ? "Base units" : hedgeType === "future" || hedgeIsOption ? "Contracts" : "Optional"}
                        className="theme-input w-full text-sm"
                        step="any"
                      />
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={hedgeType === "future" || hedgeIsOption ? row.contract_multiplier ?? "" : 1}
                        onChange={e => {
                          const v = e.target.value
                          updateHedgeRow(row._id, {
                            contract_multiplier: v === "" ? null : Number(v),
                            _contractMultiplierTouched: true,
                          })
                        }}
                        placeholder={hedgeType === "future" ? "Auto" : hedgeIsOption ? "100" : "1"}
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                        disabled={hedgeType !== "future" && !hedgeIsOption}
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
                  {hedgeIsOption && (
                    <div
                      className="grid gap-2 items-center rounded-lg border border-app bg-card-muted px-2 py-2"
                      style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}
                    >
                      <div className="col-span-5">
                        <label className="mb-1 block text-[11px] font-medium text-gray-500">OCC / Contract</label>
                        <input
                          type="text"
                          value={row.option_contract_symbol ?? ""}
                          onChange={e => updateHedgeRow(row._id, applyOptionPaste({
                            ...row,
                            option_contract_symbol: e.target.value.toUpperCase(),
                          }))}
                          placeholder="META260116C00500000"
                          className="theme-input w-full font-mono text-sm"
                        />
                      </div>
                      <div className="col-span-3">
                        <label className="mb-1 block text-[11px] font-medium text-gray-500">Expiration</label>
                        <input
                          type="date"
                          value={row.option_expiration ?? ""}
                          onChange={e => updateHedgeRow(row._id, { option_expiration: e.target.value || null })}
                          className="theme-input w-full text-sm"
                        />
                      </div>
                      <div className="col-span-2">
                        <label className="mb-1 block text-[11px] font-medium text-gray-500">Type</label>
                        <SelectInput
                          value={row.option_type ?? "call"}
                          onChange={v => updateHedgeRow(row._id, { option_type: v as HedgePosition["option_type"] })}
                          options={[...OPTION_TYPE_OPTIONS]}
                        />
                      </div>
                      <div className="col-span-2">
                        <label className="mb-1 block text-[11px] font-medium text-gray-500">Strike</label>
                        <input
                          type="number"
                          value={row.option_strike ?? ""}
                          onChange={e => {
                            const v = e.target.value
                            updateHedgeRow(row._id, { option_strike: v === "" ? null : Number(v) })
                          }}
                          placeholder="500"
                          className="theme-input w-full text-sm"
                          step="0.01"
                          min="0"
                        />
                      </div>
                      <div className="col-span-4 flex items-end">
                        <p className="pb-2 text-[11px] text-muted truncate">
                          {optionContractSymbolForRow(row) ?? "Enter fields or paste OCC symbol"}
                        </p>
                      </div>
                    </div>
                  )}
                  </div>
                )})}
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

          {tab === "Positions" && currentGroupState.errors.length > 0 && (
            <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              {currentGroupState.errors[0]}
            </div>
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
              disabled={saveDisabled}
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
