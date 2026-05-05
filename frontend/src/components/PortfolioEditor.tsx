import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2 } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput } from "@/components/shared/FormControls"
import { DecisionStateBadge, EffectScopeBadge } from "@/components/shared/DecisionStateBadge"
import {
  fetchHedgePositions,
  fetchPortfolioPositions,
  saveHedgePositions,
  savePortfolioPositions,
  type HedgePosition,
  type PortfolioPosition,
  type StagedMutationResponse,
} from "@/lib/api"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"

interface EditorRow extends PortfolioPosition {
  _id: string
  _isNew: boolean
}

interface HedgeEditorRow extends HedgePosition {
  _id: string
}

type EditorTab = "Positions" | "Hedges"

const ASSET_OPTIONS = [
  { value: "equity", label: "Equity" },
  { value: "commodity", label: "Commodity" },
  { value: "fx", label: "FX" },
  { value: "bond", label: "Bond" },
]

const INSTRUMENT_TYPE_OPTIONS = [
  { value: "security", label: "Security" },
  { value: "future", label: "Future" },
]

const DIRECTION_OPTIONS = [
  { value: "long", label: "Long" },
  { value: "short", label: "Short" },
]

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

function inferInstrumentType(ticker: string, instrumentType?: PortfolioPosition["instrument_type"] | null) {
  if (ticker.trim().toUpperCase().endsWith("=F")) return "future"
  return instrumentType ?? "security"
}

function rowQuantity(row: { quantity?: number | null; shares?: number | null }) {
  return row.quantity ?? row.shares ?? null
}

function positionToRow(p: PortfolioPosition): EditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
  return {
    ...p,
    _id: makeId(),
    _isNew: false,
    quantity,
    shares: quantity,
    instrument_type: instrumentType,
    price_symbol: p.price_symbol ?? p.ticker,
    contract_multiplier: p.contract_multiplier ?? (instrumentType === "security" ? 1 : null),
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
    contract_multiplier: 1,
  }
}

function hedgeToRow(p: HedgePosition): HedgeEditorRow {
  const instrumentType = inferInstrumentType(p.ticker, p.instrument_type)
  const quantity = rowQuantity(p)
  return {
    _id: makeId(),
    ticker: p.ticker,
    asset: p.asset ?? "equity",
    direction: p.direction,
    cost_basis: p.cost_basis,
    shares: quantity,
    quantity,
    instrument_type: instrumentType,
    price_symbol: p.price_symbol ?? p.ticker,
    contract_multiplier: p.contract_multiplier ?? (instrumentType === "security" ? 1 : null),
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
    contract_multiplier: 1,
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
  const [loadError, setLoadError] = useState<string | null>(null)
  const [positionValidationError, setPositionValidationError] = useState<string | null>(null)
  const [hedgeValidationError, setHedgeValidationError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastProposal, setLastProposal] = useState<StagedMutationResponse | null>(null)

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!open) return
    setTab("Positions")
    setLoadError(null)
    setPositionValidationError(null)
    setHedgeValidationError(null)
    setLastProposal(null)
    setIsLoading(true)
    Promise.all([fetchPortfolioPositions(), fetchHedgePositions()])
      .then(([portfolioData, hedgeData]) => {
        setPositionRows(portfolioData.positions.map(positionToRow))
        setHedgeRows(hedgeData.positions.map(hedgeToRow))
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

  function handleSavePositions() {
    setPositionValidationError(null)

    const tickers = positionRows.map(r => r.ticker.trim().toUpperCase()).filter(Boolean)
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

    const positions: PortfolioPosition[] = positionRows.map(r => {
      const instrumentType = inferInstrumentType(r.ticker, r.instrument_type)
      return {
        ticker: r.ticker.trim().toUpperCase(),
        asset: r.asset,
        direction: r.direction,
        contrarian: r.contrarian,
        conviction: r.conviction,
        cost_basis: r.cost_basis,
        shares: rowQuantity(r),
        quantity: rowQuantity(r),
        instrument_type: instrumentType,
        price_symbol: (r.price_symbol?.trim() || r.ticker).toUpperCase(),
        contract_multiplier: instrumentType === "future" ? r.contract_multiplier ?? null : 1,
      }
    })

    positionMutation.mutate(positions)
  }

  function handleSaveHedges() {
    setHedgeValidationError(null)

    const tickers = hedgeRows.map(r => r.ticker.trim().toUpperCase()).filter(Boolean)
    const unique = new Set(tickers)
    if (unique.size !== tickers.length) {
      setHedgeValidationError("Duplicate tickers detected. Each ticker must be unique.")
      return
    }
    if (hedgeRows.some(r => !r.ticker.trim())) {
      setHedgeValidationError("All hedge rows must have a ticker.")
      return
    }

    const positions: HedgePosition[] = hedgeRows.map(r => {
      const instrumentType = inferInstrumentType(r.ticker, r.instrument_type)
      return {
        ticker: r.ticker.trim().toUpperCase(),
        asset: r.asset ?? "equity",
        direction: r.direction,
        cost_basis: r.cost_basis,
        shares: rowQuantity(r),
        quantity: rowQuantity(r),
        instrument_type: instrumentType,
        price_symbol: (r.price_symbol?.trim() || r.ticker).toUpperCase(),
        contract_multiplier: instrumentType === "future" ? r.contract_multiplier ?? null : 1,
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

          {lastProposal && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
              <div className="flex flex-wrap items-center gap-2">
                <DecisionStateBadge state={lastProposal.decision_state ?? "pending_approval"} />
                <EffectScopeBadge scope={lastProposal.effect_scope ?? "internal_state"} />
                <span>
                  Proposal #{lastProposal.approval_id} staged for {lastProposal.action_id.replace(/_/g, " ")}. Review it in Workspace before app state changes.
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
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost Basis</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Quantity</p>
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
                          updatePositionRow(row._id, {
                            ticker: nextTicker,
                            price_symbol: !currentPriceSymbol || currentPriceSymbol === row.ticker.toUpperCase()
                              ? nextTicker
                              : row.price_symbol,
                            instrument_type: inferInstrumentType(nextTicker, row.instrument_type),
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "ES=F" : "AAPL"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={inferInstrumentType(row.ticker, row.instrument_type)}
                        onChange={v => updatePositionRow(row._id, {
                          instrument_type: v as PortfolioPosition["instrument_type"],
                          contract_multiplier: v === "security" ? 1 : row.contract_multiplier,
                        })}
                        options={INSTRUMENT_TYPE_OPTIONS}
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={row.asset}
                        onChange={v => updatePositionRow(row._id, { asset: v as PortfolioPosition["asset"] })}
                        options={ASSET_OPTIONS}
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

                    <div className="col-span-2 flex items-center gap-2">
                      <input
                        type="range"
                        min={1}
                        max={5}
                        step={1}
                        value={row.conviction}
                        onChange={e => updatePositionRow(row._id, { conviction: Number(e.target.value) })}
                        className="hig-slider w-full cursor-pointer"
                        style={{ accentColor: "hsl(var(--accent))" }}
                      />
                      <span className="text-xs text-gray-500 whitespace-nowrap w-24 shrink-0">
                        {row.conviction} · {CONVICTION_LABELS[row.conviction]}
                      </span>
                    </div>

                    <div className="col-span-2">
                      <input
                        type="number"
                        value={row.cost_basis ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          updatePositionRow(row._id, { cost_basis: v === "" ? null : Number(v) })
                        }}
                        placeholder="Optional"
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
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Contracts" : "Optional"}
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
                          updatePositionRow(row._id, { contract_multiplier: v === "" ? null : Number(v) })
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
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost Basis</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Quantity</p>
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
                          updateHedgeRow(row._id, {
                            ticker: nextTicker,
                            price_symbol: !currentPriceSymbol || currentPriceSymbol === row.ticker.toUpperCase()
                              ? nextTicker
                              : row.price_symbol,
                            instrument_type: inferInstrumentType(nextTicker, row.instrument_type),
                          })
                        }}
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "ES=F" : "SPY"}
                        className="theme-input w-full font-mono text-sm"
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={inferInstrumentType(row.ticker, row.instrument_type)}
                        onChange={v => updateHedgeRow(row._id, {
                          instrument_type: v as HedgePosition["instrument_type"],
                          contract_multiplier: v === "security" ? 1 : row.contract_multiplier,
                        })}
                        options={INSTRUMENT_TYPE_OPTIONS}
                      />
                    </div>

                    <div className="col-span-2">
                      <SelectInput
                        value={row.asset ?? "equity"}
                        onChange={v => updateHedgeRow(row._id, { asset: v as HedgePosition["asset"] })}
                        options={ASSET_OPTIONS}
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
                        placeholder="Optional"
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
                        placeholder={inferInstrumentType(row.ticker, row.instrument_type) === "future" ? "Contracts" : "Optional"}
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
                          updateHedgeRow(row._id, { contract_multiplier: v === "" ? null : Number(v) })
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
