import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2 } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput } from "@/components/shared/FormControls"
import {
  fetchHedgePositions,
  fetchPortfolioPositions,
  saveHedgePositions,
  savePortfolioPositions,
  type HedgePosition,
  type PortfolioPosition,
} from "@/lib/api"

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

function positionToRow(p: PortfolioPosition): EditorRow {
  return { ...p, _id: makeId(), _isNew: false }
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
  }
}

function hedgeToRow(p: HedgePosition): HedgeEditorRow {
  return {
    _id: makeId(),
    ticker: p.ticker,
    direction: p.direction,
    cost_basis: p.cost_basis,
    shares: p.shares,
  }
}

function newHedgeRow(): HedgeEditorRow {
  return {
    _id: makeId(),
    ticker: "",
    direction: "short",
    cost_basis: null,
    shares: null,
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

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!open) return
    setTab("Positions")
    setLoadError(null)
    setPositionValidationError(null)
    setHedgeValidationError(null)
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

  function handleSaved() {
    queryClient.invalidateQueries({ queryKey: ["portfolio", "all_timeframes"] })
    onOpenChange(false)
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

    const positions: PortfolioPosition[] = positionRows.map(r => ({
      ticker: r.ticker.trim().toUpperCase(),
      asset: r.asset,
      direction: r.direction,
      contrarian: r.contrarian,
      conviction: r.conviction,
      cost_basis: r.cost_basis,
      shares: r.shares,
    }))

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

    const positions: HedgePosition[] = hedgeRows.map(r => ({
      ticker: r.ticker.trim().toUpperCase(),
      direction: r.direction,
      cost_basis: r.cost_basis,
      shares: r.shares,
    }))

    hedgeMutation.mutate(positions)
  }

  const currentValidationError = tab === "Positions" ? positionValidationError : hedgeValidationError
  const currentMutationError = tab === "Positions"
    ? (positionMutation.isError ? String(positionMutation.error) : null)
    : (hedgeMutation.isError ? String(hedgeMutation.error) : null)
  const currentLoading = tab === "Positions" ? positionMutation.isPending : hedgeMutation.isPending
  const currentLoadingText = tab === "Positions" ? "Saving portfolio..." : "Saving hedges..."
  const currentSaveLabel = tab === "Positions" ? "Save Portfolio" : "Save Hedges"

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Edit Portfolio"
      description="Manage portfolio positions and hedge positions."
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

          {tab === "Positions" ? (
            <>
              <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: "repeat(14, minmax(0, 1fr))" }}>
                <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Asset Class</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Conviction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost Basis</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Shares</p>
                <p className="col-span-1 text-xs font-medium text-gray-500">Contrarian</p>
                <p className="col-span-1 text-xs font-medium text-gray-500"></p>
              </div>

              <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
                {positionRows.map(row => (
                  <div key={row._id} className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(14, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={row.ticker}
                        onChange={e => updatePositionRow(row._id, { ticker: e.target.value.toUpperCase() })}
                        placeholder="AAPL"
                        className="theme-input w-full font-mono text-sm"
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
                                  backgroundColor: "hsl(var(--success-soft))",
                                  color: "hsl(var(--success-soft-foreground))",
                                  borderColor: "hsl(var(--success-border))",
                                }
                              : {
                                  backgroundColor: "hsl(var(--danger-soft))",
                                  color: "hsl(var(--danger-soft-foreground))",
                                  borderColor: "hsl(var(--danger-border))",
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
                        className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[hsl(var(--muted-3))]"
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
                        value={row.shares ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          updatePositionRow(row._id, { shares: v === "" ? null : Number(v) })
                        }}
                        placeholder="Optional"
                        className="theme-input w-full text-sm"
                        step="any"
                        min="0"
                      />
                    </div>

                    <div className="col-span-1 flex justify-center">
                      <button
                        type="button"
                        role="switch"
                        aria-checked={row.contrarian}
                        onClick={() => updatePositionRow(row._id, { contrarian: !row.contrarian })}
                        className="relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200"
                        style={{ backgroundColor: row.contrarian ? "hsl(var(--accent))" : "hsl(var(--muted-3))" }}
                      >
                        <span
                          className={`pointer-events-none inline-block h-[18px] w-[18px] rounded-full shadow-sm transition-transform duration-200 mt-[2px] ${row.contrarian ? "translate-x-[20px]" : "translate-x-[2px]"}`}
                          style={{ backgroundColor: "hsl(var(--card))" }}
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
              <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}>
                <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
                <p className="col-span-2 text-xs font-medium text-gray-500">Cost Basis</p>
                <p className="col-span-1 text-xs font-medium text-gray-500">Shares</p>
                <p className="col-span-1 text-xs font-medium text-gray-500"></p>
              </div>

              <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
                {hedgeRows.map(row => (
                  <div key={row._id} className="grid gap-2 items-center" style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={row.ticker}
                        onChange={e => updateHedgeRow(row._id, { ticker: e.target.value.toUpperCase() })}
                        placeholder="SPY"
                        className="theme-input w-full font-mono text-sm"
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

                    <div className="col-span-1">
                      <input
                        type="number"
                        value={row.shares ?? ""}
                        onChange={e => {
                          const v = e.target.value
                          updateHedgeRow(row._id, { shares: v === "" ? null : Number(v) })
                        }}
                        placeholder="Optional"
                        className="theme-input w-full text-sm"
                        step="any"
                      />
                    </div>

                    <div className="col-span-1 flex justify-center">
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
