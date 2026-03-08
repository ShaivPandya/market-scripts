import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2 } from "lucide-react"
import { Dialog } from "@/components/shared/Dialog"
import { SelectInput, TextInput, Toggle, ActionButton } from "@/components/shared/FormControls"
import { fetchPortfolioPositions, savePortfolioPositions, type PortfolioPosition } from "@/lib/api"

interface EditorRow extends PortfolioPosition {
  _id: string
  _isNew: boolean
}

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
    distressed: false,
    conviction: 3,
    cost_basis: null,
  }
}

interface PortfolioEditorProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function PortfolioEditor({ open, onOpenChange }: PortfolioEditorProps) {
  const queryClient = useQueryClient()
  const [rows, setRows] = useState<EditorRow[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (!open) return
    setLoadError(null)
    setValidationError(null)
    setIsLoading(true)
    fetchPortfolioPositions()
      .then(data => setRows(data.positions.map(positionToRow)))
      .catch(err => setLoadError(String(err)))
      .finally(() => setIsLoading(false))
  }, [open])

  const mutation = useMutation({
    mutationFn: (positions: PortfolioPosition[]) => savePortfolioPositions(positions),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio", "all_timeframes"] })
      onOpenChange(false)
    },
  })

  function updateRow(id: string, patch: Partial<EditorRow>) {
    setRows(prev => prev.map(r => (r._id === id ? { ...r, ...patch } : r)))
  }

  function removeRow(id: string) {
    setRows(prev => prev.filter(r => r._id !== id))
  }

  function addRow() {
    setRows(prev => [...prev, newRow()])
  }

  function handleSave() {
    setValidationError(null)

    const tickers = rows.map(r => r.ticker.trim().toUpperCase()).filter(Boolean)
    const unique = new Set(tickers)
    if (unique.size !== tickers.length) {
      setValidationError("Duplicate tickers detected. Each ticker must be unique.")
      return
    }
    if (rows.some(r => !r.ticker.trim())) {
      setValidationError("All rows must have a ticker.")
      return
    }
    if (rows.length === 0) {
      setValidationError("At least one position is required.")
      return
    }

    const positions: PortfolioPosition[] = rows.map(r => ({
      ticker: r.ticker.trim().toUpperCase(),
      asset: r.asset,
      direction: r.direction,
      distressed: r.distressed,
      conviction: r.conviction,
      cost_basis: r.cost_basis,
    }))

    mutation.mutate(positions)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Edit Portfolio"
      description="Add or remove positions and update their attributes."
      maxWidth="max-w-5xl"
    >
      {isLoading && (
        <p className="text-sm text-gray-500 py-4">Loading positions...</p>
      )}

      {loadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 mb-4">
          {loadError}
        </div>
      )}

      {!isLoading && !loadError && (
        <>
          {/* Column headers */}
          <div className="grid grid-cols-12 gap-2 mb-2 px-1">
            <p className="col-span-2 text-xs font-medium text-gray-500">Ticker</p>
            <p className="col-span-2 text-xs font-medium text-gray-500">Asset Class</p>
            <p className="col-span-2 text-xs font-medium text-gray-500">Direction</p>
            <p className="col-span-2 text-xs font-medium text-gray-500">Conviction</p>
            <p className="col-span-2 text-xs font-medium text-gray-500">Cost Basis</p>
            <p className="col-span-1 text-xs font-medium text-gray-500">Distressed</p>
            <p className="col-span-1 text-xs font-medium text-gray-500"></p>
          </div>

          <div className="space-y-2 max-h-[50vh] overflow-y-auto pr-1">
            {rows.map(row => (
              <div key={row._id} className="grid grid-cols-12 gap-2 items-center">
                {/* Ticker */}
                <div className="col-span-2">
                  <input
                    type="text"
                    value={row.ticker}
                    onChange={e => updateRow(row._id, { ticker: e.target.value.toUpperCase() })}
                    placeholder="AAPL"
                    className="theme-input w-full font-mono text-sm"
                  />
                </div>

                {/* Asset class */}
                <div className="col-span-2">
                  <SelectInput
                    value={row.asset}
                    onChange={v => updateRow(row._id, { asset: v as PortfolioPosition["asset"] })}
                    options={ASSET_OPTIONS}
                  />
                </div>

                {/* Direction — locked for existing, editable for new */}
                <div className="col-span-2">
                  {row._isNew ? (
                    <SelectInput
                      value={row.direction}
                      onChange={v => updateRow(row._id, { direction: v as PortfolioPosition["direction"] })}
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

                {/* Conviction */}
                <div className="col-span-2 flex items-center gap-2">
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={row.conviction}
                    onChange={e => updateRow(row._id, { conviction: Number(e.target.value) })}
                    className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[hsl(var(--muted-3))]"
                    style={{ accentColor: "hsl(var(--accent))" }}
                  />
                  <span className="text-xs text-gray-500 whitespace-nowrap w-14 shrink-0">
                    {row.conviction} · {CONVICTION_LABELS[row.conviction]}
                  </span>
                </div>

                {/* Cost basis */}
                <div className="col-span-2">
                  <input
                    type="number"
                    value={row.cost_basis ?? ""}
                    onChange={e => {
                      const v = e.target.value
                      updateRow(row._id, { cost_basis: v === "" ? null : Number(v) })
                    }}
                    placeholder="Optional"
                    className="theme-input w-full text-sm"
                    step="0.01"
                    min="0"
                  />
                </div>

                {/* Distressed toggle */}
                <div className="col-span-1 flex justify-center">
                  <button
                    type="button"
                    role="switch"
                    aria-checked={row.distressed}
                    onClick={() => updateRow(row._id, { distressed: !row.distressed })}
                    className="relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200"
                    style={{ backgroundColor: row.distressed ? "hsl(var(--accent))" : "hsl(var(--muted-3))" }}
                  >
                    <span
                      className={`pointer-events-none inline-block h-[18px] w-[18px] rounded-full shadow-sm transition-transform duration-200 mt-[2px] ${row.distressed ? "translate-x-[20px]" : "translate-x-[2px]"}`}
                      style={{ backgroundColor: "hsl(var(--card))" }}
                    />
                  </button>
                </div>

                {/* Delete */}
                <div className="col-span-1 flex justify-center">
                  <button
                    type="button"
                    onClick={() => removeRow(row._id)}
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
            onClick={addRow}
            className="mt-4 flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-gray-800 transition-colors"
          >
            <Plus size={15} />
            Add Position
          </button>

          {(validationError || mutation.isError) && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {validationError ?? String(mutation.error)}
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
              onClick={handleSave}
              loading={mutation.isPending}
              loadingText="Saving..."
              className="w-auto px-6"
            >
              Save Portfolio
            </ActionButton>
          </div>
        </>
      )}
    </Dialog>
  )
}
