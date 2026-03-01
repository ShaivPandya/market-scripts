import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runShortScreen } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"

const columns: ColumnDef[] = [
  { key: "Ticker", header: "Ticker" },
  { key: "Company", header: "Company" },
  { key: "P/B Ratio", header: "P/B Ratio", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Gross Profit ($M)", header: "Gross Profit ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Operating Income ($M)", header: "Op. Income ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Market Cap ($M)", header: "Mkt Cap ($M)", format: v => v != null ? Number(v).toFixed(0) : "N/A" },
  { key: "Net Issuance ($M)", header: "Net Issuance ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Issuance % Mkt Cap", header: "Issuance % MktCap", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A" },
]

export function ShortScreen() {
  const [pbThreshold, setPbThreshold] = useState(3.0)
  const [lossType, setLossType] = useState<"Gross Loss" | "Operating Loss">("Gross Loss")
  const [checkIssuance, setCheckIssuance] = useState(false)

  const mutation = useMutation({ mutationFn: runShortScreen })

  function handleRun() {
    mutation.mutate({ pb_threshold: pbThreshold, loss_type: lossType, check_issuance: checkIssuance })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Short Screen</h1>
      <p className="text-sm text-gray-500 mb-4">Russell 2000 short candidates — high P/B + operating losses</p>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-6 space-y-4 max-w-md">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            P/B Threshold: <strong>{pbThreshold.toFixed(1)}</strong>
          </label>
          <input
            type="range" min={3.0} max={5.0} step={0.1}
            value={pbThreshold}
            onChange={e => setPbThreshold(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-400"><span>3.0</span><span>5.0</span></div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Loss Type</label>
          <div className="flex gap-3">
            {(["Gross Loss", "Operating Loss"] as const).map(lt => (
              <label key={lt} className="flex items-center gap-1.5 text-sm cursor-pointer">
                <input type="radio" checked={lossType === lt} onChange={() => setLossType(lt)} />
                {lt}
              </label>
            ))}
          </div>
        </div>

        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input type="checkbox" checked={checkIssuance} onChange={e => setCheckIssuance(e.target.checked)} />
          High Net Equity Issuance (top quartile)
          <span className="text-xs text-gray-400">(adds time — uses SEC EDGAR)</span>
        </label>

        <button
          onClick={handleRun}
          disabled={mutation.isPending}
          className="w-full py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {mutation.isPending ? "Screening..." : "Run Screen"}
        </button>
      </div>

      {mutation.isPending && <LoadingSpinner message="Screening Russell 2000 (this may take several minutes)..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex gap-6 text-sm text-gray-600 mb-4">
            <span>Universe: <strong>{mutation.data.phase1_count ?? "—"}</strong></span>
            <span>Pass P/B + Loss: <strong>{mutation.data.phase1_pass_count ?? "—"}</strong></span>
            <span>Final candidates: <strong>{mutation.data.final_count ?? rows.length}</strong></span>
          </div>
          {rows.length > 0 ? (
            <DataTable columns={columns} rows={rows} />
          ) : (
            <p className="text-gray-400">No candidates matching criteria.</p>
          )}
        </>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure criteria above and click Run Screen.</p>
      )}
    </div>
  )
}
