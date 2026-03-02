import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runShortScreen } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SliderInput, SegmentedControl, Toggle, ActionButton, ControlPanel } from "@/components/shared/FormControls"

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
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Short Screen</h1>
        <p className="text-sm text-gray-400 mt-0.5">Russell 2000 short candidates — high P/B + operating losses</p>
      </div>

      <ControlPanel>
        <SliderInput
          label="P/B Threshold"
          value={pbThreshold}
          onChange={setPbThreshold}
          min={3.0}
          max={5.0}
          step={0.1}
          formatValue={v => v.toFixed(1)}
          minLabel="3.0"
          maxLabel="5.0"
        />

        <div>
          <label className="block text-sm text-gray-600 mb-1.5">Loss Type</label>
          <SegmentedControl
            options={[
              { value: "Gross Loss" as const, label: "Gross Loss" },
              { value: "Operating Loss" as const, label: "Operating Loss" },
            ]}
            value={lossType}
            onChange={setLossType}
          />
        </div>

        <Toggle
          label="High Net Equity Issuance (top quartile)"
          checked={checkIssuance}
          onChange={setCheckIssuance}
          description="Adds time — uses SEC EDGAR"
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Screening...">
          Run Screen
        </ActionButton>
      </ControlPanel>

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
