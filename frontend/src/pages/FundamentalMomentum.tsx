import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runFundamentalMomentum } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, SelectInput, TextInput, ActionButton, ControlPanel } from "@/components/shared/FormControls"
import { colorPositiveNegative, colorZscore } from "@/lib/colors"

const UNIVERSE_OPTIONS = [
  "S&P 500", "Russell 2000", "S&P 400",
  "XLB — Materials", "XLC — Communication Services", "XLE — Energy",
  "XLF — Financials", "XLI — Industrials", "XLK — Technology",
  "XLP — Consumer Staples", "XLRE — Real Estate", "XLU — Utilities",
  "XLV — Health Care", "XLY — Consumer Discretionary",
]

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).map(k => ({
    key: k,
    header: k,
    colorFn: k.toLowerCase().includes("z") || k.toLowerCase().includes("score")
      ? colorZscore
      : k.toLowerCase().includes("pct") || k.toLowerCase().includes("yoy") || k.toLowerCase().includes("cagr")
        ? colorPositiveNegative
        : undefined,
    format: (v: unknown) =>
      v != null && typeof v === "number"
        ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}`
        : String(v ?? "N/A"),
  }))
}

export function FundamentalMomentum() {
  const [screenType, setScreenType] = useState<"EPS" | "Revenue" | "Both">("Both")
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("S&P 500")

  const mutation = useMutation({ mutationFn: runFundamentalMomentum })

  function handleRun() {
    mutation.mutate({ screen_type: screenType, universe, tickers, benchmark, input_mode: inputMode })
  }

  const epsRows: Record<string, unknown>[] = mutation.data?.eps?.results_df ?? []
  const revRows: Record<string, unknown>[] = mutation.data?.rev?.results_df ?? []

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Fundamental Momentum</h1>
      </div>

      <ControlPanel maxWidth="max-w-lg">
        <div>
          <label className="block text-sm text-gray-600 mb-1.5">Screen Type</label>
          <SegmentedControl
            options={[
              { value: "EPS" as const, label: "EPS" },
              { value: "Revenue" as const, label: "Revenue" },
              { value: "Both" as const, label: "Both" },
            ]}
            value={screenType}
            onChange={setScreenType}
          />
        </div>

        <SegmentedControl
          options={[
            { value: "Universe" as const, label: "Universe" },
            { value: "Custom Tickers" as const, label: "Custom Tickers" },
          ]}
          value={inputMode}
          onChange={setInputMode}
        />

        {inputMode === "Universe" ? (
          <SelectInput
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
          />
        )}

        <SelectInput
          label="Benchmark"
          value={benchmark}
          onChange={setBenchmark}
          options={[
            "S&P 500", "Same as Input",
            "XLB — Materials", "XLC — Communication Services", "XLE — Energy",
            "XLF — Financials", "XLI — Industrials", "XLK — Technology",
            "XLP — Consumer Staples", "XLRE — Real Estate", "XLU — Utilities",
            "XLV — Health Care", "XLY — Consumer Discretionary",
          ].map(o => ({ value: o, label: o }))}
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Screening...">
          Run Screen
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && <LoadingSpinner message="Running fundamental momentum screen..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <div className="space-y-6">
          {epsRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">EPS Momentum</h2>
              <DataTable columns={buildCols(epsRows)} rows={epsRows} />
            </div>
          )}
          {revRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Revenue Momentum</h2>
              <DataTable columns={buildCols(revRows)} rows={revRows} />
            </div>
          )}
          {epsRows.length === 0 && revRows.length === 0 && (
            <p className="text-gray-400">No results returned.</p>
          )}
        </div>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure inputs above and click Run Screen.</p>
      )}
    </div>
  )
}
