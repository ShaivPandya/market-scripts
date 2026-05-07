import { useEffect, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import {
  runQualityScreen,
  runShortScreen,
  runLongScreen,
  runFundamentalMomentum,
  runPriceMomentum,
  type QualityScreenRequest,
  type LongScreenRequest,
  type PriceMomentumRequest,
  type ScreenJobProgress,
  type ShortScreenRequest,
} from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import {
  SegmentedControl, SelectInput, TextInput, SliderInput,
  Toggle, ActionButton, ControlPanel,
} from "@/components/shared/FormControls"
import { colorPositiveNegative, colorZscore } from "@/lib/colors"

const UNIVERSE_OPTIONS = [
  "S&P 500", "Russell 2000", "S&P 400",
  "VAW — Materials", "VOX — Communication Services", "VDE — Energy",
  "VFH — Financials", "VIS — Industrials", "VGT — Technology",
  "VDC — Consumer Staples", "VNQ — Real Estate", "VPU — Utilities",
  "VHT — Health Care", "VCR — Consumer Discretionary",
]

const BENCHMARK_OPTIONS = [
  "Same as Input", "S&P 500",
  "VAW — Materials", "VOX — Communication Services", "VDE — Energy",
  "VFH — Financials", "VIS — Industrials", "VGT — Technology",
  "VDC — Consumer Staples", "VNQ — Real Estate", "VPU — Utilities",
  "VHT — Health Care", "VCR — Consumer Discretionary",
]

const PRICE_MOMENTUM_BENCHMARK_OPTIONS = [
  "Same as Input", "S&P 500", "Russell 2000", "S&P 400",
  "VAW — Materials", "VOX — Communication Services", "VDE — Energy",
  "VFH — Financials", "VIS — Industrials", "VGT — Technology",
  "VDC — Consumer Staples", "VNQ — Real Estate", "VPU — Utilities",
  "VHT — Health Care", "VCR — Consumer Discretionary",
  "SPY", "IWM", "QQQ",
]

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function formatHeader(key: string): string {
  const map: Record<string, string> = {
    index: "Ticker", eps: "EPS", rev: "Revenue", yoy: "YoY",
    cagr: "CAGR", pct: "%", z: "Z", roe: "ROE", roa: "ROA", fcf: "FCF",
  }
  return key.split("_").map(w => map[w.toLowerCase()] ?? (w.charAt(0).toUpperCase() + w.slice(1))).join(" ")
}

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).map(k => ({
    key: k,
    header: formatHeader(k),
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

function formatSignedPercent(v: unknown): string {
  const n = Number(v)
  return Number.isFinite(n) ? `${n >= 0 ? "+" : ""}${n.toFixed(2)}%` : "N/A"
}

function formatPlainNumber(v: unknown, digits = 2): string {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(digits) : "N/A"
}

function formatScreenProgress(progress: ScreenJobProgress | undefined, fallback: string): string {
  if (!progress?.phase) return fallback
  const labels: Record<string, string> = {
    queued: "Queued",
    prices: "Screening prices",
    quality: "Scoring quality",
    fundamentals: "Fetching fundamentals",
    issuance: "Checking issuance",
    finalizing: "Finalizing",
    done: "Complete",
  }
  const label = labels[progress.phase] ?? progress.phase
  const done = typeof progress.done === "number" ? progress.done : null
  const total = typeof progress.total === "number" ? progress.total : null
  if (done != null && total != null && total > 0) return `${label} ${done}/${total}`
  return label
}

function FailedTickerNotice({ failed }: { failed: unknown }) {
  if (!Array.isArray(failed) || failed.length === 0) return null
  const tickers = failed.filter(x => typeof x === "string") as string[]
  if (tickers.length === 0) return null
  const shown = tickers.slice(0, 20).join(", ")
  const extra = tickers.length > 20 ? `, and ${tickers.length - 20} more` : ""
  return (
    <div className="mb-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
      <strong>Some tickers failed:</strong> {shown}{extra}
    </div>
  )
}

/* ─── Quality Screen ─── */

function QualityPanel() {
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("Same as Input")
  const [progress, setProgress] = useState<ScreenJobProgress | undefined>(undefined)

  const mutation = useMutation({
    mutationFn: (body: QualityScreenRequest) => runQualityScreen(body, setProgress),
  })

  useEffect(() => {
    mutation.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleRun() {
    setProgress({ phase: "queued", done: 0, total: 0 })
    mutation.mutate({ universe, tickers, benchmark, input_mode: inputMode })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []
  const columns: ColumnDef[] = rows.length > 0
    ? Object.keys(rows[0]).map(k => ({
        key: k,
        header: formatHeader(k),
        colorFn: k.toLowerCase().includes("z") || k.toLowerCase().includes("score")
          ? colorZscore
          : k.toLowerCase().includes("pct") ? colorPositiveNegative : undefined,
        format: (v: unknown) => v != null ? (typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(2) : String(v)) : "N/A",
      }))
    : []

  return (
    <>
      <ControlPanel maxWidth="max-w-lg">
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
            label="Universe"
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            label="Tickers"
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
            uppercase
          />
        )}

        <SelectInput
          label="Benchmark"
          value={benchmark}
          onChange={setBenchmark}
          options={BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText={`Screening ${inputMode === "Custom Tickers" ? "custom tickers" : universe}...`}>
          Run Screen
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && (
        <LoadingSpinner
          message={formatScreenProgress(
            progress,
            `Scoring quality for ${inputMode === "Custom Tickers" ? "custom tickers" : universe} (this may take several minutes)...`,
          )}
        />
      )}
      {mutation.isError && <ErrorMessage message={errorMessage(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex gap-6 text-sm text-gray-600 mb-4">
            <span>Input: <strong>{mutation.data.input_count ?? "—"}</strong></span>
            <span>Universe size: <strong>{mutation.data.universe_size ?? "—"}</strong></span>
            <span>Scored: <strong>{mutation.data.scored_count ?? rows.length}</strong></span>
            <span>Benchmark: <strong>{mutation.data.benchmark_name ?? benchmark}</strong></span>
          </div>
          {Array.isArray(mutation.data.failed) && mutation.data.failed.length > 0 && (
            <div className="mb-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <strong>Some tickers failed:</strong> {mutation.data.failed.join(", ")}
            </div>
          )}
          {rows.length > 0 ? (
            <DataTable columns={columns} rows={rows} />
          ) : (
            <p className="text-gray-400">No results returned.</p>
          )}
        </>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure inputs above and click Run Screen.</p>
      )}
    </>
  )
}

/* ─── Short Screen ─── */

const SHORT_REL_BENCHMARK_OPTIONS = ["IWM", "SPY", "QQQ", "Same as Input"]

const SHORT_BASE_COLUMNS: ColumnDef[] = [
  { key: "Ticker", header: "Ticker" },
  { key: "Company", header: "Company" },
  { key: "P/B Ratio", header: "P/B Ratio", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Gross Profit ($M)", header: "Gross Profit ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Operating Income ($M)", header: "Op. Income ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  { key: "Market Cap ($M)", header: "Mkt Cap ($M)", format: v => v != null ? Number(v).toFixed(0) : "N/A" },
]

const SHORT_EXTRA_COLUMNS: Record<string, ColumnDef> = {
  "Net Issuance ($M)": { key: "Net Issuance ($M)", header: "Net Issuance ($M)", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
  "Issuance % Mkt Cap": { key: "Issuance % Mkt Cap", header: "Issuance % MktCap", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A" },
  "52w Return (%)": { key: "52w Return (%)", header: "52w Ret (%)", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "Drawdown (%)": { key: "Drawdown (%)", header: "DD from High (%)", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "3m Return (%)": { key: "3m Return (%)", header: "3m Ret (%)", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "2m Rel Return (%)": { key: "2m Rel Return (%)", header: "2m Rel (%)", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "Rev YoY Q0 (%)": { key: "Rev YoY Q0 (%)", header: "Rev Q0 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "Rev YoY Q1 (%)": { key: "Rev YoY Q1 (%)", header: "Rev Q1 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "Rev YoY Q2 (%)": { key: "Rev YoY Q2 (%)", header: "Rev Q2 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "EPS YoY Q0 (%)": { key: "EPS YoY Q0 (%)", header: "EPS Q0 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "EPS YoY Q1 (%)": { key: "EPS YoY Q1 (%)", header: "EPS Q1 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
  "EPS YoY Q2 (%)": { key: "EPS YoY Q2 (%)", header: "EPS Q2 YoY", format: v => v != null ? `${Number(v).toFixed(1)}%` : "N/A", colorFn: colorPositiveNegative },
}

function buildShortColumns(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return SHORT_BASE_COLUMNS
  const firstRow = rows[0]
  const cols = [...SHORT_BASE_COLUMNS]
  for (const [key, col] of Object.entries(SHORT_EXTRA_COLUMNS)) {
    if (key in firstRow) cols.push(col)
  }
  return cols
}

function ShortPanel() {
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("Russell 2000")
  const [tickers, setTickers] = useState("")
  const [checkPb, setCheckPb] = useState(false)
  const [pbThreshold, setPbThreshold] = useState(3.0)
  const [checkLoss, setCheckLoss] = useState(false)
  const [lossType, setLossType] = useState<"Gross Loss" | "Operating Loss">("Gross Loss")
  const [checkIssuance, setCheckIssuance] = useState(false)

  // Fundamental growth filters
  const [checkRevenue, setCheckRevenue] = useState(false)
  const [maxRevenueGrowth, setMaxRevenueGrowth] = useState(0)
  const [checkEps, setCheckEps] = useState(false)
  const [maxEpsGrowth, setMaxEpsGrowth] = useState(0)

  // Price filters
  const [check52wPositive, setCheck52wPositive] = useState(false)
  const [checkMinDrawdown, setCheckMinDrawdown] = useState(false)
  const [minDrawdownPct, setMinDrawdownPct] = useState(25)
  const [checkMaxDrawdown, setCheckMaxDrawdown] = useState(false)
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(60)
  const [check3mNegMomentum, setCheck3mNegMomentum] = useState(false)
  const [check2mNegRelMomentum, setCheck2mNegRelMomentum] = useState(false)
  const [relMomentumBenchmark, setRelMomentumBenchmark] = useState("Same as Input")
  const [progress, setProgress] = useState<ScreenJobProgress | undefined>(undefined)

  const mutation = useMutation({
    mutationFn: (body: ShortScreenRequest) => runShortScreen(body, setProgress),
  })

  function handleRun() {
    setProgress({ phase: "queued", done: 0, total: 0 })
    mutation.mutate({
      input_mode: inputMode,
      universe,
      tickers,
      pb_threshold: checkPb ? pbThreshold : null,
      loss_type: checkLoss ? lossType : null,
      check_issuance: checkIssuance,
      check_revenue: checkRevenue,
      max_revenue_growth: maxRevenueGrowth,
      check_eps: checkEps,
      max_eps_growth: maxEpsGrowth,
      check_52w_positive: check52wPositive,
      check_min_drawdown: checkMinDrawdown,
      min_drawdown_pct: minDrawdownPct,
      check_max_drawdown: checkMaxDrawdown,
      max_drawdown_pct: maxDrawdownPct,
      check_3m_neg_momentum: check3mNegMomentum,
      check_2m_neg_rel_momentum: check2mNegRelMomentum,
      rel_momentum_benchmark: relMomentumBenchmark,
    })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []
  const columns = buildShortColumns(rows)

  return (
    <>
      <ControlPanel>
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
            label="Universe"
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            label="Tickers"
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
            uppercase
          />
        )}

        <Toggle
          label="P/B Threshold"
          checked={checkPb}
          onChange={setCheckPb}
        />
        {checkPb && (
          <SliderInput
            label="P/B Min"
            value={pbThreshold}
            onChange={setPbThreshold}
            min={3.0}
            max={5.0}
            step={0.1}
            formatValue={v => v.toFixed(1)}
            minLabel="3.0"
            maxLabel="5.0"
          />
        )}

        <Toggle
          label="Loss Type Filter"
          checked={checkLoss}
          onChange={setCheckLoss}
        />
        {checkLoss && (
          <div>
            <SegmentedControl
              options={[
                { value: "Gross Loss" as const, label: "Gross Loss" },
                { value: "Operating Loss" as const, label: "Operating Loss" },
              ]}
              value={lossType}
              onChange={setLossType}
            />
          </div>
        )}

        <Toggle
          label="High Net Equity Issuance (top quartile)"
          checked={checkIssuance}
          onChange={setCheckIssuance}
          description="Adds time — uses SEC EDGAR"
        />

        <div className="pt-2 border-t border-gray-100">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Fundamental Growth Filters</h3>
          <div className="space-y-3">
            <Toggle
              label="Max YoY Revenue Growth (avg of 3 quarters)"
              checked={checkRevenue}
              onChange={setCheckRevenue}
            />
            {checkRevenue && (
              <SliderInput
                label="Max Rev Growth (%)"
                value={maxRevenueGrowth}
                onChange={setMaxRevenueGrowth}
                min={-50}
                max={50}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="-50%"
                maxLabel="50%"
              />
            )}

            <Toggle
              label="Max YoY EPS Growth (avg of 3 quarters)"
              checked={checkEps}
              onChange={setCheckEps}
            />
            {checkEps && (
              <SliderInput
                label="Max EPS Growth (%)"
                value={maxEpsGrowth}
                onChange={setMaxEpsGrowth}
                min={-100}
                max={100}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="-100%"
                maxLabel="100%"
              />
            )}
          </div>
        </div>

        <div className="pt-2 border-t border-gray-100">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Price Filters</h3>

          <div className="space-y-3">
            <Toggle
              label="52-week return is positive"
              checked={check52wPositive}
              onChange={setCheck52wPositive}
            />

            <Toggle
              label="Minimum drawdown from 52w high"
              checked={checkMinDrawdown}
              onChange={setCheckMinDrawdown}
            />
            {checkMinDrawdown && (
              <SliderInput
                label="Min Drawdown (%)"
                value={minDrawdownPct}
                onChange={setMinDrawdownPct}
                min={5}
                max={80}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="5%"
                maxLabel="80%"
              />
            )}

            <Toggle
              label="Maximum drawdown from 52w high"
              checked={checkMaxDrawdown}
              onChange={setCheckMaxDrawdown}
            />
            {checkMaxDrawdown && (
              <SliderInput
                label="Max Drawdown (%)"
                value={maxDrawdownPct}
                onChange={setMaxDrawdownPct}
                min={10}
                max={90}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="10%"
                maxLabel="90%"
              />
            )}

            <Toggle
              label="3-month negative momentum"
              checked={check3mNegMomentum}
              onChange={setCheck3mNegMomentum}
            />

            <Toggle
              label="2-month negative relative momentum"
              checked={check2mNegRelMomentum}
              onChange={setCheck2mNegRelMomentum}
            />
            {check2mNegRelMomentum && (
              <SelectInput
                label="Relative Benchmark"
                value={relMomentumBenchmark}
                onChange={setRelMomentumBenchmark}
                options={SHORT_REL_BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
              />
            )}
          </div>
        </div>

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText={`Screening ${inputMode === "Custom Tickers" ? "custom tickers" : universe}...`}>
          Run Screen
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && (
        <LoadingSpinner
          message={formatScreenProgress(
            progress,
            `Screening ${inputMode === "Custom Tickers" ? "custom tickers" : universe} (this may take several minutes)...`,
          )}
        />
      )}
      {mutation.isError && <ErrorMessage message={errorMessage(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex flex-wrap gap-6 text-sm text-gray-600 mb-4">
            <span>Universe: <strong>{mutation.data.phase1_count ?? "—"}</strong></span>
            <span>Pass P/B + Loss: <strong>{mutation.data.phase1_pass_count ?? "—"}</strong></span>
            {mutation.data.phase3_pass_count != null && (
              <span>Pass price filters: <strong>{mutation.data.phase3_pass_count}</strong></span>
            )}
            <span>Final candidates: <strong>{mutation.data.final_count ?? rows.length}</strong></span>
          </div>
          <FailedTickerNotice failed={mutation.data.failed_tickers} />
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
    </>
  )
}

/* ─── Long Screen ─── */

const LONG_REL_BENCHMARK_OPTIONS = ["IWM", "SPY", "QQQ", "Same as Input"]

const LONG_EXTRA_COLUMNS: Record<string, ColumnDef> = {
  "Mkt Cap / EBIT": { key: "Mkt Cap / EBIT", header: "Mkt Cap / EBIT", format: v => v != null ? Number(v).toFixed(1) : "N/A" },
}

function buildLongColumns(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return SHORT_BASE_COLUMNS
  const firstRow = rows[0]
  const cols = [...SHORT_BASE_COLUMNS]
  for (const [key, col] of Object.entries(LONG_EXTRA_COLUMNS)) {
    if (key in firstRow) cols.push(col)
  }
  for (const [key, col] of Object.entries(SHORT_EXTRA_COLUMNS)) {
    if (key in firstRow) cols.push(col)
  }
  return cols
}

function LongPanel() {
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [checkPb, setCheckPb] = useState(false)
  const [pbThreshold, setPbThreshold] = useState(1.5)
  const [checkProfit, setCheckProfit] = useState(false)
  const [profitType, setProfitType] = useState<"Gross Profit" | "Operating Profit">("Gross Profit")
  const [checkIssuance, setCheckIssuance] = useState(false)

  // Fundamental growth filters (min instead of max)
  const [checkRevenue, setCheckRevenue] = useState(false)
  const [minRevenueGrowth, setMinRevenueGrowth] = useState(5)
  const [checkEps, setCheckEps] = useState(false)
  const [minEpsGrowth, setMinEpsGrowth] = useState(5)
  const [checkEbitMultiple, setCheckEbitMultiple] = useState(false)
  const [maxEbitMultiple, setMaxEbitMultiple] = useState(20)

  // Price filters
  const [check52wPositive, setCheck52wPositive] = useState(false)
  const [checkMinDrawdown, setCheckMinDrawdown] = useState(false)
  const [minDrawdownPct, setMinDrawdownPct] = useState(25)
  const [checkMaxDrawdown, setCheckMaxDrawdown] = useState(false)
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(60)
  const [check3mPosMomentum, setCheck3mPosMomentum] = useState(false)
  const [check2mPosRelMomentum, setCheck2mPosRelMomentum] = useState(false)
  const [relMomentumBenchmark, setRelMomentumBenchmark] = useState("Same as Input")
  const [progress, setProgress] = useState<ScreenJobProgress | undefined>(undefined)

  const mutation = useMutation({
    mutationFn: (body: LongScreenRequest) => runLongScreen(body, setProgress),
  })

  function handleRun() {
    setProgress({ phase: "queued", done: 0, total: 0 })
    mutation.mutate({
      input_mode: inputMode,
      universe,
      tickers,
      pb_threshold: checkPb ? pbThreshold : null,
      profit_type: checkProfit ? profitType : null,
      check_issuance: checkIssuance,
      check_revenue: checkRevenue,
      min_revenue_growth: minRevenueGrowth,
      check_eps: checkEps,
      min_eps_growth: minEpsGrowth,
      check_ebit_multiple: checkEbitMultiple,
      max_ebit_multiple: maxEbitMultiple,
      check_52w_positive: check52wPositive,
      check_min_drawdown: checkMinDrawdown,
      min_drawdown_pct: minDrawdownPct,
      check_max_drawdown: checkMaxDrawdown,
      max_drawdown_pct: maxDrawdownPct,
      check_3m_pos_momentum: check3mPosMomentum,
      check_2m_pos_rel_momentum: check2mPosRelMomentum,
      rel_momentum_benchmark: relMomentumBenchmark,
    })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []
  const columns = buildLongColumns(rows)

  return (
    <>
      <ControlPanel>
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
            label="Universe"
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            label="Tickers"
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
            uppercase
          />
        )}

        <Toggle
          label="P/B Threshold"
          checked={checkPb}
          onChange={setCheckPb}
        />
        {checkPb && (
          <SliderInput
            label="P/B Max"
            value={pbThreshold}
            onChange={setPbThreshold}
            min={0.5}
            max={3.0}
            step={0.1}
            formatValue={v => v.toFixed(1)}
            minLabel="0.5"
            maxLabel="3.0"
          />
        )}

        <Toggle
          label="Profit Type Filter"
          checked={checkProfit}
          onChange={setCheckProfit}
        />
        {checkProfit && (
          <div>
            <SegmentedControl
              options={[
                { value: "Gross Profit" as const, label: "Gross Profit" },
                { value: "Operating Profit" as const, label: "Operating Profit" },
              ]}
              value={profitType}
              onChange={setProfitType}
            />
          </div>
        )}

        <Toggle
          label="Low Net Equity Issuance (bottom quartile)"
          checked={checkIssuance}
          onChange={setCheckIssuance}
          description="Buyback-heavy companies — adds time (SEC EDGAR)"
        />

        <Toggle
          label="Max Market Cap / EBIT"
          checked={checkEbitMultiple}
          onChange={setCheckEbitMultiple}
        />
        {checkEbitMultiple && (
          <SliderInput
            label="Max Mkt Cap / EBIT"
            value={maxEbitMultiple}
            onChange={setMaxEbitMultiple}
            min={1}
            max={50}
            step={1}
            formatValue={v => `${v.toFixed(0)}x`}
            minLabel="1x"
            maxLabel="50x"
          />
        )}

        <div className="pt-2 border-t border-gray-100">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Fundamental Growth Filters</h3>
          <div className="space-y-3">
            <Toggle
              label="Min YoY Revenue Growth (avg of 3 quarters)"
              checked={checkRevenue}
              onChange={setCheckRevenue}
            />
            {checkRevenue && (
              <SliderInput
                label="Min Rev Growth (%)"
                value={minRevenueGrowth}
                onChange={setMinRevenueGrowth}
                min={0}
                max={50}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="0%"
                maxLabel="50%"
              />
            )}

            <Toggle
              label="Min YoY EPS Growth (avg of 3 quarters)"
              checked={checkEps}
              onChange={setCheckEps}
            />
            {checkEps && (
              <SliderInput
                label="Min EPS Growth (%)"
                value={minEpsGrowth}
                onChange={setMinEpsGrowth}
                min={0}
                max={100}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="0%"
                maxLabel="100%"
              />
            )}
          </div>
        </div>

        <div className="pt-2 border-t border-gray-100">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Price Filters</h3>

          <div className="space-y-3">
            <Toggle
              label="52-week return is positive"
              checked={check52wPositive}
              onChange={setCheck52wPositive}
            />

            <Toggle
              label="Minimum drawdown from 52w high"
              checked={checkMinDrawdown}
              onChange={setCheckMinDrawdown}
            />
            {checkMinDrawdown && (
              <SliderInput
                label="Min Drawdown (%)"
                value={minDrawdownPct}
                onChange={setMinDrawdownPct}
                min={5}
                max={80}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="5%"
                maxLabel="80%"
              />
            )}

            <Toggle
              label="Maximum drawdown from 52w high"
              checked={checkMaxDrawdown}
              onChange={setCheckMaxDrawdown}
            />
            {checkMaxDrawdown && (
              <SliderInput
                label="Max Drawdown (%)"
                value={maxDrawdownPct}
                onChange={setMaxDrawdownPct}
                min={10}
                max={90}
                step={5}
                formatValue={v => `${v}%`}
                minLabel="10%"
                maxLabel="90%"
              />
            )}

            <Toggle
              label="3-month positive momentum"
              checked={check3mPosMomentum}
              onChange={setCheck3mPosMomentum}
            />

            <Toggle
              label="2-month positive relative momentum"
              checked={check2mPosRelMomentum}
              onChange={setCheck2mPosRelMomentum}
            />
            {check2mPosRelMomentum && (
              <SelectInput
                label="Relative Benchmark"
                value={relMomentumBenchmark}
                onChange={setRelMomentumBenchmark}
                options={LONG_REL_BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
              />
            )}
          </div>
        </div>

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText={`Screening ${inputMode === "Custom Tickers" ? "custom tickers" : universe}...`}>
          Run Screen
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && (
        <LoadingSpinner
          message={formatScreenProgress(
            progress,
            `Screening ${inputMode === "Custom Tickers" ? "custom tickers" : universe} (this may take several minutes)...`,
          )}
        />
      )}
      {mutation.isError && <ErrorMessage message={errorMessage(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex flex-wrap gap-6 text-sm text-gray-600 mb-4">
            <span>Universe: <strong>{mutation.data.phase1_count ?? "—"}</strong></span>
            <span>Pass P/B + Profit: <strong>{mutation.data.phase1_pass_count ?? "—"}</strong></span>
            {mutation.data.phase3_pass_count != null && (
              <span>Pass price filters: <strong>{mutation.data.phase3_pass_count}</strong></span>
            )}
            <span>Final candidates: <strong>{mutation.data.final_count ?? rows.length}</strong></span>
          </div>
          <FailedTickerNotice failed={mutation.data.failed_tickers} />
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
    </>
  )
}

/* ─── Price Momentum ─── */

const PRICE_MOMENTUM_COLUMNS: ColumnDef[] = [
  { key: "ticker", header: "Ticker" },
  { key: "close", header: "Close", format: v => formatPlainNumber(v, 2) },
  {
    key: "avg20_roc63",
    header: "20D Avg ROC (63D)",
    format: formatSignedPercent,
    colorFn: colorPositiveNegative,
  },
  {
    key: "roc63",
    header: "3M ROC",
    format: formatSignedPercent,
    colorFn: colorPositiveNegative,
  },
  {
    key: "rel_roc42",
    header: "Rel ROC (42D)",
    format: formatSignedPercent,
    colorFn: colorPositiveNegative,
  },
  {
    key: "avg10_rel_roc",
    header: "10D Avg Rel ROC",
    format: formatSignedPercent,
    colorFn: colorPositiveNegative,
  },
  { key: "benchmark", header: "Benchmark" },
]

function PriceMomentumPanel() {
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("Same as Input")
  const [progress, setProgress] = useState<ScreenJobProgress | undefined>(undefined)

  const mutation = useMutation({
    mutationFn: (body: PriceMomentumRequest) => runPriceMomentum(body, setProgress),
  })

  function handleRun() {
    setProgress({ phase: "queued", done: 0, total: 0 })
    mutation.mutate({
      input_mode: inputMode,
      universe,
      tickers,
      benchmark,
    })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []

  return (
    <>
      <ControlPanel maxWidth="max-w-lg">
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
            label="Universe"
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            label="Tickers"
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
            uppercase
          />
        )}

        <SelectInput
          label="Benchmark"
          value={benchmark}
          onChange={setBenchmark}
          options={PRICE_MOMENTUM_BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Running price momentum...">
          Run Screen
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && (
        <LoadingSpinner
          message={formatScreenProgress(
            progress,
            `Running price momentum for ${inputMode === "Custom Tickers" ? "custom tickers" : universe}...`,
          )}
        />
      )}
      {mutation.isError && <ErrorMessage message={errorMessage(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex flex-wrap gap-6 text-sm text-gray-600 mb-4">
            <span>Input: <strong>{mutation.data.input_count ?? "—"}</strong></span>
            <span>Scored: <strong>{mutation.data.scored_count ?? rows.length}</strong></span>
            <span>Benchmark: <strong>{mutation.data.benchmark_name ?? benchmark}</strong></span>
            {mutation.data.date != null && (
              <span>Date: <strong>{String(mutation.data.date)}</strong></span>
            )}
          </div>
          <FailedTickerNotice failed={mutation.data.failed_tickers} />
          {rows.length > 0 ? (
            <DataTable columns={PRICE_MOMENTUM_COLUMNS} rows={rows} />
          ) : (
            <p className="text-gray-400">No momentum results returned.</p>
          )}
        </>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure inputs above and click Run Screen.</p>
      )}
    </>
  )
}

/* ─── Fundamental Momentum ─── */

function FundamentalMomentumPanel() {
  const [screenType, setScreenType] = useState<"EPS" | "Revenue" | "Both">("Both")
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("Same as Input")
  const [isRunning, setIsRunning] = useState(false)

  const mutation = useMutation({ mutationFn: runFundamentalMomentum })

  useEffect(() => {
    mutation.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleRun() {
    if (isRunning) return
    const resolvedBenchmark = benchmark === "Same as Input" && inputMode === "Universe"
      ? universe
      : benchmark

    setIsRunning(true)
    try {
      await mutation.mutateAsync({
        screen_type: screenType,
        universe,
        tickers,
        benchmark: resolvedBenchmark,
        input_mode: inputMode,
      })
    } catch {
      // mutation state already captures the error
    } finally {
      setIsRunning(false)
    }
  }

  const epsRows: Record<string, unknown>[] = mutation.data?.eps?.results_df ?? []
  const revRows: Record<string, unknown>[] = mutation.data?.rev?.results_df ?? []

  return (
    <>
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
            uppercase
          />
        )}

        <SelectInput
          label="Benchmark"
          value={benchmark}
          onChange={setBenchmark}
          options={BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
        />

        <ActionButton onClick={handleRun} loading={isRunning} loadingText="Screening...">
          Run Screen
        </ActionButton>
      </ControlPanel>

      {isRunning && <LoadingSpinner message="Running fundamental momentum screen..." />}
      {mutation.isError && <ErrorMessage message={errorMessage(mutation.error)} />}

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
    </>
  )
}

/* ─── Main Screeners Page ─── */

type ScreenerTab = "Quality" | "Short" | "Long" | "Price Momentum" | "Fundamental Momentum"

export function Screeners() {
  const [activeTab, setActiveTab] = useState<ScreenerTab>("Quality")

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Screeners</h1>
      </div>

      <div className="mb-6 max-w-2xl overflow-x-auto">
        <SegmentedControl
          options={[
            { value: "Quality" as const, label: "Quality" },
            { value: "Short" as const, label: "Short" },
            { value: "Long" as const, label: "Long" },
            { value: "Price Momentum" as const, label: "Price Momentum" },
            { value: "Fundamental Momentum" as const, label: "Fund. Momentum" },
          ]}
          value={activeTab}
          onChange={setActiveTab}
        />
      </div>

      {activeTab === "Quality" && <QualityPanel />}
      {activeTab === "Short" && <ShortPanel />}
      {activeTab === "Long" && <LongPanel />}
      {activeTab === "Price Momentum" && <PriceMomentumPanel />}
      {activeTab === "Fundamental Momentum" && <FundamentalMomentumPanel />}
    </div>
  )
}
