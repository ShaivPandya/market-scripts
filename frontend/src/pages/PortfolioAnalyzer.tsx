import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { ActionButton } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative } from "@/lib/colors"
import { runPortfolioAnalyzerAsync } from "@/lib/api"

interface AnalyzerResponse {
  weights_df?: Record<string, unknown>[]
  [key: string]: unknown
}

const ANALYZER_STATE_KEY = ["portfolio-analyzer", "state"] as const

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const percentFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const COLUMN_LABELS: Record<string, string> = {
  ticker: "Ticker",
  asset: "Asset",
  direction: "Direction",
  contrarian: "Contrarian",
  drawdown_52w: "Drawdown 52W",
  stabilized_10d: "Stabilized",
  days_since_new_low: "Days Since New Low",
  no_new_high_20d: "No New High 20D",
  days_since_high: "Days Since High",
  avg20_roc63: "Avg20 ROC(63)",
  avg10_rel_roc: "Avg10 Rel ROC",
  signal: "Signal",
  quality_signal: "Quality",
  eps_mom_signal: "EPS Momentum",
  rev_mom_signal: "Revenue Momentum",
  price_mom_signal: "Price Momentum",
}

const COLUMN_ORDER = [
  "ticker",
  "asset",
  "direction",
  "contrarian",
  "drawdown_52w",
  "stabilized_10d",
  "days_since_new_low",
  "no_new_high_20d",
  "days_since_high",
  "avg20_roc63",
  "avg10_rel_roc",
  "signal",
  "quality_signal",
  "eps_mom_signal",
  "rev_mom_signal",
  "price_mom_signal",
]

function toRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value.filter((row): row is Record<string, unknown> => row != null && typeof row === "object")
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function toBoolean(value: unknown): boolean | null {
  if (typeof value === "boolean") return value
  if (typeof value === "number") {
    if (value === 1) return true
    if (value === 0) return false
    return null
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (["1", "true", "t", "yes", "y"].includes(normalized)) return true
    if (["0", "false", "f", "no", "n"].includes(normalized)) return false
  }
  return null
}

function isSignalColumn(key: string) {
  return key === "signal" || key.endsWith("_signal")
}

function buildColumns(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  const available = new Set(Object.keys(rows[0]))

  return COLUMN_ORDER
    .filter(key => available.has(key))
    .map(key => ({
      key,
      header: COLUMN_LABELS[key] ?? key,
      colorFn: isSignalColumn(key) ? colorPositiveNegative : undefined,
      format: (value: unknown) => {
        if (key === "contrarian" || key === "stabilized_10d" || key === "no_new_high_20d") {
          const parsed = toBoolean(value)
          return parsed == null ? "N/A" : parsed ? "Yes" : "No"
        }

        if (key === "drawdown_52w") {
          const num = toNumber(value)
          return num == null ? "N/A" : `${percentFormatter.format(num * 100)}%`
        }

        if (key === "days_since_new_low") {
          const num = toNumber(value)
          return num == null ? "N/A" : String(Math.round(num))
        }

        if (isSignalColumn(key)) {
          const num = toNumber(value)
          if (num == null) return "N/A"
          return `${num >= 0 ? "+" : ""}${numberFormatter.format(num)}`
        }

        return String(value ?? "N/A")
      },
    }))
}

export function PortfolioAnalyzer() {
  const queryClient = useQueryClient()
  const cachedState = queryClient.getQueryData<{ result: AnalyzerResponse | null }>(ANALYZER_STATE_KEY)

  const [cachedResult, setCachedResult] = useState<AnalyzerResponse | null>(cachedState?.result ?? null)

  const mutation = useMutation({
    mutationFn: () => runPortfolioAnalyzerAsync({}),
    onSuccess: result => setCachedResult((result as AnalyzerResponse) ?? null),
  })

  useEffect(() => {
    queryClient.setQueryData(ANALYZER_STATE_KEY, { result: cachedResult })
  }, [cachedResult, queryClient])

  const data = (mutation.data as AnalyzerResponse | undefined) ?? cachedResult
  const rows = toRows(data?.weights_df)

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Analyzer</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          Signal and factor diagnostics to guide conviction inputs for Portfolio Sizer.
        </p>
      </div>

      <div className="rounded-xl border border-gray-200/80 bg-white p-5 mb-6">
        <ActionButton onClick={() => mutation.mutate()} loading={mutation.isPending} loadingText="Analyzing portfolio...">
          Run Portfolio Analyzer
        </ActionButton>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running portfolio analyzer..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && !mutation.isError && (
        <DataTable
          label="Signal Metrics"
          columns={buildColumns(rows)}
          rows={rows}
        />
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Click Run Portfolio Analyzer to load the signal metrics table.</p>
      )}
    </div>
  )
}
