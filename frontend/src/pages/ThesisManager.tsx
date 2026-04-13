import { useMemo } from "react"
import { useNavigate } from "react-router-dom"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchThesisMeta,
  type ThesisMeta,
} from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MetricCard } from "@/components/shared/MetricCard"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"

// ---------------------------------------------------------------------------
// Evaluation color helpers
// ---------------------------------------------------------------------------

function evalDirectionColor(val: unknown) {
  const s = String(val).toLowerCase()
  if (s.includes("strengthen")) return "#00c853"
  if (s.includes("weaken")) return "#ff1744"
  return ""
}

function evalActionColor(val: unknown) {
  const s = String(val).toLowerCase()
  if (s === "hold") return "#00c853"
  if (s.includes("reassess") || s.includes("urgent")) return "#ff1744"
  return ""
}

// ---------------------------------------------------------------------------
// List View
// ---------------------------------------------------------------------------

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  under_review: "Under Review",
  invalidated: "Invalidated",
  missing: "Missing",
}

const LIST_COLUMNS: ColumnDef[] = [
  { key: "ticker", header: "Ticker" },
  { key: "status_label", header: "Status", colorFn: (v) => {
    const s = String(v).toLowerCase()
    if (s === "active") return "#00c853"
    if (s === "under review") return "#fb8c00"
    if (s === "invalidated") return "#ff1744"
    if (s === "missing") return "#9e9e9e"
    return ""
  }},
  { key: "eval_date", header: "Last Eval" },
  { key: "eval_direction", header: "Direction", colorFn: evalDirectionColor },
  { key: "eval_action", header: "Action", colorFn: evalActionColor },
  { key: "eval_confidence", header: "Confidence" },
]

function ThesisList({ onSelect }: { onSelect: (ticker: string) => void }) {
  const { data, isLoading, error } = useApiQuery<ThesisMeta[]>(
    ["thesis", "meta"],
    fetchThesisMeta,
  )

  const counts = useMemo(() => {
    if (!data) return { active: 0, under_review: 0, invalidated: 0 }
    return {
      active: data.filter(m => m.status === "active").length,
      under_review: data.filter(m => m.status === "under_review").length,
      invalidated: data.filter(m => m.status === "invalidated").length,
    }
  }, [data])

  const rows = useMemo(() => {
    if (!data) return []
    return data.map(m => {
      const ev = m.latest_evaluation
      return {
        ticker: m.ticker,
        status_label: STATUS_LABELS[m.status] ?? m.status,
        eval_date: ev?.evaluated_at ?? "-",
        eval_direction: ev?.thesis_status ?? "-",
        eval_action: ev?.action ?? "-",
        eval_confidence: ev?.confidence ?? "-",
      }
    })
  }, [data])

  if (isLoading) return <LoadingSpinner message="Loading thesis metadata..." />
  if (error) return <ErrorMessage message={String(error)} />

  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 mb-6">
        <MetricCard title="Active" value={counts.active} signal="success" signalLabel="Active" />
        <MetricCard title="Under Review" value={counts.under_review} signal="warning" signalLabel="Under Review" />
        <MetricCard title="Invalidated" value={counts.invalidated} signal="error" signalLabel="Invalidated" />
      </div>

      <section className="theme-surface rounded-xl p-4">
        <DataTable
          columns={LIST_COLUMNS}
          rows={rows}
          label="Investment Theses"
          onRowClick={row => onSelect(String(row.ticker))}
        />
      </section>
    </>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function ThesisManager() {
  const navigate = useNavigate()

  const handleSelect = (ticker: string) => {
    navigate(`/dossier/${ticker}`, { state: { from: "theses" } })
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Investment Theses</h1>
        <RefreshButton queryKeys={[["thesis", "meta"]]} />
      </div>

      <ThesisList onSelect={handleSelect} />
    </div>
  )
}
