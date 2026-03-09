import { useState, useMemo } from "react"
import { useSearchParams } from "react-router-dom"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import {
  fetchThesisMeta,
  fetchThesisDetail,
  fetchThesisStatus,
  updateThesisStatus,
  type ThesisMeta,
  type ThesisDetail,
  type ThesisStatus,
  type ThesisStatusValue,
  type ThesisEvaluation,
} from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MetricCard } from "@/components/shared/MetricCard"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { Dialog } from "@/components/shared/Dialog"
import { renderMarkdownLite } from "@/components/shared/MarkdownRenderer"
import { ActionButton, SelectInput, TextInput } from "@/components/shared/FormControls"
import { ThesisUpload } from "@/components/ThesisUpload"

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<string, string> = {
  active: "bg-green-50 text-green-700 border-green-200",
  under_review: "bg-yellow-50 text-yellow-700 border-yellow-200",
  invalidated: "bg-red-50 text-red-700 border-red-200",
}

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  under_review: "Under Review",
  invalidated: "Invalidated",
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-block rounded-md border px-2 py-0.5 text-xs font-medium ${STATUS_COLORS[status] ?? "bg-gray-50 text-gray-700 border-gray-200"}`}
    >
      {STATUS_LABELS[status] ?? status}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Evaluation color helpers
// ---------------------------------------------------------------------------

function evalDirectionColor(val: unknown) {
  const s = String(val).toLowerCase()
  if (s.includes("strengthen")) return "#00c853"
  if (s.includes("weaken")) return "#ff1744"
  return ""
}

function evalTechnicalColor(val: unknown) {
  const s = String(val).toLowerCase()
  if (s.includes("improv")) return "#00c853"
  if (s.includes("deterior")) return "#ff1744"
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

const LIST_COLUMNS: ColumnDef[] = [
  { key: "ticker", header: "Ticker" },
  { key: "status_label", header: "Status", colorFn: (v) => {
    const s = String(v).toLowerCase()
    if (s === "active") return "#00c853"
    if (s === "under review") return "#fb8c00"
    if (s === "invalidated") return "#ff1744"
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
// Detail View
// ---------------------------------------------------------------------------

const EVAL_COLUMNS: ColumnDef[] = [
  { key: "evaluated_at", header: "Date" },
  { key: "thesis_status", header: "Direction", colorFn: evalDirectionColor },
  { key: "technical_read", header: "Technical", colorFn: evalTechnicalColor },
  { key: "fundamental_read", header: "Fundamental" },
  { key: "action", header: "Action", colorFn: evalActionColor },
  { key: "confidence", header: "Confidence" },
  { key: "risk_flag", header: "Risk Flag" },
]

type DetailTab = "content" | "evaluations" | "history"

function ThesisDetailView({ ticker, onBack }: { ticker: string; onBack: () => void }) {
  const [tab, setTab] = useState<DetailTab>("content")
  const [statusDialogOpen, setStatusDialogOpen] = useState(false)
  const [newStatus, setNewStatus] = useState<ThesisStatusValue>("under_review")
  const [statusReason, setStatusReason] = useState("")
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useApiQuery<ThesisDetail>(
    ["thesis", "detail", ticker],
    () => fetchThesisDetail(ticker),
  )

  const { data: thesisStatus } = useApiQuery<Record<string, string>>(
    ["thesis", "status"],
    fetchThesisStatus,
  )

  const statusMutation = useMutation({
    mutationFn: () => updateThesisStatus(ticker, newStatus, statusReason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["thesis"] })
      setStatusDialogOpen(false)
      setStatusReason("")
    },
  })

  if (isLoading) return <LoadingSpinner message={`Loading ${ticker}...`} />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load thesis detail"} />

  const { meta, content, status_history, evaluations } = data

  const evalRows = evaluations.map(ev => ({
    ...ev,
    risk_flag: ev.risk_flag ?? "-",
  }))

  const tabs: { key: DetailTab; label: string }[] = [
    { key: "content", label: "Thesis Content" },
    { key: "evaluations", label: `Evaluations (${evaluations.length})` },
    { key: "history", label: `Status History (${status_history.length})` },
  ]

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
          >
            &larr; Back
          </button>
          <h2 className="text-xl font-semibold text-gray-900">{ticker}</h2>
          <StatusBadge status={meta.status} />
          <ThesisUpload ticker={ticker} status={(thesisStatus?.[ticker] ?? "missing") as ThesisStatus} />
        </div>
        <button
          type="button"
          onClick={() => {
            setNewStatus(meta.status === "active" ? "under_review" : "active")
            setStatusDialogOpen(true)
          }}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
        >
          Change Status
        </button>
      </div>

      <div className="text-xs text-muted mb-4">
        Created: {meta.created_at?.slice(0, 10) ?? "-"} &middot; Updated: {meta.updated_at?.slice(0, 10) ?? "-"}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-gray-200 pb-1">
        {tabs.map(t => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 text-sm font-medium rounded-t-lg transition-colors ${
              tab === t.key
                ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "content" && (
        <section className="theme-surface rounded-xl p-6">
          {content ? (
            <div className="prose-compact">{renderMarkdownLite(content)}</div>
          ) : (
            <p className="text-sm text-muted">No thesis content available. Upload a PDF to generate.</p>
          )}
        </section>
      )}

      {tab === "evaluations" && (
        <section className="theme-surface rounded-xl p-4">
          {evaluations.length > 0 ? (
            <DataTable columns={EVAL_COLUMNS} rows={evalRows} label="Weekly Evaluations" />
          ) : (
            <p className="text-sm text-muted">No evaluations yet. Run the weekly report to generate evaluations.</p>
          )}
          {evaluations.length > 0 && evaluations[0] && (
            <div className="mt-4 p-3 rounded-lg border border-gray-200 bg-gray-50">
              <p className="text-xs font-semibold text-gray-600 mb-1">Latest Key Developments ({evaluations[0].evaluated_at})</p>
              <ul className="list-disc pl-5 space-y-1">
                {(evaluations[0].key_developments ?? []).map((dev, i) => (
                  <li key={i} className="text-sm text-gray-700">{dev}</li>
                ))}
              </ul>
              {evaluations[0].earnings_note && (
                <p className="mt-2 text-sm text-gray-700">
                  <strong>Earnings:</strong> {evaluations[0].earnings_note}
                </p>
              )}
            </div>
          )}
        </section>
      )}

      {tab === "history" && (
        <section className="theme-surface rounded-xl p-4">
          {status_history.length > 0 ? (
            <div className="space-y-3">
              {status_history.map(h => (
                <div key={h.id} className="flex items-start gap-3 text-sm">
                  <span className="text-xs text-muted whitespace-nowrap mt-0.5">
                    {h.changed_at?.slice(0, 10)}
                  </span>
                  <div>
                    <span className="text-gray-700">
                      {h.old_status ? (
                        <>
                          <StatusBadge status={h.old_status} />
                          <span className="mx-1">&rarr;</span>
                        </>
                      ) : (
                        "Created as "
                      )}
                      <StatusBadge status={h.new_status} />
                    </span>
                    {h.reason && (
                      <p className="text-xs text-muted mt-0.5">{h.reason}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted">No status history available.</p>
          )}
        </section>
      )}

      {/* Status change dialog */}
      <Dialog
        open={statusDialogOpen}
        onOpenChange={setStatusDialogOpen}
        title="Change Thesis Status"
        description={`Update the status for ${ticker}`}
      >
        <div className="space-y-4">
          <SelectInput
            label="New Status"
            value={newStatus}
            onChange={v => setNewStatus(v as ThesisStatusValue)}
            options={[
              { value: "active", label: "Active" },
              { value: "under_review", label: "Under Review" },
              { value: "invalidated", label: "Invalidated" },
            ]}
          />
          <TextInput
            label="Reason"
            value={statusReason}
            onChange={setStatusReason}
            placeholder="Why is the status changing?"
          />
          {statusMutation.isError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {String(statusMutation.error)}
            </div>
          )}
          <ActionButton
            onClick={() => statusMutation.mutate()}
            loading={statusMutation.isPending}
            loadingText="Updating..."
          >
            Update Status
          </ActionButton>
        </div>
      </Dialog>
    </>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function ThesisManager() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [selectedTicker, setSelectedTicker] = useState<string | null>(
    searchParams.get("ticker"),
  )

  const handleSelect = (ticker: string) => {
    setSelectedTicker(ticker)
    setSearchParams({ ticker })
  }

  const handleBack = () => {
    setSelectedTicker(null)
    setSearchParams({})
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Investment Theses</h1>
        <RefreshButton queryKeys={[["thesis", "meta"], ["thesis", "detail", selectedTicker ?? ""]]} />
      </div>

      {selectedTicker ? (
        <ThesisDetailView ticker={selectedTicker} onBack={handleBack} />
      ) : (
        <ThesisList onSelect={handleSelect} />
      )}
    </div>
  )
}
