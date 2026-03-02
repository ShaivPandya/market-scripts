import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCentralBanks } from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { cn } from "@/lib/utils"

const SIGNAL_DOT_COLORS: Record<string, string> = {
  hawkish: "#FF4245",
  tightening: "#FF4245",
  dovish: "#30D158",
  easing: "#30D158",
  neutral: "#8E8E93",
}

const SIGNAL_TEXT_COLORS: Record<string, string> = {
  hawkish: "text-red-600",
  tightening: "text-red-600",
  dovish: "text-green-600",
  easing: "text-green-600",
  neutral: "text-gray-500",
}

const SOURCE_COLORS: Record<string, string> = {
  FED: "#30D158",
  ECB: "#0091FF",
  BOJ: "#FF4245",
  BOE: "#6D7CFF",
  BOC: "#FF9230",
  SNB: "#00DAC3",
  NORGES: "#3CD3FE",
  RBA: "#FFD600",
  RBNZ: "#DB34F2",
  RIKSBANK: "#FF375F",
}

function getSourceColor(source: string) {
  return SOURCE_COLORS[source] ?? "#64748B"
}

function SignalBadge({ label, value }: { label: string; value: string }) {
  const key = String(value).toLowerCase()
  const dotColor = SIGNAL_DOT_COLORS[key] ?? "#8E8E93"
  const textClass = SIGNAL_TEXT_COLORS[key] ?? "text-gray-500"
  return (
    <span
      title={`${label}: ${value}`}
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
        "bg-gray-50 border border-gray-100",
        textClass
      )}
    >
      <span
        className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
        style={{ backgroundColor: dotColor }}
      />
      {value}
    </span>
  )
}

interface CBItem {
  source: string
  kind: string
  title: string
  url: string
  published_at: string
  summary_bullets: string[]
  signals: Record<string, string>
  content_preview: string
}

function publishedTimestamp(value: string) {
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? 0 : parsed
}

export function CentralBanks() {
  const [refresh, setRefresh] = useState(false)
  const [viewMode, setViewMode] = useState<"grouped" | "chronological">("chronological")
  const { data, isLoading, error } = useApiQuery(
    ["central-banks", refresh],
    () => fetchCentralBanks(refresh),
    60 * 60 * 1000,
  )

  const items: CBItem[] = data?.items ?? []
  const bySource: Record<string, CBItem[]> = data?.by_source ?? {}

  const sources = Object.keys(bySource)
  const chronologicalItems = [...items].sort((a, b) => {
    const left = publishedTimestamp(a.published_at)
    const right = publishedTimestamp(b.published_at)
    return right - left
  })

  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Central Bank Monitor</h1>
            {data && !isLoading && (
              <p className="text-sm text-gray-400 mt-0.5">
                {data.counts?.total ?? items.length} items across {sources.length} central banks
              </p>
            )}
          </div>
          <div className="flex items-center gap-3 shrink-0">
            {data && !isLoading && (
              <div className="inline-flex items-center rounded-full bg-gray-100 p-0.5">
                {(["grouped", "chronological"] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={cn(
                      "px-3.5 py-1.5 text-sm rounded-full transition-all duration-150",
                      viewMode === mode
                        ? "bg-white text-gray-900 font-medium shadow-sm"
                        : "text-gray-500 hover:text-gray-700"
                    )}
                  >
                    {mode === "grouped" ? "By bank" : "Latest"}
                  </button>
                ))}
              </div>
            )}
            <button
              onClick={() => setRefresh(r => !r)}
              className="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-200 bg-white text-gray-700 shadow-sm hover:bg-gray-50 transition-colors"
            >
              Refresh Data
            </button>
          </div>
        </div>
      </div>

      {isLoading && <LoadingSpinner message="Fetching central bank data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {viewMode === "grouped" ? (
            sources.map(source => (
              <section key={source} className="mb-8">
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className="inline-block w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: getSourceColor(source) }}
                  />
                  <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400">
                    {source}
                  </h2>
                  <span className="text-xs text-gray-300 ml-1">
                    {bySource[source].length}
                  </span>
                </div>
                <div className="space-y-3">
                  {bySource[source].map((item, i) => (
                    <CBItemCard key={item.url || `${source}-${item.title}-${item.published_at}-${i}`} item={item} />
                  ))}
                </div>
              </section>
            ))
          ) : (
            <section>
              <div className="flex items-center gap-2 mb-3">
                <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400">
                  Latest
                </h2>
                <span className="text-xs text-gray-300">
                  {chronologicalItems.length}
                </span>
              </div>
              <div className="space-y-3">
                {chronologicalItems.map((item, i) => (
                  <CBItemCard key={item.url || `${item.source}-${item.title}-${item.published_at}-${i}`} item={item} showSource />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}

function CBItemCard({ item, showSource = false }: { item: CBItem; showSource?: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const signals = item.signals ?? {}
  const sourceColor = getSourceColor(item.source)

  return (
    <div className="relative flex rounded-xl border border-gray-200/80 bg-white overflow-hidden">
      {/* Left accent bar */}
      <div className="w-[3px] shrink-0" style={{ backgroundColor: sourceColor }} />

      {/* Card body */}
      <div className="flex-1 px-4 py-3.5">
        {/* Eyebrow row: kind + source + date */}
        <div className="flex items-start justify-between gap-4 mb-1.5">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] tracking-widest uppercase text-gray-400 font-semibold">
              {item.kind}
            </span>
            {showSource && item.source && (
              <span
                className="text-[10px] tracking-widest uppercase font-semibold"
                style={{ color: sourceColor }}
              >
                {item.source}
              </span>
            )}
          </div>
          {item.published_at && (
            <span className="text-[11px] text-gray-400 whitespace-nowrap tabular-nums shrink-0">
              {new Date(item.published_at).toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </span>
          )}
        </div>

        {/* Title */}
        {item.url ? (
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="block text-sm font-semibold text-gray-900 hover:underline decoration-gray-300 underline-offset-2 mb-2 leading-snug"
          >
            {item.title}
          </a>
        ) : (
          <p className="text-sm font-semibold text-gray-900 mb-2 leading-snug">{item.title}</p>
        )}

        {/* Signal badges */}
        {Object.keys(signals).length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2.5">
            {Object.entries(signals).map(([k, v]) =>
              v ? <SignalBadge key={k} label={k.replace(/_/g, " ")} value={String(v)} /> : null
            )}
          </div>
        )}

        {/* Summary bullets */}
        {(item.summary_bullets ?? []).length > 0 && (
          <div className="space-y-1.5 mb-2.5">
            {item.summary_bullets.map((b, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-gray-600 leading-snug">
                <span className="text-gray-300 select-none mt-px">·</span>
                <span>{b}</span>
              </div>
            ))}
          </div>
        )}

        {/* Expand toggle */}
        {item.content_preview && (
          <div>
            <button
              onClick={() => setExpanded(e => !e)}
              className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors mt-0.5"
            >
              <span>{expanded ? "Show less" : "Show preview"}</span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                className={cn("transition-transform duration-200", expanded ? "rotate-180" : "rotate-0")}
              >
                <path
                  d="M2.5 4.5L6 8L9.5 4.5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            {expanded && (
              <p className="text-xs text-gray-500 mt-2.5 leading-relaxed whitespace-pre-wrap border-t border-gray-100 pt-2.5">
                {item.content_preview}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
