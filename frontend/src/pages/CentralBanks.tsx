import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCentralBanks } from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"

const SIGNAL_COLORS: Record<string, string> = {
  hawkish: "bg-red-100 text-red-700",
  dovish: "bg-green-100 text-green-700",
  neutral: "bg-gray-100 text-gray-600",
  tightening: "bg-red-100 text-red-700",
  easing: "bg-green-100 text-green-700",
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
  const classes = SIGNAL_COLORS[String(value).toLowerCase()] ?? "bg-gray-100 text-gray-600"
  return (
    <span className={`text-xs px-2 py-0.5 rounded font-medium ${classes}`}>
      {label}: {value}
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
  const [viewMode, setViewMode] = useState<"grouped" | "chronological">("grouped")
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
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Central Bank Monitor</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setRefresh(r => !r)}
            className="px-3 py-1.5 text-sm rounded border border-gray-300 bg-white hover:bg-gray-50"
          >
            Refresh
          </button>
          <RefreshButton queryKeys={[["central-banks", refresh]]} />
        </div>
      </div>

      {isLoading && <LoadingSpinner message="Fetching central bank data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div className="flex gap-4 text-sm text-gray-500">
              <span>Total: <strong>{data.counts?.total ?? items.length}</strong></span>
              {sources.map(s => (
                <span key={s}>{s}: <strong>{bySource[s].length}</strong></span>
              ))}
            </div>
            <div className="inline-flex rounded-lg border border-gray-300 bg-white p-0.5">
              <button
                onClick={() => setViewMode("grouped")}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  viewMode === "grouped"
                    ? "bg-blue-600 text-white"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                By central bank
              </button>
              <button
                onClick={() => setViewMode("chronological")}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  viewMode === "chronological"
                    ? "bg-blue-600 text-white"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Chronological
              </button>
            </div>
          </div>

          {viewMode === "grouped" ? (
            sources.map(source => (
              <section key={source} className="mb-8">
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: getSourceColor(source) }} />
                  {source}
                </h2>
                <div className="space-y-4">
                  {bySource[source].map((item, i) => (
                    <CBItemCard key={item.url || `${source}-${item.title}-${item.published_at}-${i}`} item={item} />
                  ))}
                </div>
              </section>
            ))
          ) : (
            <section>
              <h2 className="text-lg font-semibold mb-3">Latest Updates</h2>
              <div className="space-y-4">
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
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-4 mb-2">
        <div>
          <span className="text-xs font-medium text-gray-400 uppercase">{item.kind}</span>
          {item.url ? (
            <a href={item.url} target="_blank" rel="noreferrer" className="block text-sm font-semibold text-blue-600 hover:underline">
              {item.title}
            </a>
          ) : (
            <p className="text-sm font-semibold text-gray-800">{item.title}</p>
          )}
          {showSource && item.source && (
            <span
              className="inline-flex mt-1 px-2 py-0.5 rounded text-xs font-medium"
              style={{ backgroundColor: sourceColor, color: "#fff" }}
            >
              {item.source}
            </span>
          )}
        </div>
        {item.published_at && (
          <span className="text-xs text-gray-400 whitespace-nowrap">
            {new Date(item.published_at).toLocaleDateString()}
          </span>
        )}
      </div>

      {Object.keys(signals).length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {Object.entries(signals).map(([k, v]) =>
            v ? <SignalBadge key={k} label={k.replace(/_/g, " ")} value={String(v)} /> : null
          )}
        </div>
      )}

      {(item.summary_bullets ?? []).length > 0 && (
        <ul className="text-sm text-gray-700 list-disc list-inside space-y-1 mb-2">
          {item.summary_bullets.map((b, i) => <li key={i}>{b}</li>)}
        </ul>
      )}

      {item.content_preview && (
        <div>
          <button
            onClick={() => setExpanded(e => !e)}
            className="text-xs text-blue-500 hover:underline"
          >
            {expanded ? "Show less" : "Show more"}
          </button>
          {expanded && (
            <p className="text-xs text-gray-500 mt-2 whitespace-pre-wrap">{item.content_preview}</p>
          )}
        </div>
      )}
    </div>
  )
}
