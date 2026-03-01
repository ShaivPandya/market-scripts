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

export function CentralBanks() {
  const [refresh, setRefresh] = useState(false)
  const { data, isLoading, error } = useApiQuery(
    ["central-banks", refresh],
    () => fetchCentralBanks(refresh),
    60 * 60 * 1000,
  )

  const items: CBItem[] = data?.items ?? []
  const bySource: Record<string, CBItem[]> = data?.by_source ?? {}

  const sources = Object.keys(bySource)

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
          <div className="flex gap-4 mb-4 text-sm text-gray-500">
            <span>Total: <strong>{data.counts?.total ?? items.length}</strong></span>
            {sources.map(s => (
              <span key={s}>{s}: <strong>{bySource[s].length}</strong></span>
            ))}
          </div>

          {sources.map(source => (
            <section key={source} className="mb-8">
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-full bg-blue-500" />
                {source}
              </h2>
              <div className="space-y-4">
                {bySource[source].map((item, i) => (
                  <CBItemCard key={i} item={item} />
                ))}
              </div>
            </section>
          ))}
        </>
      )}
    </div>
  )
}

function CBItemCard({ item }: { item: CBItem }) {
  const [expanded, setExpanded] = useState(false)
  const signals = item.signals ?? {}

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
