import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPortfolioNews } from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { cn } from "@/lib/utils"

/* ── colour helpers ───────────────────────────────────────────────────────── */

const PROVIDER_STYLES: Record<string, { bg: string; text: string }> = {
    IBKR: { bg: "bg-amber-50", text: "text-amber-700" },
}

function tickerColor(ticker: string) {
    // deterministic hue from ticker string
    let h = 0
    for (let i = 0; i < ticker.length; i++) h = (h * 31 + ticker.charCodeAt(i)) % 360
    return `hsl(${h}, 60%, 50%)`
}

/* ── types ────────────────────────────────────────────────────────────────── */

interface NewsArticle {
    ticker: string
    title: string
    url: string
    source: string
    seendate: string
    socialimage: string
    language: string
    provider: string
}

interface NewsData {
    items: NewsArticle[]
    by_ticker: Record<string, NewsArticle[]>
    ticker_names: Record<string, string>
    counts: { total: number; tickers: number }
}

/* ── main component ───────────────────────────────────────────────────────── */

export function PortfolioNews() {
    const [refresh, setRefresh] = useState(false)
    const [viewMode, setViewMode] = useState<"grouped" | "chronological">("chronological")
    const { data, isLoading, error } = useApiQuery<NewsData>(
        ["portfolio-news", refresh],
        () => fetchPortfolioNews(refresh),
        60 * 60 * 1000,
    )

    const items: NewsArticle[] = data?.items ?? []
    const byTicker: Record<string, NewsArticle[]> = data?.by_ticker ?? {}
    const tickerNames: Record<string, string> = data?.ticker_names ?? {}

    const tickers = Object.keys(byTicker).filter(t => byTicker[t].length > 0)
    const chronologicalItems = [...items].sort((a, b) => {
        const left = Date.parse(a.seendate) || 0
        const right = Date.parse(b.seendate) || 0
        return right - left
    })

    return (
        <div>
            {/* Page header */}
            <div className="mb-6">
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio News</h1>
                        {data && !isLoading && (
                            <p className="text-sm text-gray-400 mt-0.5">
                                {data.counts?.total ?? items.length} articles across {data.counts?.tickers ?? tickers.length} positions
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
                                        {mode === "grouped" ? "By ticker" : "Latest"}
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

            {isLoading && <LoadingSpinner message="Fetching portfolio news..." />}
            {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

            {data && !isLoading && (
                <>
                    {viewMode === "grouped" ? (
                        tickers.map(ticker => (
                            <section key={ticker} className="mb-8">
                                <div className="flex items-center gap-2 mb-3">
                                    <span
                                        className="inline-block w-2 h-2 rounded-full shrink-0"
                                        style={{ backgroundColor: tickerColor(ticker) }}
                                    />
                                    <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400">
                                        {ticker}
                                    </h2>
                                    {tickerNames[ticker] && tickerNames[ticker] !== ticker && (
                                        <span className="text-xs text-gray-300 ml-1">
                                            {tickerNames[ticker]}
                                        </span>
                                    )}
                                    <span className="text-xs text-gray-300 ml-1">
                                        {byTicker[ticker].length}
                                    </span>
                                </div>
                                <div className="space-y-3">
                                    {byTicker[ticker].map((article, i) => (
                                        <NewsCard key={article.url || `${ticker}-${article.title}-${i}`} article={article} />
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
                                {chronologicalItems.map((article, i) => (
                                    <NewsCard
                                        key={article.url || `${article.ticker}-${article.title}-${i}`}
                                        article={article}
                                        showTicker
                                        tickerName={tickerNames[article.ticker]}
                                    />
                                ))}
                            </div>
                        </section>
                    )}
                </>
            )}
        </div>
    )
}

/* ── article card ─────────────────────────────────────────────────────────── */

function NewsCard({
    article,
    showTicker = false,
    tickerName,
}: {
    article: NewsArticle
    showTicker?: boolean
    tickerName?: string
}) {
    const color = tickerColor(article.ticker)

    return (
        <div className="relative flex rounded-xl border border-gray-200/80 bg-white overflow-hidden">
            {/* Left accent bar */}
            <div className="w-[3px] shrink-0" style={{ backgroundColor: color }} />

            {/* Card body */}
            <div className="flex-1 px-4 py-3.5">
                {/* Eyebrow row */}
                <div className="flex items-start justify-between gap-4 mb-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                        {showTicker && (
                            <span
                                className="text-[10px] tracking-widest uppercase font-semibold"
                                style={{ color }}
                            >
                                {article.ticker}
                            </span>
                        )}
                        {showTicker && tickerName && tickerName !== article.ticker && (
                            <span className="text-[10px] text-gray-400">
                                {tickerName}
                            </span>
                        )}
                        {article.provider && (
                            <span
                                className={cn(
                                    "inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider uppercase",
                                    PROVIDER_STYLES[article.provider]?.bg ?? "bg-gray-50",
                                    PROVIDER_STYLES[article.provider]?.text ?? "text-gray-500",
                                )}
                            >
                                {article.provider}
                            </span>
                        )}
                        {article.source && (
                            <span className="text-[10px] tracking-widest uppercase text-gray-400 font-semibold">
                                {article.source}
                            </span>
                        )}
                    </div>
                    {article.seendate && (
                        <span className="text-[11px] text-gray-400 whitespace-nowrap tabular-nums shrink-0">
                            {new Date(article.seendate).toLocaleDateString(undefined, {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                                hour: "numeric",
                                minute: "2-digit",
                            })}
                        </span>
                    )}
                </div>

                {/* Title */}
                {article.url ? (
                    <a
                        href={article.url}
                        target="_blank"
                        rel="noreferrer"
                        className="block text-sm font-semibold text-gray-900 hover:underline decoration-gray-300 underline-offset-2 mb-2 leading-snug"
                    >
                        {article.title}
                    </a>
                ) : (
                    <p className="text-sm font-semibold text-gray-900 mb-2 leading-snug">{article.title}</p>
                )}
            </div>

            {/* Thumbnail */}
            {article.socialimage && (
                <div className="w-20 h-20 shrink-0 self-center mr-3">
                    <img
                        src={article.socialimage}
                        alt=""
                        className="w-full h-full object-cover rounded-lg"
                        loading="lazy"
                        onError={e => {
                            ; (e.target as HTMLImageElement).style.display = "none"
                        }}
                    />
                </div>
            )}
        </div>
    )
}
