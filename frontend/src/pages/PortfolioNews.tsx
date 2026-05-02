import { useEffect, useMemo, useRef, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Calendar, FileText, Loader2, RefreshCw, Search, Trash2, Upload } from "lucide-react"

import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
    deletePortfolioNewsDigest,
    fetchPortfolioNews,
    fetchPortfolioNewsDigest,
    type NewsDigestListResponse,
    type NewsDigestStory,
    type NewsDigestSummary,
    uploadPortfolioNewsDigest,
} from "@/lib/api"
import { cn } from "@/lib/utils"

function formatDate(value?: string | null, withTime = false) {
    if (!value) return "Unknown"
    const parsed = Date.parse(value)
    if (!Number.isFinite(parsed)) return value
    return new Date(parsed).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        ...(withTime ? { hour: "numeric", minute: "2-digit" } : {}),
    })
}

function matchText(story: NewsDigestStory, query: string) {
    const haystack = [
        story.headline,
        story.section,
        story.digest_title,
        story.generated_date,
        ...(story.notes ?? []),
    ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
    return haystack.includes(query)
}

function summaryMatches(summary: NewsDigestSummary, query: string, storyDigestIds: Set<string>) {
    if (storyDigestIds.has(summary.id)) return true
    const haystack = [
        summary.title,
        summary.filename,
        summary.generated_date,
        ...summary.sections.map(section => section.name),
    ]
        .join(" ")
        .toLowerCase()
    return haystack.includes(query)
}

export function PortfolioNews() {
    const queryClient = useQueryClient()
    const fileInputRef = useRef<HTMLInputElement | null>(null)
    const [selectedId, setSelectedId] = useState<string | null>(null)
    const [search, setSearch] = useState("")
    const [uploadError, setUploadError] = useState<string | null>(null)
    const [isUploading, setIsUploading] = useState(false)
    const [deletingId, setDeletingId] = useState<string | null>(null)

    const listQuery = useApiQuery<NewsDigestListResponse>(
        ["portfolio-news"],
        fetchPortfolioNews,
        60 * 1000,
    )

    const items = listQuery.data?.items ?? []
    const stories = listQuery.data?.stories ?? []

    useEffect(() => {
        if (!items.length) {
            if (selectedId) setSelectedId(null)
            return
        }
        if (!selectedId || !items.some(item => item.id === selectedId)) {
            setSelectedId(items[0].id)
        }
    }, [items, selectedId])

    const detailQuery = useQuery({
        queryKey: ["portfolio-news", selectedId],
        queryFn: () => fetchPortfolioNewsDigest(selectedId as string),
        enabled: Boolean(selectedId),
        staleTime: 60 * 1000,
        retry: 1,
    })

    const normalizedSearch = search.trim().toLowerCase()
    const filteredStories = useMemo(() => {
        if (!normalizedSearch) return []
        return stories.filter(story => matchText(story, normalizedSearch))
    }, [normalizedSearch, stories])

    const visibleDigests = useMemo(() => {
        if (!normalizedSearch) return items
        const storyDigestIds = new Set(filteredStories.map(story => story.digest_id).filter(Boolean) as string[])
        return items.filter(item => summaryMatches(item, normalizedSearch, storyDigestIds))
    }, [filteredStories, items, normalizedSearch])

    async function handleUpload(file: File | undefined) {
        if (!file) return
        setUploadError(null)
        if (!file.name.toLowerCase().endsWith(".md")) {
            setUploadError("Markdown files only")
            return
        }
        setIsUploading(true)
        try {
            const result = await uploadPortfolioNewsDigest(file)
            queryClient.setQueryData(["portfolio-news", result.digest.id], result.digest)
            await queryClient.invalidateQueries({ queryKey: ["portfolio-news"] })
            setSelectedId(result.digest.id)
        } catch (err) {
            setUploadError(err instanceof Error ? err.message : "Upload failed")
        } finally {
            setIsUploading(false)
            if (fileInputRef.current) fileInputRef.current.value = ""
        }
    }

    async function handleDelete(digest: NewsDigestSummary) {
        const ok = window.confirm(`Delete "${digest.title}"?`)
        if (!ok) return
        setDeletingId(digest.id)
        setUploadError(null)
        try {
            await deletePortfolioNewsDigest(digest.id)
            queryClient.removeQueries({ queryKey: ["portfolio-news", digest.id] })
            if (selectedId === digest.id) setSelectedId(null)
            await queryClient.invalidateQueries({ queryKey: ["portfolio-news"] })
        } catch (err) {
            setUploadError(err instanceof Error ? err.message : "Delete failed")
        } finally {
            setDeletingId(null)
        }
    }

    const selectedSummary = items.find(item => item.id === selectedId) ?? null
    const selectedDetail = detailQuery.data
    const selectedDigestForDelete = selectedSummary ?? selectedDetail ?? null

    return (
        <div className="space-y-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight text-gray-900">News Digests</h1>
                    <p className="mt-0.5 text-sm text-gray-400">
                        {listQuery.data
                            ? `${listQuery.data.counts.digests} digests, ${listQuery.data.counts.stories} stories`
                            : "Loading digest library"}
                    </p>
                </div>

                <div className="flex items-center gap-2">
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".md,text/markdown,text/x-markdown"
                        className="hidden"
                        onChange={event => handleUpload(event.currentTarget.files?.[0])}
                    />
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                        Upload
                    </button>
                    <button
                        type="button"
                        onClick={() => listQuery.refetch()}
                        className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-500 shadow-sm transition-colors hover:bg-gray-50 hover:text-gray-700"
                        title="Refresh"
                    >
                        <RefreshCw className="h-4 w-4" />
                    </button>
                </div>
            </div>

            {uploadError && <ErrorMessage message={uploadError} />}
            {listQuery.isLoading && <LoadingSpinner message="Loading news digests..." />}
            {!listQuery.isLoading && listQuery.error && <ErrorMessage message={String(listQuery.error)} />}

            {!listQuery.isLoading && !listQuery.error && (
                <div className="grid grid-cols-1 gap-5 xl:grid-cols-[380px_minmax(0,1fr)]">
                    <aside className="space-y-4">
                        <div className="relative">
                            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                            <input
                                value={search}
                                onChange={event => setSearch(event.target.value)}
                                placeholder="Search stories"
                                className="w-full rounded-lg border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm text-gray-900 outline-none transition-colors placeholder:text-gray-400 focus:border-gray-400"
                            />
                        </div>

                        <div className="rounded-lg border border-gray-200 bg-white">
                            <div className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
                                <span className="text-xs font-semibold uppercase tracking-widest text-gray-400">Digests</span>
                                <span className="text-xs tabular-nums text-gray-400">{visibleDigests.length}</span>
                            </div>
                            <div className="max-h-[48vh] overflow-y-auto p-2">
                                {visibleDigests.length ? (
                                    <div className="space-y-2">
                                        {visibleDigests.map(digest => (
                                            <button
                                                key={digest.id}
                                                type="button"
                                                onClick={() => setSelectedId(digest.id)}
                                                className={cn(
                                                    "w-full rounded-lg border px-3 py-3 text-left transition-colors",
                                                    selectedId === digest.id
                                                        ? "border-gray-900 bg-gray-50"
                                                        : "border-gray-100 bg-white hover:border-gray-200 hover:bg-gray-50",
                                                )}
                                            >
                                                <div className="flex items-start gap-2">
                                                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-gray-400" />
                                                    <div className="min-w-0 flex-1">
                                                        <div className="line-clamp-2 text-sm font-semibold leading-5 text-gray-900">
                                                            {digest.title}
                                                        </div>
                                                        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-400">
                                                            <span className="inline-flex items-center gap-1">
                                                                <Calendar className="h-3.5 w-3.5" />
                                                                {formatDate(digest.generated_date)}
                                                            </span>
                                                            <span>{digest.story_count} stories</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="px-3 py-8 text-center text-sm text-gray-400">No digests</div>
                                )}
                            </div>
                        </div>

                        {normalizedSearch && (
                            <div className="rounded-lg border border-gray-200 bg-white">
                                <div className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
                                    <span className="text-xs font-semibold uppercase tracking-widest text-gray-400">Stories</span>
                                    <span className="text-xs tabular-nums text-gray-400">{filteredStories.length}</span>
                                </div>
                                <div className="max-h-[42vh] overflow-y-auto p-2">
                                    {filteredStories.length ? (
                                        <div className="space-y-2">
                                            {filteredStories.slice(0, 80).map(story => (
                                                <button
                                                    key={`${story.digest_id}-${story.id}`}
                                                    type="button"
                                                    onClick={() => story.digest_id && setSelectedId(story.digest_id)}
                                                    className="w-full rounded-lg border border-gray-100 bg-white px-3 py-2 text-left transition-colors hover:border-gray-200 hover:bg-gray-50"
                                                >
                                                    <div className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                                                        {story.section}
                                                    </div>
                                                    <div className="mt-1 text-sm font-medium leading-5 text-gray-900">
                                                        {story.headline}
                                                    </div>
                                                    {story.digest_title && (
                                                        <div className="mt-1 text-xs text-gray-400">{story.digest_title}</div>
                                                    )}
                                                </button>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="px-3 py-8 text-center text-sm text-gray-400">No matching stories</div>
                                    )}
                                </div>
                            </div>
                        )}
                    </aside>

                    <main className="min-w-0 rounded-lg border border-gray-200 bg-white">
                        {selectedId && detailQuery.isLoading && (
                            <div className="p-8">
                                <LoadingSpinner message="Loading digest..." />
                            </div>
                        )}
                        {selectedId && detailQuery.error && (
                            <div className="p-4">
                                <ErrorMessage message={String(detailQuery.error)} />
                            </div>
                        )}
                        {!selectedId && (
                            <div className="px-6 py-16 text-center text-sm text-gray-400">No digest selected</div>
                        )}
                        {selectedDetail && (
                            <div>
                                <div className="flex items-start justify-between gap-4 border-b border-gray-100 px-5 py-4">
                                    <div className="min-w-0">
                                        <h2 className="text-lg font-semibold text-gray-900">{selectedDetail.title}</h2>
                                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400">
                                            <span>{formatDate(selectedDetail.generated_date)}</span>
                                            <span>{selectedDetail.story_count} stories</span>
                                            <span>Updated {formatDate(selectedDetail.updated_at, true)}</span>
                                        </div>
                                    </div>
                                    {selectedDigestForDelete && (
                                        <button
                                            type="button"
                                            onClick={() => handleDelete(selectedDigestForDelete)}
                                            disabled={deletingId === selectedDigestForDelete.id}
                                            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-400 transition-colors hover:border-red-200 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-60"
                                            title="Delete"
                                        >
                                            {deletingId === selectedDigestForDelete.id ? (
                                                <Loader2 className="h-4 w-4 animate-spin" />
                                            ) : (
                                                <Trash2 className="h-4 w-4" />
                                            )}
                                        </button>
                                    )}
                                </div>
                                <div className="max-h-[72vh] overflow-y-auto px-5 py-5">
                                    <div className="prose prose-sm max-w-none">
                                        <MarkdownRenderer content={selectedDetail.content} />
                                    </div>
                                </div>
                            </div>
                        )}
                    </main>
                </div>
            )}
        </div>
    )
}
