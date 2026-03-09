import { Fragment, useEffect, useMemo, useRef, useState } from "react"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { renderMarkdownLite } from "@/components/shared/MarkdownRenderer"
import { fetchWeeklyReportCached, generateWeeklyReport } from "@/lib/api"
import { Check, Copy } from "lucide-react"

interface WeeklyReportResponse {
    report: string
}

const WEEKLY_REPORT_LOCAL_KEY = "weekly_report:last"

function loadLocalWeeklyReport(): WeeklyReportResponse | null {
    try {
        const raw = window.localStorage.getItem(WEEKLY_REPORT_LOCAL_KEY)
        if (!raw) return null
        const parsed = JSON.parse(raw) as { report?: unknown }
        if (typeof parsed?.report !== "string" || !parsed.report.trim()) return null
        return { report: parsed.report }
    } catch {
        return null
    }
}

function saveLocalWeeklyReport(report: string) {
    try {
        if (!report.trim()) return
        window.localStorage.setItem(WEEKLY_REPORT_LOCAL_KEY, JSON.stringify({ report }))
    } catch {
        // ignore
    }
}

async function copyToClipboard(text: string) {
    if (!text) return false
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text)
            return true
        }
    } catch {
        // fall back below
    }

    try {
        const el = document.createElement("textarea")
        el.value = text
        el.setAttribute("readonly", "true")
        el.style.position = "fixed"
        el.style.top = "0"
        el.style.left = "0"
        el.style.opacity = "0"
        document.body.appendChild(el)
        el.focus()
        el.select()
        const ok = document.execCommand("copy")
        document.body.removeChild(el)
        return ok
    } catch {
        return false
    }
}

export function WeeklyReport() {
    const [isLoading, setIsLoading] = useState(false)
    const [isCheckingCache, setIsCheckingCache] = useState(() => !loadLocalWeeklyReport())
    const [error, setError] = useState<string | null>(null)
    const [data, setData] = useState<WeeklyReportResponse | null>(() => loadLocalWeeklyReport())
    const [copied, setCopied] = useState(false)
    const cacheFetchIdRef = useRef(0)

    const handleGenerate = async () => {
        cacheFetchIdRef.current += 1
        setIsCheckingCache(false)
        setIsLoading(true)
        setError(null)
        setCopied(false)
        try {
            const json = await generateWeeklyReport()
            setData(json)
            saveLocalWeeklyReport(json.report)
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err))
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        const fetchId = ++cacheFetchIdRef.current
        async function loadCached() {
            setIsCheckingCache(!loadLocalWeeklyReport())
            try {
                const cached = await fetchWeeklyReportCached()
                if (cacheFetchIdRef.current !== fetchId) return
                if (cached) {
                    setData(cached)
                    saveLocalWeeklyReport(cached.report)
                }
            } catch (err: unknown) {
                if (cacheFetchIdRef.current !== fetchId) return
                setError(err instanceof Error ? err.message : String(err))
            } finally {
                if (cacheFetchIdRef.current !== fetchId) return
                setIsCheckingCache(false)
            }
        }
        loadCached()
        return () => {
            cacheFetchIdRef.current += 1
        }
    }, [])

    const rendered = useMemo(() => {
        if (!data?.report) return null
        return renderMarkdownLite(data.report)
    }, [data?.report])

    const handleCopy = async () => {
        if (!data?.report) return
        const ok = await copyToClipboard(data.report)
        if (!ok) return
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1500)
    }

    return (
        <div className="max-w-4xl mx-auto pb-12">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Weekly Report</h1>
                    <p className="mt-2 text-sm text-gray-500">
                        A summary of all notable market moves from the past week.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        type="button"
                        onClick={handleCopy}
                        disabled={!data?.report || isLoading}
                        className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-200 bg-white text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                        title={data?.report ? "Copy report markdown" : "Generate a report first"}
                    >
                        {copied ? (
                            <Fragment>
                                <Check className="h-4 w-4" />
                                Copied
                            </Fragment>
                        ) : (
                            <Fragment>
                                <Copy className="h-4 w-4" />
                                Copy
                            </Fragment>
                        )}
                    </button>
                    <button
                        type="button"
                        onClick={handleGenerate}
                        disabled={isLoading}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? "Generating..." : "Generate Report"}
                    </button>
                </div>
            </div>

            {(isLoading || (isCheckingCache && !data)) && (
                <div className="bg-white p-8 rounded-xl border shadow-sm flex flex-col items-center justify-center min-h-[400px]">
                    <LoadingSpinner />
                    <p className="mt-4 text-sm text-gray-500 animate-pulse">
                        {isLoading
                            ? "Compiling index, commodity, FX, and technical data. This may take a minute..."
                            : "Loading cached report..."}
                    </p>
                </div>
            )}

            {!isLoading && !isCheckingCache && error && (
                <div className="bg-white p-6 rounded-xl border shadow-sm">
                    <ErrorMessage message={String(error)} />
                </div>
            )}

            {data && !isLoading && (
                <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
                    <div className="px-6 py-8 md:px-8">
                        <div className="max-w-none break-words">
                            {rendered ?? <p>{data.report}</p>}
                        </div>
                    </div>
                </div>
            )}

            {!data && !isLoading && !isCheckingCache && !error && (
                <div className="bg-white p-12 rounded-xl border shadow-sm text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <h3 className="mt-2 text-sm font-medium text-gray-900">No report generated</h3>
                    <p className="mt-1 text-sm text-gray-500">
                        Click the button above to fetch data and generate a new summary.
                    </p>
                </div>
            )}
        </div>
    )
}
