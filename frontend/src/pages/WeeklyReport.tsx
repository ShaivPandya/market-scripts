import { Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from "react"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { fetchWeeklyReportCached, generateWeeklyReport } from "@/lib/api"
import { Check, Copy } from "lucide-react"

interface WeeklyReportResponse {
    report: string
}

type ListType = "ul" | "ol"

function isTableSeparatorLine(line: string) {
    const trimmed = line.trim()
    if (!trimmed.includes("-") || !trimmed.includes("|")) return false
    return /^(\|?\s*:?-{3,}:?\s*)+\|?$/.test(trimmed.replace(/\|/g, "|"))
}

function splitTableRow(line: string) {
    const trimmed = line.trim()
    const noOuter = trimmed.replace(/^\|/, "").replace(/\|$/, "")
    return noOuter.split("|").map(cell => cell.trim())
}

function renderInlineMarkdown(text: string) {
    const nodes: ReactNode[] = []
    let i = 0
    let key = 0

    const pushText = (value: string) => {
        if (!value) return
        nodes.push(value)
    }

    while (i < text.length) {
        const rest = text.slice(i)

        if (rest.startsWith("`")) {
            const end = text.indexOf("`", i + 1)
            if (end !== -1) {
                const code = text.slice(i + 1, end)
                nodes.push(
                    <code key={`code-${key++}`} className="rounded bg-gray-50 px-1 py-0.5 text-[0.95em]">
                        {code}
                    </code>,
                )
                i = end + 1
                continue
            }
        }

        if (rest.startsWith("**")) {
            const end = text.indexOf("**", i + 2)
            if (end !== -1) {
                const bold = text.slice(i + 2, end)
                nodes.push(
                    <strong key={`strong-${key++}`}>{renderInlineMarkdown(bold)}</strong>,
                )
                i = end + 2
                continue
            }
        }

        if (rest.startsWith("*")) {
            const end = text.indexOf("*", i + 1)
            if (end !== -1) {
                const italic = text.slice(i + 1, end)
                nodes.push(
                    <em key={`em-${key++}`}>{renderInlineMarkdown(italic)}</em>,
                )
                i = end + 1
                continue
            }
        }

        if (rest.startsWith("[")) {
            const closeLabel = text.indexOf("]", i + 1)
            const openUrl = closeLabel !== -1 ? text[closeLabel + 1] : ""
            if (closeLabel !== -1 && openUrl === "(") {
                const closeUrl = text.indexOf(")", closeLabel + 2)
                if (closeUrl !== -1) {
                    const label = text.slice(i + 1, closeLabel)
                    const url = text.slice(closeLabel + 2, closeUrl).trim()
                    const safeUrl = /^https?:\/\//i.test(url) ? url : null
                    if (safeUrl) {
                        nodes.push(
                            <a
                                key={`a-${key++}`}
                                href={safeUrl}
                                target="_blank"
                                rel="noreferrer"
                            >
                                {label}
                            </a>,
                        )
                        i = closeUrl + 1
                        continue
                    }
                }
            }
        }

        const nextSpecial = (() => {
            const candidates = [
                text.indexOf("`", i),
                text.indexOf("**", i),
                text.indexOf("*", i),
                text.indexOf("[", i),
            ]
            const next = candidates.filter(x => x !== -1).sort((a, b) => a - b)[0]
            return next === undefined ? -1 : next
        })()

        if (nextSpecial === -1) {
            pushText(text.slice(i))
            break
        }
        pushText(text.slice(i, nextSpecial))
        i = nextSpecial
    }

    return nodes.length ? nodes : text
}

function renderMarkdownLite(markdown: string) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n")
    const blocks: ReactNode[] = []

    let inCodeBlock = false
    let codeFenceLang = ""
    let codeLines: string[] = []

    let pendingList: { type: ListType; items: string[] } | null = null

    const flushList = () => {
        if (!pendingList) return
        const list = pendingList
        pendingList = null

        const Tag = list.type
        blocks.push(
            <Tag key={`list-${blocks.length}`}>
                {list.items.map((item, idx) => (
                    <li key={`li-${idx}`}>{renderInlineMarkdown(item)}</li>
                ))}
            </Tag>,
        )
    }

    const flushCode = () => {
        if (!inCodeBlock) return
        blocks.push(
            <pre key={`pre-${blocks.length}`}>
                <code className={codeFenceLang ? `language-${codeFenceLang}` : undefined}>
                    {codeLines.join("\n")}
                </code>
            </pre>,
        )
        inCodeBlock = false
        codeFenceLang = ""
        codeLines = []
    }

    for (let idx = 0; idx < lines.length; idx += 1) {
        const rawLine = lines[idx]
        const line = rawLine ?? ""

        const fenceMatch = line.trim().match(/^```(\w+)?\s*$/)
        if (fenceMatch) {
            if (inCodeBlock) {
                flushCode()
            } else {
                flushList()
                inCodeBlock = true
                codeFenceLang = fenceMatch[1] ?? ""
                codeLines = []
            }
            continue
        }

        if (inCodeBlock) {
            codeLines.push(line)
            continue
        }

        if (!line.trim()) {
            flushList()
            continue
        }

        const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
        if (headingMatch) {
            flushList()
            const level = headingMatch[1].length
            const content = headingMatch[2] ?? ""
            const keyId = `h-${blocks.length}`
            const body = renderInlineMarkdown(content)
            const clamped = Math.min(6, Math.max(1, level))
            if (clamped === 1) blocks.push(<h1 key={keyId}>{body}</h1>)
            else if (clamped === 2) blocks.push(<h2 key={keyId}>{body}</h2>)
            else if (clamped === 3) blocks.push(<h3 key={keyId}>{body}</h3>)
            else if (clamped === 4) blocks.push(<h4 key={keyId}>{body}</h4>)
            else if (clamped === 5) blocks.push(<h5 key={keyId}>{body}</h5>)
            else blocks.push(<h6 key={keyId}>{body}</h6>)
            continue
        }

        // GitHub-style tables: header row + separator + rows
        const nextLine = lines[idx + 1] ?? ""
        if (line.includes("|") && isTableSeparatorLine(nextLine)) {
            flushList()
            const headerCells = splitTableRow(line)
            idx += 1 // skip separator line

            const rowLines: string[] = []
            while (idx + 1 < lines.length && (lines[idx + 1] ?? "").trim() && (lines[idx + 1] ?? "").includes("|")) {
                rowLines.push(lines[idx + 1] ?? "")
                idx += 1
            }

            const rows = rowLines.map(splitTableRow)
            blocks.push(
                <table key={`table-${blocks.length}`}>
                    <thead>
                        <tr>
                            {headerCells.map((c, i) => (
                                <th key={`th-${i}`}>{renderInlineMarkdown(c)}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row, r) => (
                            <tr key={`tr-${r}`}>
                                {row.map((c, i) => (
                                    <td key={`td-${r}-${i}`}>{renderInlineMarkdown(c)}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>,
            )
            continue
        }

        const ulMatch = line.match(/^\s*[-*]\s+(.*)$/)
        const olMatch = line.match(/^\s*(\d+)\.\s+(.*)$/)
        if (ulMatch) {
            if (!pendingList || pendingList.type !== "ul") {
                flushList()
                pendingList = { type: "ul", items: [] }
            }
            pendingList.items.push(ulMatch[1] ?? "")
            continue
        }
        if (olMatch) {
            if (!pendingList || pendingList.type !== "ol") {
                flushList()
                pendingList = { type: "ol", items: [] }
            }
            pendingList.items.push(olMatch[2] ?? "")
            continue
        }

        flushList()
        blocks.push(
            <p key={`p-${blocks.length}`}>{renderInlineMarkdown(line)}</p>,
        )
    }

    flushList()
    flushCode()

    return blocks.length ? blocks : <p>{markdown}</p>
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
    const [isCheckingCache, setIsCheckingCache] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [data, setData] = useState<WeeklyReportResponse | null>(null)
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
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err))
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        const fetchId = ++cacheFetchIdRef.current
        async function loadCached() {
            setIsCheckingCache(true)
            try {
                const cached = await fetchWeeklyReportCached()
                if (cacheFetchIdRef.current !== fetchId) return
                if (cached) setData(cached)
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

            {(isLoading || isCheckingCache) && (
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

            {data && !isLoading && !isCheckingCache && (
                <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
                    <div className="px-6 py-8 md:px-8">
                        <div className="prose prose-blue prose-sm sm:prose-base max-w-none break-words prose-headings:font-semibold prose-a:text-blue-600 prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:rounded-lg prose-pre:px-4 prose-pre:py-3 prose-table:border-collapse prose-th:px-4 prose-th:py-2 prose-th:bg-gray-50 prose-th:border prose-th:border-gray-200 prose-td:px-4 prose-td:py-2 prose-td:border prose-td:border-gray-200">
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
