import { type ReactNode } from "react"

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

export function renderInlineMarkdown(text: string) {
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
            pushText("`")
            i += 1
            continue
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
            pushText("**")
            i += 2
            continue
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
            pushText("*")
            i += 1
            continue
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
                                className="text-blue-600 underline underline-offset-2 hover:text-blue-700"
                            >
                                {label}
                            </a>,
                        )
                        i = closeUrl + 1
                        continue
                    }
                }
            }
            pushText("[")
            i += 1
            continue
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

export function renderMarkdownLite(markdown: string) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n")
    const blocks: ReactNode[] = []

    let inCodeBlock = false
    let codeFenceLang = ""
    let codeLines: string[] = []

    type ListNode = { type: ListType; items: ListItem[] }
    type ListItem = { text: string; children?: ListNode }
    type ListFrame = { indent: number; node: ListNode; lastItem: ListItem | null }

    let listRoot: ListNode | null = null
    let listStack: ListFrame[] = []

    const renderListNode = (node: ListNode, depth: number): ReactNode => {
        const Tag = node.type
        const marker = node.type === "ul" ? "list-disc" : "list-decimal"
        return (
            <Tag
                className={[
                    depth === 0 ? "my-3" : "my-2",
                    "pl-5",
                    marker,
                    "space-y-1",
                    "text-sm sm:text-base text-gray-900",
                ].join(" ")}
            >
                {node.items.map((item, itemIdx) => (
                    <li key={`li-${depth}-${itemIdx}`} className="leading-6">
                        <span>{renderInlineMarkdown(item.text)}</span>
                        {item.children ? renderListNode(item.children, depth + 1) : null}
                    </li>
                ))}
            </Tag>
        )
    }

    const flushList = () => {
        if (!listRoot) return
        const root = listRoot
        listRoot = null
        listStack = []
        blocks.push(
            <div key={`list-${blocks.length}`}>
                {renderListNode(root, 0)}
            </div>,
        )
    }

    const startListAt = (indent: number, type: ListType) => {
        const node: ListNode = { type, items: [] }
        listRoot = node
        listStack = [{ indent, node, lastItem: null }]
        return listStack[0]
    }

    const addListItem = (indent: number, type: ListType, text: string) => {
        const normalizedIndent = Math.max(0, indent)

        if (!listRoot || listStack.length === 0) {
            startListAt(normalizedIndent, type)
        }

        while (listStack.length > 0 && normalizedIndent < listStack[listStack.length - 1].indent) {
            listStack.pop()
        }

        if (listStack.length === 0) {
            startListAt(normalizedIndent, type)
        }

        let frame = listStack[listStack.length - 1]

        if (normalizedIndent > frame.indent) {
            const parentItem = frame.lastItem
            if (parentItem) {
                if (!parentItem.children || parentItem.children.type !== type) {
                    parentItem.children = { type, items: [] }
                }
                const child = parentItem.children
                frame = { indent: normalizedIndent, node: child, lastItem: null }
                listStack.push(frame)
            }
        } else if (frame.node.type !== type) {
            flushList()
            frame = startListAt(normalizedIndent, type)
        }

        const item: ListItem = { text }
        frame.node.items.push(item)
        frame.lastItem = item
    }

    const flushCode = () => {
        if (!inCodeBlock) return
        blocks.push(
            <pre
                key={`pre-${blocks.length}`}
                className="my-4 overflow-x-auto rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs sm:text-sm text-gray-900"
            >
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
            if (clamped === 1) blocks.push(<h1 key={keyId} className="mt-0 mb-3 text-xl sm:text-2xl font-semibold text-gray-900">{body}</h1>)
            else if (clamped === 2) blocks.push(<h2 key={keyId} className="mt-7 mb-2 text-lg sm:text-xl font-semibold text-gray-900">{body}</h2>)
            else if (clamped === 3) blocks.push(<h3 key={keyId} className="mt-6 mb-2 text-base sm:text-lg font-semibold text-gray-900">{body}</h3>)
            else if (clamped === 4) blocks.push(<h4 key={keyId} className="mt-5 mb-2 text-sm sm:text-base font-semibold text-gray-900">{body}</h4>)
            else if (clamped === 5) blocks.push(<h5 key={keyId} className="mt-4 mb-2 text-sm font-semibold text-gray-900">{body}</h5>)
            else blocks.push(<h6 key={keyId} className="mt-4 mb-2 text-sm font-semibold text-gray-900">{body}</h6>)
            continue
        }

        const nextLine = lines[idx + 1] ?? ""
        if (line.includes("|") && isTableSeparatorLine(nextLine)) {
            flushList()
            const headerCells = splitTableRow(line)
            idx += 1

            const rowLines: string[] = []
            while (idx + 1 < lines.length && (lines[idx + 1] ?? "").trim() && (lines[idx + 1] ?? "").includes("|")) {
                rowLines.push(lines[idx + 1] ?? "")
                idx += 1
            }

            const rows = rowLines.map(splitTableRow)
            blocks.push(
                <div key={`tablewrap-${blocks.length}`} className="my-4 -mx-2 overflow-x-auto">
                    <div className="px-2">
                        <table className="w-full border-collapse text-sm text-gray-900">
                            <thead>
                                <tr className="bg-gray-50">
                                    {headerCells.map((c, i) => (
                                        <th
                                            key={`th-${i}`}
                                            className="border border-gray-200 px-3 py-2 text-left font-semibold"
                                        >
                                            {renderInlineMarkdown(c)}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {rows.map((row, r) => (
                                    <tr key={`tr-${r}`} className="odd:bg-white even:bg-gray-50/40">
                                        {row.map((c, i) => (
                                            <td
                                                key={`td-${r}-${i}`}
                                                className="border border-gray-200 px-3 py-2 align-top"
                                            >
                                                {renderInlineMarkdown(c)}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>,
            )
            continue
        }

        const ulMatch = line.match(/^(\s*)[-*]\s+(.*)$/)
        const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/)
        if (ulMatch) {
            const indent = (ulMatch[1] ?? "").replace(/\t/g, "    ").length
            addListItem(indent, "ul", ulMatch[2] ?? "")
            continue
        }
        if (olMatch) {
            const indent = (olMatch[1] ?? "").replace(/\t/g, "    ").length
            addListItem(indent, "ol", olMatch[3] ?? "")
            continue
        }

        flushList()
        blocks.push(
            <p key={`p-${blocks.length}`} className="my-2 text-sm sm:text-base leading-6 text-gray-900 whitespace-pre-wrap">
                {renderInlineMarkdown(line)}
            </p>,
        )
    }

    flushList()
    flushCode()

    return blocks.length ? blocks : <p>{markdown}</p>
}

export function MarkdownRenderer({ content }: { content: string }) {
    return <div>{renderMarkdownLite(content)}</div>
}
