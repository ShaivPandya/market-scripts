import { memo, useCallback, useMemo, useState } from "react"

export interface ColumnDef {
  key: string
  header: string
  /** Return a CSS color string or "" for no color. */
  colorFn?: (val: unknown, row: Record<string, unknown>) => string
  format?: (val: unknown) => string
  width?: string
}

interface DataTableProps {
  columns: ColumnDef[]
  rows: Record<string, unknown>[]
  maxHeight?: string
  label?: string
  onRowClick?: (row: Record<string, unknown>) => void
}

export const DataTable = memo(function DataTable({ columns, rows, maxHeight = "600px", label, onRowClick }: DataTableProps) {
  const [copied, setCopied] = useState(false)

  const displayColumns = useMemo(
    () => columns.filter(c => rows.some(r => c.key in r)),
    [columns, rows],
  )

  const handleCopy = useCallback(() => {
    const header = displayColumns.map(c => c.header).join("\t")
    const body = rows.map(row =>
      displayColumns.map(col => {
        const raw = row[col.key]
        const display = col.format ? col.format(raw) : (raw ?? "")
        return String(display).replace(/\t/g, " ")
      }).join("\t"),
    ).join("\n")
    const text = label ? `${label}\n${header}\n${body}` : `${header}\n${body}`
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }).catch(() => {
      // Fallback for environments where Clipboard API is blocked.
      const ta = document.createElement("textarea")
      ta.value = text
      ta.style.position = "fixed"
      ta.style.opacity = "0"
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      try {
        document.execCommand("copy")
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } finally {
        document.body.removeChild(ta)
      }
    })
  }, [displayColumns, rows, label])

  if (rows.length === 0) {
    return <p className="py-4 text-sm text-subtle">No data available.</p>
  }

  const copyButton = (
    <button
      type="button"
      onClick={handleCopy}
      className="theme-button-secondary inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium"
    >
      {copied ? (
        <>
          <svg className="w-3.5 h-3.5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
          Copied
        </>
      ) : (
        <>
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
          Copy
        </>
      )}
    </button>
  )

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        {label ? <h3 className="text-sm font-semibold text-app">{label}</h3> : <span />}
        {copyButton}
      </div>
      <div
        style={{ maxHeight, overflowY: "auto" }}
        className="overflow-x-auto rounded-xl border border-app bg-card"
      >
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-muted-surface">
            <tr>
              {displayColumns.map(col => (
                <th
                  key={col.key}
                  className="whitespace-nowrap border-b border-app px-3 py-2 text-left font-semibold text-muted"
                  style={col.width ? { width: col.width } : undefined}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr
                key={ri}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={`border-b border-app transition-colors hover:bg-muted-surface${onRowClick ? " cursor-pointer" : ""}`}
              >
                {displayColumns.map(col => {
                  const raw = row[col.key]
                  const display = col.format ? col.format(raw) : (raw ?? "N/A")
                  const colorStr = col.colorFn ? col.colorFn(raw, row) : ""

                  // Parse "color; font-weight: bold" format from colors.ts
                  let color = ""
                  let fontWeight: string | undefined
                  if (colorStr) {
                    const parts = colorStr.split(";").map(s => s.trim())
                    color = parts[0] || ""
                    if (parts.some(p => p.includes("bold"))) fontWeight = "bold"
                  }

                  return (
                    <td
                      key={col.key}
                      className="whitespace-nowrap px-3 py-2"
                      style={{
                        color: color || undefined,
                        fontWeight: fontWeight,
                      }}
                    >
                      {String(display)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
})
