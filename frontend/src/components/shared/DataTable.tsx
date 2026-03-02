import { useMemo } from "react"

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
}

export function DataTable({ columns, rows, maxHeight = "600px" }: DataTableProps) {
  const displayColumns = useMemo(
    () => columns.filter(c => rows.some(r => c.key in r)),
    [columns, rows],
  )

  if (rows.length === 0) {
    return <p className="text-sm text-gray-400 py-4">No data available.</p>
  }

  return (
    <div style={{ maxHeight, overflowY: "auto" }} className="rounded-xl border border-gray-200 overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead className="sticky top-0 bg-gray-50 z-10">
          <tr>
            {displayColumns.map(col => (
              <th
                key={col.key}
                className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap"
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
              className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
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
                    className="px-3 py-2 whitespace-nowrap"
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
  )
}
