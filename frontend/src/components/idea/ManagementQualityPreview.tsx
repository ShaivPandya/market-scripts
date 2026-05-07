import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import type { ParsedManagementQuality } from "@/lib/managementQualityTypes"
import { cn } from "@/lib/utils"

function cleanManagementRatingText(value?: string | null) {
  return String(value || "").replace(/[*_`~]+/g, "").trim()
}

function canonicalManagementRating(value?: string | null) {
  const normalized = cleanManagementRatingText(value).toLowerCase().replace(/\s+/g, " ")
  if (normalized === "strong") return "Strong"
  if (normalized === "mixed") return "Mixed"
  if (normalized === "weak") return "Weak"
  if (normalized === "insufficient evidence") return "Insufficient evidence"
  return null
}

function splitManagementRatingText(value?: string | null) {
  const raw = String(value || "").trim()
  const match = raw.match(/^\s*(?:[*_`~]+)?\s*(Strong|Mixed|Weak|Insufficient evidence)\b\s*(?:[*_`~]+)?\s*(?:(?:[:—–-]+)\s*(.+)|\s+(.+))?$/i)
  const rating = canonicalManagementRating(match?.[1])
  return {
    rating,
    text: rating ? (match?.[2] || match?.[3] || "").trim() : raw,
  }
}

function resolveManagementSummaryQuestion(item?: { rating?: string | null; text?: string | null } | null) {
  const ratingValue = splitManagementRatingText(item?.rating)
  const textValue = splitManagementRatingText(item?.text)
  const rating = ratingValue.rating || textValue.rating || item?.rating || null
  const text = textValue.rating ? textValue.text : item?.text
  return { rating, text }
}

function ManagementRatingBadge({ value }: { value?: string | null }) {
  const rating = splitManagementRatingText(value).rating || cleanManagementRatingText(value) || "Insufficient evidence"
  const normalized = rating.toLowerCase()
  const className = normalized.includes("strong")
    ? "border-green-200 bg-green-50 text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-300"
    : normalized.includes("weak") || normalized.includes("poor")
      ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
      : normalized.includes("mixed") || normalized.includes("too early")
        ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
        : "border-app bg-card-muted text-muted"

  return <span className={cn("rounded border px-2 py-0.5 text-xs font-semibold", className)}>{rating}</span>
}

export function ManagementQualityPreview({ parsed, content }: { parsed?: ParsedManagementQuality | null; content: string }) {
  const summary = parsed?.summary ?? null
  const scorecard = parsed?.scorecard ?? []
  const hasStructured = Boolean(summary || scorecard.length)

  return (
    <div className="space-y-4">
      {summary && (
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="border-l border-app pl-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h4 className="section-title text-xs">Overall</h4>
              <ManagementRatingBadge value={summary.overall_rating} />
            </div>
            {summary.bottom_line && <p className="text-sm leading-6 text-muted">{summary.bottom_line}</p>}
          </div>
          {[
            ["Owner Mindset", summary.owner_mindset],
            ["Business Value", summary.business_value_understanding],
            ["Follow-through", summary.follow_through],
          ].map(([label, item]) => {
            const row = item as { rating?: string | null; text?: string | null } | undefined
            const resolved = resolveManagementSummaryQuestion(row)
            return (
              <div key={String(label)} className="border-l border-app pl-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <h4 className="section-title text-xs">{String(label)}</h4>
                  <ManagementRatingBadge value={resolved.rating} />
                </div>
                {resolved.text && <p className="text-sm leading-6 text-muted">{resolved.text}</p>}
              </div>
            )
          })}
        </div>
      )}
      {scorecard.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-app">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-app text-xs uppercase text-subtle">
              <tr>
                <th className="px-3 py-2 font-semibold">Question</th>
                <th className="px-3 py-2 font-semibold">Rating</th>
                <th className="px-3 py-2 font-semibold">Evidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[hsl(var(--border))]">
              {scorecard.map(row => (
                <tr key={row.question}>
                  <td className="px-3 py-2 text-app">{row.question}</td>
                  <td className="px-3 py-2"><ManagementRatingBadge value={row.rating} /></td>
                  <td className="px-3 py-2 text-muted">{row.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className={cn("prose prose-sm dark:prose-invert max-w-none", hasStructured && "border-t border-app pt-3")}>
        <MarkdownRenderer content={content} />
      </div>
    </div>
  )
}
