export const GROUP_CONVICTION_MIN = 1
export const GROUP_CONVICTION_MAX = 5
const UNGROUPED_SENTINELS = new Set(["-", "na", "n/a", "none", "null", "ungrouped"])

export function normalizeGroupName(value: unknown): string | null {
  const text = String(value ?? "")
    .normalize("NFC")
    .replace(/\s+/g, " ")
    .trim()
  if (UNGROUPED_SENTINELS.has(text.toLocaleLowerCase())) return null
  return text || null
}

export function groupKey(value: unknown): string | null {
  return normalizeGroupName(value)?.toLocaleLowerCase() ?? null
}

export function normalizeGroupConviction(value: unknown): number | null {
  if (value == null || value === "") return null
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return null
  const rounded = Math.round(numeric)
  if (rounded < GROUP_CONVICTION_MIN || rounded > GROUP_CONVICTION_MAX) return null
  return rounded
}
