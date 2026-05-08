const CITATION_ARTIFACT_RE = /\b(?:cite\s*)?turn\d+(?:view|search)\d+\b/gi

export function cleanDossierDisplayText(value: unknown): string {
  let text = String(value ?? "").trim()
  if (!text) return ""

  text = text
    .replace(/\r\n/g, "\n")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/【[^】]*?(?:cite\s*)?turn\d+(?:view|search)\d+[^】]*?】/gi, "")
    .replace(CITATION_ARTIFACT_RE, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/[*_`~]+/g, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([.;:!?])\s*,+/g, "$1")
    .replace(/^[,;]\s*/g, "")
    .replace(/\s+[,;]\s+/g, " ")
    .replace(/,\s*([.;:!?])/g, "$1")
    .replace(/(?:\s*,\s*){2,}/g, ", ")
    .replace(/\s*,\s*$/g, "")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .replace(/\[\s+/g, "[")
    .replace(/\s+\]/g, "]")
    .replace(/\s{2,}/g, " ")
    .trim()

  if (["-", "--", "\u2014", "\u2013"].includes(text)) return ""
  return text
}
