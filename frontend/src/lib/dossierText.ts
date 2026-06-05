// Assistant-export citation/navigation tokens (e.g. from ChatGPT web/file search) wrap
// references in Unicode Private Use Area delimiters: U+E200 (start), U+E202 (separator),
// U+E201 (end) -- producing blocks like "\ue200cite\ue202turn8view0\ue202turn3view0\ue201".
// The delimiters have no glyph, so pasted markdown renders the leftover literal
// "citeturn8view0turn3view0" (often beside tofu boxes). Strip them everywhere we display
// ingested text. Genuine prose such as the word "cited" is preserved.
const CITATION_BLOCK_RE = / ?\ue200[\s\S]*?\ue201/g
const CITATION_BRACKETED_RE = /【[^】]*?(?:cite\s*)?turn\d+[a-z]+\d+[^】]*?】/gi
const CITATION_BARE_RE = / ?(?:cite|navlist|filecite|videocite)?(?:turn\d+[a-z]+\d+)+/gi
const CITATION_STRAY_RE = /[\ue200-\ue20f]/g

// Removes citation/navigation tokens while preserving surrounding markdown structure.
// Safe to run over full markdown documents (headings, tables, emphasis are untouched).
export function stripCitationTokens(value: unknown): string {
  const text = String(value ?? "")
  if (!text) return ""
  return text
    .replace(CITATION_BLOCK_RE, "")
    .replace(CITATION_BRACKETED_RE, "")
    .replace(CITATION_BARE_RE, "")
    .replace(CITATION_STRAY_RE, "")
}

export function cleanDossierDisplayText(value: unknown): string {
  let text = stripCitationTokens(value).trim()
  if (!text) return ""

  text = text
    .replace(/\r\n/g, "\n")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
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
