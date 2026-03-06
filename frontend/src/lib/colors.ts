/**
 * Cell color functions for data tables
 * in gui/app.py lines 270-356.
 *
 * Each function returns a CSS color string (e.g. "#00c853") or "" for no color.
 * Use as: style={{ color: colorPositiveNegative(val) }}
 */

const GREEN_BOLD = "#00c853"
const GREEN = "#00c853"
const RED_BOLD = "#ff1744"
const RED = "#ff1744"
const YELLOW = "#ffc107"
const GRAY = "gray"

/** Positive → green, negative → red, zero/null → "" */
export function colorPositiveNegative(val: unknown): string {
  if (val === null || val === undefined) return GRAY
  let num: number
  if (typeof val === "string") {
    const cleaned = val.replace(/%/g, "").replace(/\+/g, "").trim()
    if (cleaned === "" || cleaned === "N/A") return GRAY
    num = parseFloat(cleaned)
  } else {
    num = val as number
  }
  if (isNaN(num)) return ""
  if (num > 0) return GREEN_BOLD
  if (num < 0) return RED_BOLD
  return ""
}

/** Z-score colouring: ≥1 bold green, >0 green, ≤-1 bold red, <0 red, ~0 yellow */
export function colorZscore(val: unknown): string {
  if (val === null || val === undefined) return GRAY
  let num: number
  if (typeof val === "string") {
    if (val === "N/A") return GRAY
    num = parseFloat(val.replace(/\+/g, ""))
  } else {
    num = val as number
  }
  if (isNaN(num)) return GRAY
  if (num >= 1) return `${GREEN_BOLD}; font-weight: bold`
  if (num > 0) return GREEN
  if (num <= -1) return `${RED_BOLD}; font-weight: bold`
  if (num < 0) return RED
  return YELLOW
}

/** "YES" → green, anything else → gray */
export function colorSignalFlag(val: unknown): string {
  if (val === "YES") return `${GREEN_BOLD}; font-weight: bold`
  return GRAY
}

/**
 * Return-vs-benchmark colouring.
 * Strings containing "(+)" → green, "(-)" → red.
 * Plain numeric strings → sign-based coloring.
 */
export function colorReturnVsBenchmark(val: unknown): string {
  if (val === null || val === undefined || val === "N/A") return GRAY
  if (typeof val === "string") {
    if (val.includes("(+)")) return `${GREEN_BOLD}; font-weight: bold`
    if (val.includes("(-)")) return `${RED_BOLD}; font-weight: bold`
    // Plain numeric
    try {
      const num = parseFloat(val.replace(/%/g, "").replace(/\+/g, "").split(" ")[0])
      if (!isNaN(num)) {
        if (num > 0) return GREEN
        if (num < 0) return RED
      }
    } catch {
      // ignore
    }
  }
  return ""
}

/** short_covering → green, long_liquidation → red */
export function colorForcedFlow(val: unknown): string {
  if (!val || val === "N/A") return GRAY
  if (val === "short_covering") return `${GREEN_BOLD}; font-weight: bold`
  if (val === "long_liquidation") return `${RED_BOLD}; font-weight: bold`
  return ""
}

/** VIX signals */
export function colorVixSignal(val: unknown): string {
  if (val === "Fear") return `${RED_BOLD}; font-weight: bold`
  if (val === "Complacency") return `${YELLOW}; font-weight: bold`
  return GRAY
}

/**
 * Polarity-aware coloring for the Liquidity changes table.
 * polarity=1 → increase is good (green); polarity=-1 → decrease is good.
 */
export function colorPolarityChange(val: unknown, polarity: number): string {
  if (val === null || val === undefined || val === "N/A") return GRAY
  let num: number
  if (typeof val === "string") {
    num = parseFloat(val.replace(/[B%+]/g, "").trim())
  } else {
    num = val as number
  }
  if (isNaN(num)) return GRAY
  const effective = num * polarity
  if (effective > 0) return `${GREEN_BOLD}; font-weight: bold`
  if (effective < 0) return `${RED_BOLD}; font-weight: bold`
  return ""
}

/** Sentiment: bullish → green, bearish → red, neutral → gray */
export function colorSentiment(val: unknown): string {
  if (val === "bullish") return GREEN_BOLD
  if (val === "bearish") return RED_BOLD
  return GRAY
}

/** Economic signal for industry monitor */
export function colorEconomicSignal(val: unknown): string {
  if (val === "expanding") return GREEN_BOLD
  if (val === "slowing") return YELLOW
  if (val === "contracting") return RED_BOLD
  return GRAY
}
