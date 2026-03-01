import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Format a number as a percentage string with sign */
export function fmtPct(val: number | null | undefined, decimals = 1): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A"
  return `${val >= 0 ? "+" : ""}${val.toFixed(decimals)}%`
}

/** Format a number with fixed decimals */
export function fmtNum(val: number | null | undefined, decimals = 2): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A"
  return val.toFixed(decimals)
}

/** Format a large number with commas */
export function fmtInt(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A"
  return Math.round(val).toLocaleString()
}
