const CHUNK_RELOAD_PREFIX = "talisman:chunk-reload"

const CHUNK_LOAD_ERROR_PATTERNS = [
  /ChunkLoadError/i,
  /CSS_CHUNK_LOAD_FAILED/i,
  /Failed to fetch dynamically imported module/i,
  /error loading dynamically imported module/i,
  /Importing a module script failed/i,
  /Loading module .* failed/i,
  /Unable to preload CSS/i,
]

function entryModuleSignature() {
  if (typeof document === "undefined") return "unknown-entry"

  const entryScript = Array.from(document.scripts).find(script =>
    script.type === "module" && script.src.includes("/assets/index-"),
  )

  return entryScript?.src || "unknown-entry"
}

function reloadKey(reason: string) {
  const path = typeof window === "undefined" ? "unknown-path" : window.location.pathname
  return `${CHUNK_RELOAD_PREFIX}:${reason}:${entryModuleSignature()}:${path}`
}

export function isChunkLoadError(error: unknown) {
  const message =
    error instanceof Error
      ? `${error.name} ${error.message}`
      : typeof error === "string"
        ? error
        : ""

  return CHUNK_LOAD_ERROR_PATTERNS.some(pattern => pattern.test(message))
}

export function reloadOnceForStaleAssetLoad(reason: string, detail?: unknown) {
  if (typeof window === "undefined") return false

  const key = reloadKey(reason)
  try {
    if (window.sessionStorage.getItem(key) === "1") return false
    window.sessionStorage.setItem(key, "1")
  } catch {
    return false
  }

  console.warn("Reloading after failed route asset load.", detail)
  window.location.reload()
  return true
}

export function maybeReloadForChunkLoadError(error: unknown, reason: string) {
  if (!isChunkLoadError(error)) return false
  return reloadOnceForStaleAssetLoad(reason, error)
}
