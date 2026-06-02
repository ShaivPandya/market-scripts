import * as Sentry from "@sentry/react"

const DROP_KEYS = new Set([
  "authorization",
  "cookie",
  "cookies",
  "password",
  "prompt",
  "prompts",
  "messages",
  "payload",
  "result",
  "body",
  "csrf",
  "csrfToken",
  "x-csrf-token",
  "x-api-proxy-secret",
  "holdings",
  "positions",
  "portfolio",
  "thesis",
])

function envBool(name: keyof ImportMetaEnv, defaultValue: boolean): boolean {
  const raw = import.meta.env[name]
  if (raw === undefined || raw === "") return defaultValue
  const normalized = String(raw).trim().toLowerCase()
  if (["1", "true", "yes", "on", "enabled"].includes(normalized)) return true
  if (["0", "false", "no", "off", "disabled"].includes(normalized)) return false
  return defaultValue
}

function envFloat(name: keyof ImportMetaEnv, defaultValue: number): number {
  const raw = import.meta.env[name]
  if (raw === undefined || raw === "") return defaultValue
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) return defaultValue
  return Math.max(0, Math.min(1, parsed))
}

function stripQuery(url: string): string {
  try {
    const parsed = new URL(url, window.location.origin)
    parsed.search = ""
    parsed.hash = ""
    return parsed.toString()
  } catch {
    return url.split("?")[0] ?? url
  }
}

function scrubValue(value: unknown): unknown {
  if (typeof value === "string") {
    return value.length > 500 ? `${value.slice(0, 500)}…` : value
  }
  if (Array.isArray(value)) {
    return value.slice(0, 5).map(scrubValue)
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) {
      const lowered = key.toLowerCase()
      if (DROP_KEYS.has(lowered) || [...DROP_KEYS].some(part => lowered.includes(part))) {
        out[key] = "[REDACTED]"
      } else if (typeof item === "object" && item !== null) {
        out[key] = scrubValue(item)
      } else {
        out[key] = item
      }
    }
    return out
  }
  return value
}

function scrubContexts(contexts: Sentry.ErrorEvent["contexts"]): Sentry.ErrorEvent["contexts"] {
  if (!contexts) return contexts
  return Object.fromEntries(
    Object.entries(contexts).map(([key, value]) => [key, scrubValue(value) as Record<string, unknown>]),
  )
}

function scrubEvent(event: Sentry.ErrorEvent): Sentry.ErrorEvent | null {
  const next = { ...event }
  if (next.request) {
    next.request = {
      ...next.request,
      url: next.request.url ? stripQuery(next.request.url) : next.request.url,
      headers: undefined,
      data: undefined,
      cookies: undefined,
    }
  }
  if (next.extra) {
    next.extra = scrubValue(next.extra) as Record<string, unknown>
  }
  if (next.contexts) {
    next.contexts = scrubContexts(next.contexts)
  }
  return next
}

function scrubBreadcrumb(breadcrumb: Sentry.Breadcrumb): Sentry.Breadcrumb | null {
  if (breadcrumb.category === "xhr" || breadcrumb.category === "fetch") {
    return {
      ...breadcrumb,
      data: breadcrumb.data
        ? (scrubValue({
            ...breadcrumb.data,
            url: typeof breadcrumb.data.url === "string" ? stripQuery(breadcrumb.data.url) : breadcrumb.data.url,
            body: undefined,
            request_body: undefined,
            response_body: undefined,
          }) as Record<string, unknown>)
        : undefined,
    }
  }
  return breadcrumb
}

export function sentryFrontendEnabled(): boolean {
  const dsn = (import.meta.env.VITE_SENTRY_DSN ?? "").trim()
  if (!envBool("VITE_SENTRY_ENABLED", true)) return false
  return Boolean(dsn)
}

export function initFrontendObservability(): void {
  const dsn = (import.meta.env.VITE_SENTRY_DSN ?? "").trim()
  if (!dsn || !envBool("VITE_SENTRY_ENABLED", true)) return

  const environment =
    (import.meta.env.VITE_SENTRY_ENVIRONMENT ?? "").trim() || import.meta.env.MODE || "development"
  const release =
    (import.meta.env.VITE_SENTRY_RELEASE ?? "").trim() ||
    (import.meta.env.VITE_TALISMAN_RELEASE_GIT_SHA_SHORT ?? "").trim() ||
    undefined

  Sentry.init({
    dsn,
    environment,
    release,
    enabled: import.meta.env.PROD || envBool("VITE_SENTRY_FORCE_DEV", false),
    sendDefaultPii: false,
    tracesSampleRate: envFloat("VITE_SENTRY_TRACES_SAMPLE_RATE", 0.05),
    beforeSend: scrubEvent,
    beforeBreadcrumb: scrubBreadcrumb,
    integrations: [Sentry.browserTracingIntegration()],
  })
}

export function captureRouteError(
  error: unknown,
  info: { componentStack?: string | null; route?: string },
): void {
  if (!sentryFrontendEnabled()) return
  Sentry.captureException(error, {
    contexts: {
      react: {
        componentStack: info.componentStack ?? undefined,
      },
      route: {
        path: info.route,
      },
    },
  })
}

export { Sentry }
