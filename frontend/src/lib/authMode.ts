export type AuthMode = "cloudflare" | "password"

const requested = (import.meta.env.VITE_AUTH_MODE ?? "password").toLowerCase()

let currentAuthMode: AuthMode = requested === "cloudflare" ? "cloudflare" : "password"

export function getAuthMode(): AuthMode {
  return currentAuthMode
}

export function setAuthMode(mode: AuthMode) {
  currentAuthMode = mode
}

