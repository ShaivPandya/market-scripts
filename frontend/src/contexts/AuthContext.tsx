/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"
import { authApi } from "@/lib/api"
import { setAuthMode, type AuthMode } from "@/lib/authMode"

interface AuthState {
  mode: AuthMode
  isAuthenticated: boolean
  isLoading: boolean
  login: (password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

const REQUESTED_AUTH_MODE: AuthMode =
  (import.meta.env.VITE_AUTH_MODE ?? "password").toLowerCase() === "cloudflare"
    ? "cloudflare"
    : "password"

async function probeCloudflareAccess(): Promise<{ available: boolean; authenticated: boolean }> {
  try {
    const res = await fetch("/cdn-cgi/access/get-identity", {
      method: "GET",
      redirect: "manual",
      credentials: "include",
      cache: "no-store",
    })

    if (res.status === 404) return { available: false, authenticated: false }
    return { available: true, authenticated: res.ok }
  } catch {
    return { available: false, authenticated: false }
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<AuthMode>(REQUESTED_AUTH_MODE)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // On mount: silently check if a valid cookie already exists
  useEffect(() => {
    let cancelled = false

    async function checkPasswordSession() {
      if (!sessionStorage.getItem('auth_session')) {
        // No tab-scoped flag — force logout to clear any stale cookie, then require login
        try { await authApi.logout() } catch { /* ignore */ }
        if (!cancelled) { setIsAuthenticated(false); setIsLoading(false) }
        return
      }
      try {
        await authApi.me()
        if (!cancelled) setIsAuthenticated(true)
      } catch {
        if (!cancelled) { sessionStorage.removeItem('auth_session'); setIsAuthenticated(false) }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    async function init() {
      if (REQUESTED_AUTH_MODE !== "cloudflare") {
        setAuthMode("password")
        setMode("password")
        await checkPasswordSession()
        return
      }

      const probe = await probeCloudflareAccess()
      if (cancelled) return

      if (!probe.available) {
        // Cloudflare Access isn't configured for this hostname — fall back to password mode.
        setAuthMode("password")
        setMode("password")
        await checkPasswordSession()
        return
      }

      setAuthMode("cloudflare")
      setMode("cloudflare")
      setIsAuthenticated(probe.authenticated)
      setIsLoading(false)
    }

    init().catch(() => {
      if (!cancelled) {
        setAuthMode("password")
        setMode("password")
        setIsAuthenticated(false)
        setIsLoading(false)
      }
    })

    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (password: string) => {
    if (mode === "cloudflare") {
      // Cloudflare Access handles auth via the edge. Reload a protected route to trigger login.
      window.location.href = "/"
      return
    }
    await authApi.login(password)
    sessionStorage.setItem('auth_session', '1')
    setIsAuthenticated(true)
  }, [mode])

  const logout = useCallback(async () => {
    if (mode === "cloudflare") {
      // Cloudflare Access logout endpoint for the current application.
      window.location.href = "/cdn-cgi/access/logout"
      return
    }
    await authApi.logout()
    sessionStorage.removeItem('auth_session')
    setIsAuthenticated(false)
  }, [mode])

  return (
    <AuthContext.Provider value={{ mode, isAuthenticated, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider")
  return ctx
}
