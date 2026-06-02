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
import { setCsrfToken } from "@/lib/csrfToken"

interface AuthState {
  mode: AuthMode
  isAuthenticated: boolean
  isLoading: boolean
  username: string | null
  login: (password: string, username?: string) => Promise<void>
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
  const [username, setUsername] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function checkPasswordSession() {
      try {
        const me = await authApi.me()
        if (!cancelled) {
          setUsername(me.username)
          setIsAuthenticated(true)
        }
      } catch {
        if (!cancelled) {
          setCsrfToken(null)
          setUsername(null)
          setIsAuthenticated(false)
        }
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
        setAuthMode("password")
        setMode("password")
        await checkPasswordSession()
        return
      }

      setAuthMode("cloudflare")
      setMode("cloudflare")
      if (probe.authenticated) {
        try {
          const me = await authApi.me()
          if (!cancelled) {
            setUsername(me.username)
            setIsAuthenticated(true)
          }
        } catch {
          if (!cancelled) setIsAuthenticated(probe.authenticated)
        }
      } else {
        setIsAuthenticated(false)
      }
      if (!cancelled) setIsLoading(false)
    }

    init().catch(() => {
      if (!cancelled) {
        setAuthMode("password")
        setMode("password")
        setCsrfToken(null)
        setUsername(null)
        setIsAuthenticated(false)
        setIsLoading(false)
      }
    })

    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(
    async (password: string, loginUsername?: string) => {
      if (mode === "cloudflare") {
        window.location.href = "/"
        return
      }
      const result = await authApi.login(password, loginUsername)
      setUsername(result.username)
      setIsAuthenticated(true)
    },
    [mode],
  )

  const logout = useCallback(async () => {
    if (mode === "cloudflare") {
      window.location.href = "/cdn-cgi/access/logout"
      return
    }
    await authApi.logout()
    setCsrfToken(null)
    setUsername(null)
    setIsAuthenticated(false)
  }, [mode])

  return (
    <AuthContext.Provider
      value={{ mode, isAuthenticated, isLoading, username, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider")
  return ctx
}
