import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"
import { authApi } from "@/lib/api"

interface AuthState {
  isAuthenticated: boolean
  isLoading: boolean
  login: (password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

const AUTH_MODE = (import.meta.env.VITE_AUTH_MODE ?? "password").toLowerCase()
const IS_CLOUDFLARE_AUTH = AUTH_MODE === "cloudflare"

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(IS_CLOUDFLARE_AUTH)
  const [isLoading, setIsLoading] = useState(!IS_CLOUDFLARE_AUTH)

  // On mount: silently check if a valid cookie already exists
  useEffect(() => {
    if (IS_CLOUDFLARE_AUTH) return
    authApi.me()
      .then(() => setIsAuthenticated(true))
      .catch(() => setIsAuthenticated(false))
      .finally(() => setIsLoading(false))
  }, [])

  const login = useCallback(async (password: string) => {
    if (IS_CLOUDFLARE_AUTH) return
    await authApi.login(password)
    setIsAuthenticated(true)
  }, [])

  const logout = useCallback(async () => {
    if (IS_CLOUDFLARE_AUTH) {
      // Cloudflare Access logout endpoint for the current application.
      window.location.href = "/cdn-cgi/access/logout"
      return
    }
    await authApi.logout()
    setIsAuthenticated(false)
  }, [])

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider")
  return ctx
}
