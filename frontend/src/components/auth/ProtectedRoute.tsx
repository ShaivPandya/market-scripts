import { Navigate, Outlet } from "react-router-dom"
import { useAuth } from "@/contexts/AuthContext"

export function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-app px-4 text-sm text-muted">
        Loading...
      </div>
    )
  }

  return isAuthenticated ? <Outlet /> : <Navigate to="/login" replace />
}
