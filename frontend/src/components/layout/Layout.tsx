import { useState } from "react"
import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-app text-app">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className="flex-1 overflow-auto p-3 md:p-6 bg-app min-w-0">
        <button
          className="theme-button-secondary mb-3 rounded-lg p-2 md:hidden"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open navigation"
        >
          <div className="mb-1 h-0.5 w-5 bg-[hsl(var(--muted-foreground))]" />
          <div className="mb-1 h-0.5 w-5 bg-[hsl(var(--muted-foreground))]" />
          <div className="h-0.5 w-5 bg-[hsl(var(--muted-foreground))]" />
        </button>

        <Outlet />
      </main>
    </div>
  )
}
