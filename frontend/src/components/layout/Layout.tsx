import { useState } from "react"
import { Outlet } from "react-router-dom"
import { MessageCircle } from "lucide-react"
import { Sidebar } from "./Sidebar"
import { AgentChat } from "../agent/AgentChat"

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [agentOpen, setAgentOpen] = useState(false)

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

      {/* AI Agent floating button */}
      <button
        onClick={() => setAgentOpen(true)}
        className="fixed bottom-6 right-6 z-40 flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg hover:bg-blue-700 transition-colors hover:scale-105"
        aria-label="Open AI Agent"
      >
        <MessageCircle size={20} />
      </button>

      {/* AI Agent drawer */}
      <AgentChat open={agentOpen} onClose={() => setAgentOpen(false)} />
    </div>
  )
}
