import { useEffect, useState, type Dispatch, type SetStateAction } from "react"
import { Outlet, useLocation } from "react-router-dom"
import { Menu, MessageCircle, PanelRightOpen } from "lucide-react"
import { Sidebar, getRouteLabel } from "./Sidebar"
import { SidebarSearchDialog } from "./SidebarSearchDialog"
import { AgentChat } from "../agent/AgentChat"
import { ScreenContextProvider, useScreenContext, useAutoScreenContext } from "@/contexts/ScreenContext"

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [agentOpen, setAgentOpen] = useState(false)
  const [pageSearchOpen, setPageSearchOpen] = useState(false)

  return (
    <ScreenContextProvider>
      <LayoutInner
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        agentOpen={agentOpen}
        setAgentOpen={setAgentOpen}
        pageSearchOpen={pageSearchOpen}
        setPageSearchOpen={setPageSearchOpen}
      />
    </ScreenContextProvider>
  )
}

interface LayoutInnerProps {
  sidebarOpen: boolean
  setSidebarOpen: (v: boolean) => void
  agentOpen: boolean
  setAgentOpen: Dispatch<SetStateAction<boolean>>
  pageSearchOpen: boolean
  setPageSearchOpen: (v: boolean) => void
}

function LayoutInner({
  sidebarOpen,
  setSidebarOpen,
  agentOpen,
  setAgentOpen,
  pageSearchOpen,
  setPageSearchOpen,
}: LayoutInnerProps) {
  const { screenContext } = useScreenContext()
  const autoContext = useAutoScreenContext()
  const effectiveContext = screenContext ?? autoContext
  const location = useLocation()
  const routeLabel = getRouteLabel(location.pathname)

  useEffect(() => {
    function handleGlobalKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented || event.isComposing || event.repeat) return
      if (!event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return

      const key = event.key.toLowerCase()
      if (key === "j") {
        event.preventDefault()
        setPageSearchOpen(true)
        setSidebarOpen(false)
        return
      }

      if (key === "k") {
        event.preventDefault()
        setAgentOpen(open => !open)
        setPageSearchOpen(false)
      }
    }

    window.addEventListener("keydown", handleGlobalKeyDown, true)
    return () => window.removeEventListener("keydown", handleGlobalKeyDown, true)
  }, [setAgentOpen, setPageSearchOpen, setSidebarOpen])

  return (
    <div className="flex min-h-screen bg-app text-app">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className="theme-page flex-1 overflow-auto min-w-0">
        <div className="theme-page-content">
          <div className="theme-floating mb-4 flex items-center justify-between gap-3 px-3 py-3 md:hidden">
            <button
              type="button"
              className="theme-icon-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={18} />
            </button>
            <div className="min-w-0 text-center">
              <p className="text-xs uppercase tracking-[0.14em] text-subtle">Market Dashboard</p>
              <p className="truncate text-sm font-semibold text-app">{routeLabel}</p>
            </div>
            <button
              type="button"
              className="theme-icon-button"
              onClick={() => setAgentOpen(true)}
              aria-label="Open Stan"
            >
              <PanelRightOpen size={18} />
            </button>
          </div>

          <Outlet />
        </div>
      </main>

      <button
        type="button"
        onClick={() => setAgentOpen(true)}
        className="theme-floating fixed bottom-[max(1.25rem,calc(1.25rem+var(--safe-bottom)))] right-[max(1rem,calc(1rem+var(--safe-right)))] z-40 flex h-14 w-14 items-center justify-center text-link transition-colors hover:bg-selected"
        aria-label="Open Stan"
      >
        <MessageCircle size={20} />
      </button>

      <AgentChat open={agentOpen} onClose={() => setAgentOpen(false)} screenContext={effectiveContext} />
      <SidebarSearchDialog
        open={pageSearchOpen}
        onOpenChange={setPageSearchOpen}
        onNavigate={() => setSidebarOpen(false)}
      />
    </div>
  )
}
