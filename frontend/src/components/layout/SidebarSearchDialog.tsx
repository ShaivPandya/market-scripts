import * as RadixDialog from "@radix-ui/react-dialog"
import { ArrowUpRight, Search, X } from "lucide-react"
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { cn } from "@/lib/utils"
import { NAV_SECTIONS } from "./Sidebar"

interface SidebarSearchDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onNavigate?: () => void
}

interface SidebarSearchPage {
  label: string
  path: string
  section: string
}

const SIDEBAR_PAGES: SidebarSearchPage[] = NAV_SECTIONS.flatMap(section =>
  section.pages.map(page => ({
    ...page,
    section: section.label ?? "Workflows",
  })),
)

function normalizeSearchText(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()
}

function pageMatchesQuery(page: SidebarSearchPage, query: string) {
  const terms = normalizeSearchText(query).split(/\s+/).filter(Boolean)
  if (terms.length === 0) return true

  const haystack = normalizeSearchText(`${page.label} ${page.section} ${page.path}`)
  return terms.every(term => haystack.includes(term))
}

export function SidebarSearchDialog({ open, onOpenChange, onNavigate }: SidebarSearchDialogProps) {
  const [query, setQuery] = useState("")
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const location = useLocation()

  const filteredPages = useMemo(
    () => SIDEBAR_PAGES.filter(page => pageMatchesQuery(page, query)),
    [query],
  )
  const activeSelectedIndex = Math.min(selectedIndex, Math.max(filteredPages.length - 1, 0))

  useEffect(() => {
    if (!open) return

    const focusTimer = window.setTimeout(() => inputRef.current?.focus(), 40)
    return () => window.clearTimeout(focusTimer)
  }, [open])

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen) {
      setQuery("")
      setSelectedIndex(0)
    }
    onOpenChange(nextOpen)
  }

  function handleQueryChange(value: string) {
    setQuery(value)
    setSelectedIndex(0)
  }

  function handleNavigate(page: SidebarSearchPage) {
    navigate(page.path)
    handleOpenChange(false)
    onNavigate?.()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (filteredPages.length === 0) return

    if (event.key === "ArrowDown") {
      event.preventDefault()
      setSelectedIndex(index => Math.min(index + 1, filteredPages.length - 1))
      return
    }

    if (event.key === "ArrowUp") {
      event.preventDefault()
      setSelectedIndex(index => Math.max(index - 1, 0))
      return
    }

    if (event.key === "Enter") {
      const selectedPage = filteredPages[activeSelectedIndex]
      if (selectedPage) {
        event.preventDefault()
        handleNavigate(selectedPage)
      }
    }
  }

  return (
    <RadixDialog.Root open={open} onOpenChange={handleOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-[60] bg-[hsl(var(--background-overlay))]/45 backdrop-blur-[2px]" />
        <RadixDialog.Content
          className="theme-floating fixed left-1/2 top-[max(5rem,calc(4rem+var(--safe-top)))] z-[61] w-[min(calc(100vw-1.5rem),36rem)] -translate-x-1/2 overflow-hidden rounded-[1.1rem] focus:outline-none"
        >
          <RadixDialog.Title className="sr-only">Search workflows</RadixDialog.Title>
          <div className="flex items-center gap-3 border-b border-app px-4 py-3">
            <Search size={17} className="shrink-0 text-subtle" aria-hidden="true" />
            <input
              ref={inputRef}
              type="search"
              value={query}
              onChange={event => handleQueryChange(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search workflows"
              className="h-10 min-w-0 flex-1 bg-transparent text-sm text-app outline-none placeholder:text-[hsl(var(--foreground-quaternary))] focus-visible:shadow-none"
            />
            <RadixDialog.Close asChild>
              <button
                type="button"
                className="theme-icon-button h-9 w-9 shrink-0"
                aria-label="Close page search"
              >
                <X size={15} />
              </button>
            </RadixDialog.Close>
          </div>

          <div className="max-h-[min(60vh,28rem)] overflow-y-auto p-2" role="listbox" aria-label="Workflow pages">
            {filteredPages.length > 0 ? (
              filteredPages.map((page, index) => {
                const isSelected = index === activeSelectedIndex
                const isCurrent = page.path === location.pathname

                return (
                  <button
                    key={page.path}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onMouseEnter={() => setSelectedIndex(index)}
                    onClick={() => handleNavigate(page)}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors",
                      isSelected ? "bg-selected" : "hover:bg-hover",
                    )}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-app">{page.label}</span>
                      <span className="mt-0.5 block truncate text-xs text-muted">
                        {page.section} / {page.path}
                      </span>
                    </span>
                    {isCurrent && (
                      <span className="rounded-full bg-[hsl(var(--accent-muted))] px-2 py-1 text-[10px] font-medium uppercase tracking-[0.12em] text-link">
                        Current
                      </span>
                    )}
                    <ArrowUpRight size={14} className="shrink-0 text-subtle" aria-hidden="true" />
                  </button>
                )
              })
            ) : (
              <div className="px-3 py-10 text-center text-sm text-muted">
                No pages found.
              </div>
            )}
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
