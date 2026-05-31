import * as RadixDialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import type { ReactNode } from "react"

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  children: ReactNode
  maxWidth?: string
}

export function Dialog({ open, onOpenChange, title, description, children, maxWidth = "max-w-4xl" }: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--background-overlay))]/45 backdrop-blur-[2px]" />
        <RadixDialog.Content
          className={`theme-floating fixed left-1/2 top-1/2 z-50 w-[min(calc(100vw-1.5rem),64rem)] -translate-x-1/2 -translate-y-1/2 ${maxWidth} max-h-[min(88vh,calc(100dvh-2rem-var(--safe-top)-var(--safe-bottom)))] overflow-y-auto rounded-[1.4rem] focus:outline-none max-sm:top-auto max-sm:bottom-[max(0.75rem,calc(0.75rem+var(--safe-bottom)))] max-sm:max-h-[min(92dvh,calc(100dvh-1.5rem-var(--safe-top)-var(--safe-bottom)))] max-sm:translate-y-0`}
        >
          <div className="flex items-center justify-between gap-4 border-b border-app px-5 py-4 sm:px-6">
            <div>
              <RadixDialog.Title className="text-lg font-semibold tracking-[-0.02em] text-app">
                {title}
              </RadixDialog.Title>
              {description && (
                <RadixDialog.Description className="mt-1 text-sm leading-6 text-muted">
                  {description}
                </RadixDialog.Description>
              )}
            </div>
            <RadixDialog.Close asChild>
              <button
                type="button"
                className="theme-icon-button h-11 w-11 shrink-0"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </RadixDialog.Close>
          </div>
          <div className="px-5 py-5 sm:px-6">{children}</div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
