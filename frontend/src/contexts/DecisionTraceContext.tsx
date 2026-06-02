import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react"

import type { ProvenanceSelector } from "@/lib/api"
import {
  buildDecisionTrace,
  buildProvenanceOnlyTrace,
  type DecisionTraceEntityKind,
  type DecisionTraceModel,
} from "@/lib/decisionTrace"
import type { AgentMessage } from "@/hooks/agentChatShared"

interface OpenDecisionTraceOptions {
  kind: DecisionTraceEntityKind
  record: Record<string, unknown>
  sessionId?: string | null
  message?: AgentMessage
  title?: string
}

interface DecisionTraceContextValue {
  trace: DecisionTraceModel | null
  open: boolean
  openDecisionTrace: (options: OpenDecisionTraceOptions) => void
  openProvenanceTrace: (selector: ProvenanceSelector, title?: string) => void
  closeDecisionTrace: () => void
}

const DecisionTraceContext = createContext<DecisionTraceContextValue | null>(null)

export function DecisionTraceProvider({ children }: { children: ReactNode }) {
  const [trace, setTrace] = useState<DecisionTraceModel | null>(null)
  const [open, setOpen] = useState(false)

  const openDecisionTrace = useCallback((options: OpenDecisionTraceOptions) => {
    const model = buildDecisionTrace(options.kind, options.record, {
      sessionId: options.sessionId,
      message: options.message,
    })
    if (options.title) {
      model.summary.title = options.title
    }
    setTrace(model)
    setOpen(true)
  }, [])

  const openProvenanceTrace = useCallback((selector: ProvenanceSelector, title?: string) => {
    setTrace(buildProvenanceOnlyTrace(selector, title))
    setOpen(true)
  }, [])

  const closeDecisionTrace = useCallback(() => {
    setOpen(false)
    setTrace(null)
  }, [])

  const value = useMemo(
    () => ({
      trace,
      open,
      openDecisionTrace,
      openProvenanceTrace,
      closeDecisionTrace,
    }),
    [trace, open, openDecisionTrace, openProvenanceTrace, closeDecisionTrace],
  )

  return <DecisionTraceContext.Provider value={value}>{children}</DecisionTraceContext.Provider>
}

// Hooks are exported from the provider module for convenience; fast refresh treats them separately.
// eslint-disable-next-line react-refresh/only-export-components
export function useDecisionTrace(): DecisionTraceContextValue {
  const context = useContext(DecisionTraceContext)
  if (!context) {
    throw new Error("useDecisionTrace must be used within DecisionTraceProvider")
  }
  return context
}

// eslint-disable-next-line react-refresh/only-export-components
export function useOptionalDecisionTrace(): DecisionTraceContextValue | null {
  return useContext(DecisionTraceContext)
}
