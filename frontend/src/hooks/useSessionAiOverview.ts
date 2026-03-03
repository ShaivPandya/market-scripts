import { useEffect, useState } from "react"

type PersistedAiOverview = {
  analysis: string | null
  isOpen: boolean
}

const EMPTY_STATE: PersistedAiOverview = {
  analysis: null,
  isOpen: false,
}

function readPersistedAiOverview(storageKey: string): PersistedAiOverview {
  if (typeof window === "undefined") return EMPTY_STATE
  try {
    const raw = window.sessionStorage.getItem(storageKey)
    if (!raw) return EMPTY_STATE
    const parsed = JSON.parse(raw) as Partial<PersistedAiOverview>
    return {
      analysis: typeof parsed.analysis === "string" && parsed.analysis.trim() ? parsed.analysis : null,
      isOpen: parsed.isOpen === true,
    }
  } catch {
    return EMPTY_STATE
  }
}

export function useSessionAiOverview(storageKey: string) {
  const [state, setState] = useState<PersistedAiOverview>(() => readPersistedAiOverview(storageKey))

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.setItem(storageKey, JSON.stringify(state))
    } catch {
      // Ignore storage write failures (private mode/quota)
    }
  }, [storageKey, state])

  const setAnalysis = (analysis: string | null) => {
    setState(prev => ({
      ...prev,
      analysis: typeof analysis === "string" && analysis.trim() ? analysis : null,
    }))
  }

  const setIsOpen = (next: boolean | ((current: boolean) => boolean)) => {
    setState(prev => ({
      ...prev,
      isOpen: typeof next === "function" ? (next as (current: boolean) => boolean)(prev.isOpen) : next,
    }))
  }

  return {
    analysis: state.analysis,
    isOpen: state.isOpen,
    setAnalysis,
    setIsOpen,
    hasAnalysis: Boolean(state.analysis && state.analysis.trim()),
  }
}
