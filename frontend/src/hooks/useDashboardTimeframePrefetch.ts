import { useEffect } from "react"
import { useQueryClient } from "@tanstack/react-query"

const SHORT_TIMEFRAME_STALE_TIME_MS = 5 * 60 * 1000
const LONG_TIMEFRAME_STALE_TIME_MS = 60 * 60 * 1000

export function dashboardTimeframeStaleTime(timeframe: string): number {
  return timeframe === "This Week"
    ? SHORT_TIMEFRAME_STALE_TIME_MS
    : LONG_TIMEFRAME_STALE_TIME_MS
}

interface DashboardTimeframePrefetchOptions<Timeframe extends string> {
  queryKeyRoot: string
  timeframes: readonly Timeframe[]
  activeTimeframe: Timeframe
  isReady: boolean
  fetchTimeframe: (timeframe: Timeframe) => Promise<unknown>
  staleTimeForTimeframe?: (timeframe: Timeframe) => number
}

export function useDashboardTimeframePrefetch<Timeframe extends string>({
  queryKeyRoot,
  timeframes,
  activeTimeframe,
  isReady,
  fetchTimeframe,
  staleTimeForTimeframe = dashboardTimeframeStaleTime,
}: DashboardTimeframePrefetchOptions<Timeframe>) {
  const queryClient = useQueryClient()

  useEffect(() => {
    if (!isReady) return

    let cancelled = false
    let timeoutId: number | null = null
    let idleId: number | null = null

    const prefetchRemainingTimeframes = async () => {
      for (const nextTimeframe of timeframes) {
        if (cancelled || nextTimeframe === activeTimeframe) continue
        try {
          await queryClient.prefetchQuery({
            queryKey: [queryKeyRoot, nextTimeframe],
            queryFn: () => fetchTimeframe(nextTimeframe),
            staleTime: staleTimeForTimeframe(nextTimeframe),
          })
        } catch {
          // Background prefetch is opportunistic; the active query will surface errors when selected.
        }
      }
    }

    const startPrefetch = () => {
      void prefetchRemainingTimeframes()
    }

    if (typeof window.requestIdleCallback === "function") {
      idleId = window.requestIdleCallback(startPrefetch)
    } else {
      timeoutId = window.setTimeout(startPrefetch, 1000)
    }

    return () => {
      cancelled = true
      if (idleId !== null && typeof window.cancelIdleCallback === "function") {
        window.cancelIdleCallback(idleId)
      }
      if (timeoutId !== null) window.clearTimeout(timeoutId)
    }
  }, [
    activeTimeframe,
    fetchTimeframe,
    isReady,
    queryClient,
    queryKeyRoot,
    staleTimeForTimeframe,
    timeframes,
  ])
}
