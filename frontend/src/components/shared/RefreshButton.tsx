import { useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { clearCache } from "@/lib/api"

interface RefreshButtonProps {
  queryKeys?: unknown[][]
  beforeRefetch?: () => Promise<unknown>
  onError?: (error: unknown) => void
  onSuccess?: () => void
}

export function RefreshButton({ queryKeys, beforeRefetch, onError, onSuccess }: RefreshButtonProps) {
  const qc = useQueryClient()
  const [isRefreshing, setIsRefreshing] = useState(false)

  async function handleRefresh() {
    setIsRefreshing(true)
    try {
      await clearCache()
    } catch {
      // ignore cache clear errors — still refetch
    }
    try {
      if (beforeRefetch) await beforeRefetch()
      if (queryKeys && queryKeys.length > 0) {
        await Promise.all(queryKeys.map(key => qc.refetchQueries({ queryKey: key })))
      } else {
        await qc.refetchQueries()
      }
      onSuccess?.()
    } catch (err) {
      onError?.(err)
    } finally {
      setIsRefreshing(false)
    }
  }

  return (
    <button
      onClick={handleRefresh}
      disabled={isRefreshing}
      className="theme-button-secondary rounded-lg px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50"
    >
      {isRefreshing ? "Refreshing..." : "Refresh Data"}
    </button>
  )
}
