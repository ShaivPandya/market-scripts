import { useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { clearCache } from "@/lib/api"

interface RefreshButtonProps {
  queryKeys?: unknown[][]
}

export function RefreshButton({ queryKeys }: RefreshButtonProps) {
  const qc = useQueryClient()
  const [isRefreshing, setIsRefreshing] = useState(false)

  async function handleRefresh() {
    setIsRefreshing(true)
    try {
      await clearCache()
    } catch {
      // ignore cache clear errors — still refetch
    }
    if (queryKeys && queryKeys.length > 0) {
      await Promise.all(queryKeys.map(key => qc.refetchQueries({ queryKey: key })))
    } else {
      await qc.refetchQueries()
    }
    setIsRefreshing(false)
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
