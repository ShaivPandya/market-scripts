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
      className="px-3 py-1.5 text-sm rounded border border-gray-300 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isRefreshing ? "Refreshing..." : "Refresh Data"}
    </button>
  )
}
