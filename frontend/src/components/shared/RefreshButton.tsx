import { useQueryClient } from "@tanstack/react-query"

interface RefreshButtonProps {
  queryKeys?: unknown[][]
}

export function RefreshButton({ queryKeys }: RefreshButtonProps) {
  const qc = useQueryClient()

  function handleRefresh() {
    if (queryKeys && queryKeys.length > 0) {
      queryKeys.forEach(key => qc.invalidateQueries({ queryKey: key }))
    } else {
      qc.invalidateQueries()
    }
  }

  return (
    <button
      onClick={handleRefresh}
      className="px-3 py-1.5 text-sm rounded border border-gray-300 bg-white hover:bg-gray-50 transition-colors"
    >
      Refresh Data
    </button>
  )
}
