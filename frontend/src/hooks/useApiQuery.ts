import { useQuery, type QueryKey } from "@tanstack/react-query"

/**
 * Thin wrapper around useQuery with a default staleTime matching
 * the FastAPI TTL caches (5 min for market data).
 */
export function useApiQuery<T>(
  key: QueryKey,
  fn: () => Promise<T>,
  staleTime = 5 * 60 * 1000,
) {
  return useQuery<T>({
    queryKey: key as QueryKey,
    queryFn: fn,
    staleTime,
    retry: 1,
  })
}
