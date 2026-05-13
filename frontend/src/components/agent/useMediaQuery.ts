import { useSyncExternalStore } from "react"

export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    onStoreChange => {
      if (typeof window === "undefined") return () => undefined
      const mediaQuery = window.matchMedia(query)
      mediaQuery.addEventListener("change", onStoreChange)
      return () => mediaQuery.removeEventListener("change", onStoreChange)
    },
    () => {
      if (typeof window === "undefined") return false
      return window.matchMedia(query).matches
    },
    () => false,
  )
}
