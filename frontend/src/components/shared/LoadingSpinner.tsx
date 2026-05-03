import { Notice } from "./Notice"

interface LoadingSpinnerProps {
  message?: string
}

export function LoadingSpinner({ message = "Loading..." }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center gap-3 py-8 text-muted">
      <div
        className="h-5 w-5 animate-spin rounded-full border-2 border-app border-t-[hsl(var(--accent))]"
        style={{
          borderTopColor: "hsl(var(--accent))",
        }}
      />
      <span className="text-sm">{message}</span>
    </div>
  )
}

export function ErrorMessage({ message }: { message: string }) {
  return (
    <Notice tone="error"><strong>Error:</strong> {message}</Notice>
  )
}
