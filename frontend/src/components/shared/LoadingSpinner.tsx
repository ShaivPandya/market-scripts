interface LoadingSpinnerProps {
  message?: string
}

export function LoadingSpinner({ message = "Loading..." }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center gap-3 py-8 text-gray-500">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-blue-500" />
      <span className="text-sm">{message}</span>
    </div>
  )
}

export function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      <strong>Error:</strong> {message}
    </div>
  )
}
