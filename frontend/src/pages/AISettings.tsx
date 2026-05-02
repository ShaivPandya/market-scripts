import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, CheckCircle, Save } from "lucide-react"

import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchLLMSettings,
  updateLLMSettings,
  type LLMProvider,
  type LLMProviderStatus,
  type LLMSettings,
} from "@/lib/api"
import { cn } from "@/lib/utils"

const QUERY_KEY = ["llm-settings"]
const MODEL_TIERS = [
  { key: "low", label: "Low" },
  { key: "mid", label: "Mid" },
  { key: "high", label: "High" },
] as const

function providerDescription(provider: LLMProvider) {
  return provider === "anthropic" ? "Claude runtime requests" : "OpenAI runtime requests"
}

function statusBadge(provider: LLMProviderStatus) {
  if (provider.configured) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
        <CheckCircle size={13} aria-hidden="true" />
        Configured
      </span>
    )
  }

  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700 dark:bg-amber-950/40 dark:text-amber-300">
      <AlertTriangle size={13} aria-hidden="true" />
      Missing {provider.api_key_env}
    </span>
  )
}

export function AISettings() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useApiQuery<LLMSettings>(QUERY_KEY, fetchLLMSettings, 30_000)
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider>("anthropic")

  useEffect(() => {
    if (data?.provider) setSelectedProvider(data.provider)
  }, [data?.provider])

  const selectedStatus = useMemo(
    () => data?.available_providers.find(provider => provider.provider === selectedProvider),
    [data?.available_providers, selectedProvider],
  )

  const mutation = useMutation({
    mutationFn: updateLLMSettings,
    onSuccess: settings => {
      queryClient.setQueryData(QUERY_KEY, settings)
      setSelectedProvider(settings.provider)
    },
  })

  const hasChanges = data ? selectedProvider !== data.provider : false
  const canSave = Boolean(hasChanges && selectedStatus?.configured && !mutation.isPending)

  if (isLoading) return <LoadingSpinner message="Loading AI settings..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load AI settings"} />

  return (
    <div className="max-w-4xl">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-app">AI Settings</h1>
        <p className="mt-1 text-sm text-muted">
          Runtime provider for live app AI features.
        </p>
      </div>

      <section className="theme-surface rounded-xl p-5">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-app">Provider</h2>
            <p className="mt-1 text-xs text-muted">
              Active provider: {data.provider === "anthropic" ? "Claude" : "OpenAI"}
            </p>
          </div>
          <button
            type="button"
            onClick={() => mutation.mutate(selectedProvider)}
            disabled={!canSave}
            className="theme-button-primary inline-flex h-9 items-center gap-2 rounded-lg px-3 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Save size={15} aria-hidden="true" />
            Save
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {data.available_providers.map(provider => {
            const checked = selectedProvider === provider.provider
            return (
              <button
                key={provider.provider}
                type="button"
                onClick={() => setSelectedProvider(provider.provider)}
                className={cn(
                  "rounded-lg border px-4 py-3 text-left transition-colors",
                  checked
                    ? "border-[hsl(var(--accent))] bg-[hsl(var(--muted-2))]"
                    : "border-app hover:bg-[hsl(var(--muted-2))]",
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "inline-block h-3 w-3 rounded-full border",
                          checked
                            ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent))]"
                            : "border-[hsl(var(--muted-3))]",
                        )}
                      />
                      <span className="text-sm font-semibold text-app">{provider.label}</span>
                    </div>
                    <p className="mt-2 text-xs text-muted">{providerDescription(provider.provider)}</p>
                  </div>
                  {statusBadge(provider)}
                </div>
              </button>
            )
          })}
        </div>

        {mutation.isError && (
          <div className="mt-4">
            <ErrorMessage message={String(mutation.error)} />
          </div>
        )}
        {hasChanges && !selectedStatus?.configured && (
          <p className="mt-4 text-sm text-amber-700 dark:text-amber-300">
            {selectedStatus?.api_key_env ?? "Provider API key"} is not configured on the API server.
          </p>
        )}
      </section>

      <section className="theme-surface mt-5 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-app">Resolved Model Tiers</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {MODEL_TIERS.map(tier => (
            <div key={tier.key} className="rounded-lg border border-app px-3 py-3">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted">{tier.label}</p>
              <p className="mt-2 break-words font-mono text-sm text-app">{data.models[tier.key]}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
