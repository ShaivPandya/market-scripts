import { useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Save } from "lucide-react"

import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchLLMSettings,
  updateLLMSettings,
  type LLMModelTier,
  type LLMProvider,
  type LLMProviderStatus,
  type LLMReasoningEffort,
  type LLMReasoningEffortMap,
  type LLMSettings,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import { PageHeader } from "@/components/shared/PageHeader"
import { SurfaceCard } from "@/components/shared/SurfaceCard"
import { StatusBadge } from "@/components/shared/StatusBadge"

const QUERY_KEY = ["llm-settings"]
const MODEL_TIERS = [
  { key: "low", label: "Low" },
  { key: "mid", label: "Mid" },
  { key: "high", label: "High" },
] as const

const DEFAULT_REASONING_EFFORTS_BY_PROVIDER: Record<LLMProvider, LLMReasoningEffortMap> = {
  anthropic: {
    low: "high",
    mid: "high",
    high: "high",
  },
  openai: {
    low: "medium",
    mid: "medium",
    high: "medium",
  },
}

function providerDescription(provider: LLMProvider) {
  return provider === "anthropic" ? "Claude runtime requests" : "OpenAI runtime requests"
}

function statusBadge(provider: LLMProviderStatus) {
  if (provider.configured) {
    return <StatusBadge tone="success">Configured</StatusBadge>
  }

  return <StatusBadge tone="warning">Missing {provider.api_key_env}</StatusBadge>
}

export function AISettings() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useApiQuery<LLMSettings>(QUERY_KEY, fetchLLMSettings, 30_000)
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider | null>(null)
  const [draftReasoningEfforts, setDraftReasoningEfforts] =
    useState<Record<LLMProvider, LLMReasoningEffortMap> | null>(null)
  const effectiveProvider = selectedProvider ?? data?.provider ?? "anthropic"

  const selectedStatus = useMemo(
    () => data?.available_providers.find(provider => provider.provider === effectiveProvider),
    [data?.available_providers, effectiveProvider],
  )

  const effectiveModels = data?.models_by_provider?.[effectiveProvider] ?? data?.models
  const providerDefaultReasoningEfforts = DEFAULT_REASONING_EFFORTS_BY_PROVIDER[effectiveProvider]
  const savedReasoningEfforts = data?.reasoning_efforts?.[effectiveProvider] ?? providerDefaultReasoningEfforts
  const effectiveReasoningEfforts = draftReasoningEfforts?.[effectiveProvider] ?? savedReasoningEfforts

  const mutation = useMutation({
    mutationFn: updateLLMSettings,
    onSuccess: settings => {
      queryClient.setQueryData(QUERY_KEY, settings)
      setSelectedProvider(null)
      setDraftReasoningEfforts(null)
    },
  })

  const hasReasoningChanges = data
    ? MODEL_TIERS.some(tier => effectiveReasoningEfforts[tier.key] !== savedReasoningEfforts[tier.key])
    : false
  const hasChanges = data ? effectiveProvider !== data.provider || hasReasoningChanges : false
  const canSave = Boolean(hasChanges && selectedStatus?.configured && !mutation.isPending)

  const updateReasoningEffort = (tier: LLMModelTier, effort: LLMReasoningEffort) => {
    if (!data) return
    setDraftReasoningEfforts(prev => {
      const base = prev ?? data.reasoning_efforts
      return {
        ...base,
        [effectiveProvider]: {
          ...(base[effectiveProvider] ?? providerDefaultReasoningEfforts),
          [tier]: effort,
        },
      }
    })
  }

  if (isLoading) return <LoadingSpinner message="Loading AI settings..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load AI settings"} />

  return (
    <div className="max-w-4xl">
      <PageHeader
        title="AI Settings"
        subtitle="Runtime provider controls for live app AI features."
      />

      <SurfaceCard className="p-5">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="section-title">Provider</h2>
            <p className="mt-1 text-xs text-muted">
              Active provider: {data.provider === "anthropic" ? "Claude" : "OpenAI"}
            </p>
          </div>
          <button
            type="button"
            onClick={() => mutation.mutate({
              provider: effectiveProvider,
              reasoning_efforts: effectiveReasoningEfforts,
            })}
            disabled={!canSave}
            className="theme-button-base theme-button-primary px-4 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Save size={15} aria-hidden="true" />
            Save
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {data.available_providers.map(provider => {
            return (
              <button
                key={provider.provider}
                type="button"
                onClick={() => setSelectedProvider(provider.provider)}
                className={cn(
                  "theme-surface rounded-[1rem] px-4 py-4 text-left transition-colors",
                  effectiveProvider === provider.provider
                    ? "border-[hsl(var(--accent))] bg-selected"
                    : "hover:bg-hover",
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "inline-block h-3 w-3 rounded-full border",
                          effectiveProvider === provider.provider
                            ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent))]"
                            : "border-strong",
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
          <p className="mt-4 text-sm text-[hsl(var(--warning))]">
            {selectedStatus?.api_key_env ?? "Provider API key"} is not configured on the API server.
          </p>
        )}
      </SurfaceCard>

      <SurfaceCard className="mt-5 p-5">
        <div>
          <h2 className="section-title">Reasoning Effort</h2>
          <p className="mt-1 text-xs text-muted">
            {effectiveProvider === "anthropic" ? "Claude" : "OpenAI"} thinking depth by model tier.
          </p>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {MODEL_TIERS.map(tier => {
            const options = data.reasoning_options?.[effectiveProvider]?.[tier.key] ?? []
            return (
              <div key={tier.key} className="theme-surface-muted px-3 py-3">
                <div className="min-h-[4.5rem]">
                  <p className="label-text">{tier.label}</p>
                  <p className="mt-2 break-words font-mono text-sm text-app">
                    {effectiveModels?.[tier.key] ?? data.models[tier.key]}
                  </p>
                </div>
                <select
                  value={effectiveReasoningEfforts[tier.key]}
                  onChange={event => updateReasoningEffort(tier.key, event.target.value as LLMReasoningEffort)}
                  className="theme-input mt-3 w-full px-3 py-2 text-sm"
                >
                  {options.map(option => (
                    <option key={option.effort} value={option.effort}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            )
          })}
        </div>
      </SurfaceCard>
    </div>
  )
}
