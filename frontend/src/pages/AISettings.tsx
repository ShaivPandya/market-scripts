import { useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Save } from "lucide-react"

import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchLLMSettings,
  updateLLMSettings,
  type GatewayDataSensitivity,
  type GatewayDeniedRule,
  type GatewayPolicySettings,
  type LLMModelTier,
  type LLMProvider,
  type LLMProviderMode,
  type LLMProviderStatus,
  type LLMProviderTierMap,
  type LLMReasoningEffort,
  type LLMReasoningEffortMap,
  type LLMSettings,
  type ToolLifecycleState,
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
const REASONING_LEVELS = [
  { key: "low", label: "Low" },
  { key: "mid", label: "Medium" },
  { key: "high", label: "High" },
] as const

const DEFAULT_REASONING_EFFORTS_BY_PROVIDER: Record<LLMProvider, LLMReasoningEffortMap> = {
  anthropic: {
    low: "medium",
    mid: "high",
    high: "max",
  },
  openai: {
    low: "low",
    mid: "medium",
    high: "xhigh",
  },
  gemini: {
    low: "low",
    mid: "medium",
    high: "high",
  },
}

const PROVIDER_FALLBACK_LABELS: Record<LLMProvider, string> = {
  anthropic: "Claude",
  openai: "OpenAI",
  gemini: "Gemini",
}
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "claude-haiku-4-5": "Claude Haiku 4.5",
  "claude-sonnet-4-6": "Claude Sonnet 4.6",
  "claude-opus-4-7": "Claude Opus 4.7",
  "gpt-5.4-mini": "GPT 5.4 Mini",
  "gpt-5.4": "GPT 5.4",
  "gpt-5.5": "GPT 5.5",
  "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
  "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
  "gemini-3.1-pro-preview-customtools": "Gemini 3.1 Pro Preview (Custom Tools)",
}
const MODEL_DISPLAY_TOKEN_NAMES: Record<string, string> = {
  api: "API",
  chatgpt: "ChatGPT",
  claude: "Claude",
  customtools: "Custom Tools",
  flash: "Flash",
  gemini: "Gemini",
  gpt: "GPT",
  haiku: "Haiku",
  high: "High",
  lite: "Lite",
  low: "Low",
  mid: "Mid",
  mini: "Mini",
  o: "O",
  opus: "Opus",
  preview: "Preview",
  pro: "Pro",
  sonnet: "Sonnet",
}
const LIFECYCLE_STATES: ToolLifecycleState[] = ["draft", "enabled", "deprecated", "disabled"]
const DATA_SENSITIVITIES: GatewayDataSensitivity[] = [
  "public_market",
  "portfolio_private",
  "research_private",
  "account_private",
  "operational_private",
]

function providerDisplayName(provider: LLMProvider | "*") {
  return provider === "*" ? "Any provider" : PROVIDER_FALLBACK_LABELS[provider]
}

function providerByTierForProvider(provider: LLMProvider): LLMProviderTierMap {
  return {
    low: provider,
    mid: provider,
    high: provider,
  }
}

function mapsEqual(a: unknown, b: unknown) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null)
}

function titleCaseModelToken(token: string) {
  if (!token) return ""
  const known = MODEL_DISPLAY_TOKEN_NAMES[token.toLowerCase()]
  if (known) return known
  if (/^\d/.test(token)) return token
  return token.charAt(0).toUpperCase() + token.slice(1)
}

function modelDisplayName(model: string) {
  const normalized = model.trim()
  const exact = MODEL_DISPLAY_NAMES[normalized]
  if (exact) return exact

  const rawTokens = normalized.split(/[-_\s]+/).filter(Boolean)
  const tokens: string[] = []
  for (let index = 0; index < rawTokens.length; index += 1) {
    const token = rawTokens[index]
    const nextToken = rawTokens[index + 1]
    if (
      nextToken &&
      /^\d{1,2}$/.test(token) &&
      /^\d{1,2}$/.test(nextToken)
    ) {
      tokens.push(`${token}.${nextToken}`)
      index += 1
      continue
    }
    tokens.push(titleCaseModelToken(token))
  }
  return tokens.length ? tokens.join(" ") : normalized
}

function gatewayRuleModelDisplayName(model: string) {
  return model === "*" ? "Any model" : modelDisplayName(model)
}

function statusBadge(provider: LLMProviderStatus, className?: string) {
  if (provider.configured) {
    return <StatusBadge tone="success" className={className}>Configured</StatusBadge>
  }

  return <StatusBadge tone="warning" className={className}>Missing {provider.api_key_env}</StatusBadge>
}

function policiesEqual(a: GatewayPolicySettings | undefined, b: GatewayPolicySettings | undefined) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null)
}

function lifecycleLabel(state: ToolLifecycleState) {
  return state.replace("_", " ")
}

function nativeEffortForReasoningLevel(provider: LLMProvider, level: LLMModelTier): LLMReasoningEffort {
  return DEFAULT_REASONING_EFFORTS_BY_PROVIDER[provider][level]
}

function reasoningLevelForNativeEffort(
  provider: LLMProvider,
  tier: LLMModelTier,
  effort: LLMReasoningEffort,
): LLMModelTier {
  const mapped = DEFAULT_REASONING_EFFORTS_BY_PROVIDER[provider]
  const match = REASONING_LEVELS.find(level => mapped[level.key] === effort)
  return match?.key ?? tier
}

export function AISettings() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useApiQuery<LLMSettings>(QUERY_KEY, fetchLLMSettings, 30_000)
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider | null>(null)
  const [selectedProviderMode, setSelectedProviderMode] = useState<LLMProviderMode | null>(null)
  const [draftProviderByTier, setDraftProviderByTier] = useState<LLMProviderTierMap | null>(null)
  const [draftReasoningEfforts, setDraftReasoningEfforts] =
    useState<Record<LLMProvider, LLMReasoningEffortMap> | null>(null)
  const [draftGatewayPolicy, setDraftGatewayPolicy] = useState<GatewayPolicySettings | null>(null)
  const [gatewayNote, setGatewayNote] = useState("")
  const [newDeniedRule, setNewDeniedRule] = useState<GatewayDeniedRule>({
    provider: "*",
    model: "*",
    data_sensitivity: "portfolio_private",
  })
  const fallbackProvider = data?.provider ?? "anthropic"
  const effectiveProvider = selectedProvider ?? fallbackProvider
  const savedProviderMode = data?.provider_mode ?? "single"
  const effectiveProviderMode = selectedProviderMode ?? savedProviderMode
  const savedProviderByTier = data?.provider_by_tier ?? providerByTierForProvider(fallbackProvider)
  const effectiveProviderByTier =
    effectiveProviderMode === "custom"
      ? draftProviderByTier ?? savedProviderByTier
      : providerByTierForProvider(effectiveProvider)

  const selectedStatus = useMemo(
    () => data?.available_providers.find(provider => provider.provider === effectiveProvider),
    [data?.available_providers, effectiveProvider],
  )
  const effectiveProviderLabel =
    effectiveProviderMode === "custom"
      ? "Custom"
      : selectedStatus?.label ?? PROVIDER_FALLBACK_LABELS[effectiveProvider]

  const effectiveModels = MODEL_TIERS.reduce((models, tier) => {
    const tierProvider = effectiveProviderByTier[tier.key]
    models[tier.key] = data?.models_by_provider?.[tierProvider]?.[tier.key] ?? data?.models?.[tier.key] ?? ""
    return models
  }, {} as Record<LLMModelTier, string>)
  const effectiveReasoningEffortsByProvider =
    draftReasoningEfforts ?? data?.reasoning_efforts ?? DEFAULT_REASONING_EFFORTS_BY_PROVIDER
  const effectiveGatewayPolicy = draftGatewayPolicy ?? data?.gateway_policy
  const customSetupProviderByTier = draftProviderByTier ?? savedProviderByTier
  const customSetupStatuses = MODEL_TIERS.map(tier => (
    data?.available_providers.find(provider => provider.provider === customSetupProviderByTier[tier.key])
  ))
  const customSetupConfigured = customSetupStatuses.every(provider => provider?.configured)
  const customProviderStatuses = MODEL_TIERS.map(tier => (
    data?.available_providers.find(provider => provider.provider === effectiveProviderByTier[tier.key])
  ))
  const missingCustomProviderLabels = customProviderStatuses
    .filter(provider => provider && !provider.configured)
    .map(provider => provider?.api_key_env)
    .filter(Boolean)
  const providerSelectionConfigured =
    effectiveProviderMode === "custom"
      ? customProviderStatuses.every(provider => provider?.configured)
      : Boolean(selectedStatus?.configured)

  const mutation = useMutation({
    mutationFn: updateLLMSettings,
    onSuccess: settings => {
      queryClient.setQueryData(QUERY_KEY, settings)
      setSelectedProvider(null)
      setSelectedProviderMode(null)
      setDraftProviderByTier(null)
      setDraftReasoningEfforts(null)
      setDraftGatewayPolicy(null)
      setGatewayNote("")
    },
  })

  const hasReasoningChanges = data
    ? !mapsEqual(effectiveReasoningEffortsByProvider, data.reasoning_efforts)
    : false
  const hasProviderChanges = data
    ? effectiveProviderMode !== savedProviderMode ||
      (effectiveProviderMode === "custom"
        ? !mapsEqual(effectiveProviderByTier, savedProviderByTier)
        : effectiveProvider !== data.provider)
    : false
  const hasGatewayChanges = data ? !policiesEqual(effectiveGatewayPolicy, data.gateway_policy) : false
  const hasChanges = data ? hasProviderChanges || hasReasoningChanges || hasGatewayChanges : false
  const canSave = Boolean(
    hasChanges &&
    providerSelectionConfigured &&
    !mutation.isPending &&
    (!hasGatewayChanges || gatewayNote.trim()),
  )

  const updateProviderByTier = (tier: LLMModelTier, provider: LLMProvider) => {
    setDraftProviderByTier(prev => ({
      ...(prev ?? effectiveProviderByTier),
      [tier]: provider,
    }))
  }

  const updateReasoningEffort = (tier: LLMModelTier, provider: LLMProvider, level: LLMModelTier) => {
    if (!data) return
    setDraftReasoningEfforts(prev => {
      const base = prev ?? data.reasoning_efforts
      return {
        ...base,
        [provider]: {
          ...(base[provider] ?? DEFAULT_REASONING_EFFORTS_BY_PROVIDER[provider]),
          [tier]: nativeEffortForReasoningLevel(provider, level),
        },
      }
    })
  }

  const updateGatewayPolicy = (updater: (policy: GatewayPolicySettings) => GatewayPolicySettings) => {
    if (!data?.gateway_policy) return
    setDraftGatewayPolicy(prev => updater(prev ?? data.gateway_policy))
  }

  const updateProviderLifecycle = (provider: LLMProvider, lifecycle: ToolLifecycleState) => {
    updateGatewayPolicy(policy => ({
      ...policy,
      provider_lifecycle: { ...policy.provider_lifecycle, [provider]: lifecycle },
    }))
  }

  const updateModelLifecycle = (model: string, lifecycle: ToolLifecycleState) => {
    updateGatewayPolicy(policy => ({
      ...policy,
      model_lifecycle: { ...policy.model_lifecycle, [model]: lifecycle },
    }))
  }

  const addDeniedRule = () => {
    updateGatewayPolicy(policy => ({
      ...policy,
      denied_rules: [...policy.denied_rules, { ...newDeniedRule }],
    }))
  }

  const removeDeniedRule = (index: number) => {
    updateGatewayPolicy(policy => ({
      ...policy,
      denied_rules: policy.denied_rules.filter((_rule, ruleIndex) => ruleIndex !== index),
    }))
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
              Active setup: {data.provider_mode === "custom"
                ? "Custom"
                : (data.available_providers.find(provider => provider.provider === data.provider)?.label ?? PROVIDER_FALLBACK_LABELS[data.provider])}
            </p>
          </div>
          <button
            type="button"
            onClick={() => mutation.mutate({
              provider: effectiveProvider,
              provider_mode: effectiveProviderMode,
              provider_by_tier: effectiveProviderByTier,
              reasoning_efforts_by_provider: effectiveReasoningEffortsByProvider,
              ...(hasGatewayChanges && effectiveGatewayPolicy
                ? { gateway_policy: effectiveGatewayPolicy, gateway_note: gatewayNote.trim() }
                : {}),
            })}
            disabled={!canSave}
            className="theme-button-base theme-button-primary px-4 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Save size={15} aria-hidden="true" />
            Save
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          {data.available_providers.map(provider => {
            return (
              <button
                key={provider.provider}
                type="button"
                onClick={() => {
                  setSelectedProvider(provider.provider)
                  setSelectedProviderMode("single")
                }}
                className={cn(
                  "theme-surface rounded-[1rem] px-4 py-4 text-left transition-colors",
                  effectiveProviderMode === "single" && effectiveProvider === provider.provider
                    ? "border-[hsl(var(--accent))] bg-selected"
                    : "hover:bg-hover",
                )}
              >
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className={cn(
                        "h-3 w-3 flex-none rounded-full border",
                        effectiveProviderMode === "single" && effectiveProvider === provider.provider
                          ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent))]"
                          : "border-strong",
                      )}
                    />
                    <span className="truncate text-sm font-semibold text-app">{provider.label}</span>
                  </div>
                  {statusBadge(provider, "shrink-0")}
                </div>
              </button>
            )
          })}
          <button
            type="button"
            onClick={() => {
              setSelectedProviderMode("custom")
              setSelectedProvider(null)
              setDraftProviderByTier(prev => prev ?? savedProviderByTier)
            }}
            className={cn(
              "theme-surface rounded-[1rem] px-4 py-4 text-left transition-colors",
              effectiveProviderMode === "custom"
                ? "border-[hsl(var(--accent))] bg-selected"
                : "hover:bg-hover",
            )}
          >
            <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
              <div className="flex min-w-0 items-center gap-2">
                <span
                  className={cn(
                    "h-3 w-3 flex-none rounded-full border",
                    effectiveProviderMode === "custom"
                      ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent))]"
                      : "border-strong",
                  )}
                />
                <span className="truncate text-sm font-semibold text-app">Custom</span>
              </div>
              {customSetupConfigured
                ? <StatusBadge tone="success" className="shrink-0">Configured</StatusBadge>
                : <StatusBadge tone="warning" className="shrink-0">Missing key</StatusBadge>}
            </div>
          </button>
        </div>

        {effectiveProviderMode === "custom" && (
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            {MODEL_TIERS.map(tier => {
              const tierProvider = effectiveProviderByTier[tier.key]
              const tierStatus = data.available_providers.find(provider => provider.provider === tierProvider)
              const tierModel = data.models_by_provider?.[tierProvider]?.[tier.key] ?? effectiveModels[tier.key]
              return (
                <div key={tier.key} className="theme-surface-muted px-3 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="label-text">{tier.label}</p>
                      <p className="mt-2 break-words text-sm text-app" title={tierModel}>
                        {modelDisplayName(tierModel)}
                      </p>
                    </div>
                    {tierStatus ? statusBadge(tierStatus) : null}
                  </div>
                  <select
                    value={tierProvider}
                    onChange={event => updateProviderByTier(tier.key, event.target.value as LLMProvider)}
                    className="theme-input mt-3 w-full px-3 py-2 text-sm"
                  >
                    {data.available_providers.map(provider => (
                      <option key={provider.provider} value={provider.provider}>{provider.label}</option>
                    ))}
                  </select>
                </div>
              )
            })}
          </div>
        )}

        {mutation.isError && (
          <div className="mt-4">
            <ErrorMessage message={String(mutation.error)} />
          </div>
        )}
        {hasChanges && !providerSelectionConfigured && (
          <p className="mt-4 text-sm text-[hsl(var(--warning))]">
            {effectiveProviderMode === "custom"
              ? `${Array.from(new Set(missingCustomProviderLabels)).join(", ") || "A custom provider API key"} is not configured on the API server.`
              : `${selectedStatus?.api_key_env ?? "Provider API key"} is not configured on the API server.`}
          </p>
        )}
      </SurfaceCard>

      <SurfaceCard className="mt-5 p-5">
        <div>
          <h2 className="section-title">Reasoning Effort</h2>
          <p className="mt-1 text-xs text-muted">
            {effectiveProviderLabel} thinking depth by model tier.
          </p>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {MODEL_TIERS.map(tier => {
            const tierProvider = effectiveProviderByTier[tier.key]
            const tierReasoningEfforts =
              effectiveReasoningEffortsByProvider[tierProvider] ?? DEFAULT_REASONING_EFFORTS_BY_PROVIDER[tierProvider]
            const selectedReasoningLevel = reasoningLevelForNativeEffort(
              tierProvider,
              tier.key,
              tierReasoningEfforts[tier.key],
            )
            return (
              <div key={tier.key} className="theme-surface-muted px-3 py-3">
                <div className="min-h-[4.5rem]">
                  <p className="label-text">{tier.label}</p>
                  {effectiveProviderMode === "custom" && (
                    <p className="mt-2 text-xs font-semibold text-muted">
                      {PROVIDER_FALLBACK_LABELS[tierProvider]}
                    </p>
                  )}
                  <p className="mt-2 break-words text-sm text-app">
                    {modelDisplayName(effectiveModels?.[tier.key] ?? data.models[tier.key])}
                  </p>
                </div>
                <select
                  value={selectedReasoningLevel}
                  onChange={event => updateReasoningEffort(tier.key, tierProvider, event.target.value as LLMModelTier)}
                  className="theme-input mt-3 w-full px-3 py-2 text-sm"
                >
                  {REASONING_LEVELS.map(option => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            )
          })}
        </div>
      </SurfaceCard>

      {effectiveGatewayPolicy && (
        <SurfaceCard className="mt-5 p-5">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="section-title">Gateway</h2>
              <p className="mt-1 text-xs text-muted">
                Private egress: {effectiveGatewayPolicy.private_egress_mode.replace(/_/g, " ")}
              </p>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {data.available_providers.map(provider => (
              <div key={provider.provider} className="theme-surface-muted px-3 py-3">
                <p className="label-text">{provider.label}</p>
                <select
                  value={effectiveGatewayPolicy.provider_lifecycle[provider.provider] ?? "enabled"}
                  onChange={event => updateProviderLifecycle(provider.provider, event.target.value as ToolLifecycleState)}
                  className="theme-input mt-3 w-full px-3 py-2 text-sm"
                >
                  {LIFECYCLE_STATES.map(state => (
                    <option key={state} value={state}>{lifecycleLabel(state)}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <h3 className="label-text">Current Models</h3>
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              {MODEL_TIERS.map(tier => {
                const model = effectiveModels?.[tier.key] ?? data.models[tier.key]
                const tierProvider = effectiveProviderByTier[tier.key]
                return (
                  <div key={tier.key} className="theme-surface-muted px-3 py-3">
                    <p className="label-text">{tier.label}</p>
                    {effectiveProviderMode === "custom" && (
                      <p className="mt-2 text-xs font-semibold text-muted">
                        {PROVIDER_FALLBACK_LABELS[tierProvider]}
                      </p>
                    )}
                    <p className="mt-2 break-words text-sm text-app" title={model}>
                      {modelDisplayName(model)}
                    </p>
                    <select
                      value={effectiveGatewayPolicy.model_lifecycle[model] ?? "enabled"}
                      onChange={event => updateModelLifecycle(model, event.target.value as ToolLifecycleState)}
                      className="theme-input mt-3 w-full px-3 py-2 text-sm"
                    >
                      {LIFECYCLE_STATES.map(state => (
                        <option key={state} value={state}>{lifecycleLabel(state)}</option>
                      ))}
                    </select>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="mt-5">
            <h3 className="label-text">Denied Egress Rules</h3>
            <div className="mt-3 grid gap-2 md:grid-cols-[1fr_1fr_1fr_auto]">
              <select
                value={newDeniedRule.provider}
                onChange={event => setNewDeniedRule(prev => ({ ...prev, provider: event.target.value as LLMProvider | "*" }))}
                className="theme-input px-3 py-2 text-sm"
              >
                <option value="*">Any provider</option>
                {data.available_providers.map(provider => (
                  <option key={provider.provider} value={provider.provider}>{provider.label}</option>
                ))}
              </select>
              <input
                value={newDeniedRule.model}
                onChange={event => setNewDeniedRule(prev => ({ ...prev, model: event.target.value || "*" }))}
                className="theme-input px-3 py-2 text-sm"
                placeholder="Model or *"
              />
              <select
                value={newDeniedRule.data_sensitivity}
                onChange={event => setNewDeniedRule(prev => ({ ...prev, data_sensitivity: event.target.value as GatewayDataSensitivity }))}
                className="theme-input px-3 py-2 text-sm"
              >
                {DATA_SENSITIVITIES.map(sensitivity => (
                  <option key={sensitivity} value={sensitivity}>{sensitivity.replace(/_/g, " ")}</option>
                ))}
              </select>
              <button type="button" onClick={addDeniedRule} className="theme-button-base theme-button-secondary px-3">
                Add
              </button>
            </div>
            <div className="mt-3 space-y-2">
              {effectiveGatewayPolicy.denied_rules.map((rule, index) => (
                <div key={`${rule.provider}-${rule.model}-${rule.data_sensitivity}-${index}`} className="theme-surface-muted flex flex-wrap items-center gap-2 px-3 py-2 text-sm">
                  <span>{providerDisplayName(rule.provider)}</span>
                  <span title={rule.model}>{gatewayRuleModelDisplayName(rule.model)}</span>
                  <span>{rule.data_sensitivity.replace(/_/g, " ")}</span>
                  <button type="button" onClick={() => removeDeniedRule(index)} className="ml-auto text-xs text-muted hover:text-app">
                    Remove
                  </button>
                </div>
              ))}
              {!effectiveGatewayPolicy.denied_rules.length && (
                <p className="text-xs text-muted">No explicit deny rules.</p>
              )}
            </div>
          </div>

          {hasGatewayChanges && (
            <label className="mt-5 block">
              <span className="label-text">Change Note</span>
              <textarea
                value={gatewayNote}
                onChange={event => setGatewayNote(event.target.value)}
                className="theme-input mt-2 min-h-[5rem] w-full px-3 py-2 text-sm"
                placeholder="Required before saving gateway policy changes."
              />
            </label>
          )}
        </SurfaceCard>
      )}
    </div>
  )
}
