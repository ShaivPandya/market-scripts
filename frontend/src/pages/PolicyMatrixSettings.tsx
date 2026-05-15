import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, Plus, RotateCcw, Save, Trash2 } from "lucide-react"

import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { PageHeader } from "@/components/shared/PageHeader"
import { SelectInput, TextInput, Toggle } from "@/components/shared/FormControls"
import { SurfaceCard } from "@/components/shared/SurfaceCard"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchFinancialPolicyMatrixSettings,
  updateFinancialPolicyMatrix,
  validateFinancialPolicyMatrix,
  type FinancialPolicyApprovalMode,
  type FinancialPolicyMatrix,
  type FinancialPolicyMatrixSettings,
  type FinancialPolicyOutcome,
  type FinancialPolicyRule,
  type FinancialPolicyRuleMatch,
} from "@/lib/api"
import { cn } from "@/lib/utils"

const QUERY_KEY = ["financial-policy-matrix"]
const MATCH_FIELDS: { key: keyof FinancialPolicyRuleMatch; label: string; optionsKey?: keyof FinancialPolicyMatrixSettings["metadata"] }[] = [
  { key: "action_ids", label: "Actions", optionsKey: "action_ids" },
  { key: "action_kinds", label: "Action Kinds", optionsKey: "action_kinds" },
  { key: "request_modes", label: "Request Modes", optionsKey: "request_modes" },
  { key: "risk_levels", label: "Risk Levels", optionsKey: "risk_levels" },
  { key: "data_freshness", label: "Data Freshness", optionsKey: "data_freshness" },
]
const TEXT_MATCH_FIELDS: { key: keyof FinancialPolicyRuleMatch; label: string; placeholder: string }[] = [
  { key: "actor_roles", label: "Actor Roles", placeholder: "admin, owner" },
  { key: "actor_ids", label: "Actor IDs", placeholder: "admin, user@example.com" },
  { key: "account_ids", label: "Account IDs", placeholder: "default-account" },
  { key: "portfolio_ids", label: "Portfolio IDs", placeholder: "default-portfolio" },
]

function clonePolicy(policy: FinancialPolicyMatrix): FinancialPolicyMatrix {
  return JSON.parse(JSON.stringify(policy)) as FinancialPolicyMatrix
}

function emptyMatch(): FinancialPolicyRuleMatch {
  return {
    action_ids: [],
    action_kinds: [],
    request_modes: [],
    actor_roles: [],
    actor_ids: [],
    account_ids: [],
    portfolio_ids: [],
    risk_levels: [],
    data_freshness: [],
  }
}

function labelFor(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase())
}

function textToList(value: string) {
  return value.split(",").map(item => item.trim()).filter(Boolean)
}

function listToText(value: string[]) {
  return value.join(", ")
}

function nextRuleId(rules: FinancialPolicyRule[]) {
  const base = "custom.rule"
  let index = rules.length + 1
  let id = `${base}_${index}`
  const existing = new Set(rules.map(rule => rule.id.toLowerCase()))
  while (existing.has(id.toLowerCase())) {
    index += 1
    id = `${base}_${index}`
  }
  return id
}

function newRule(rules: FinancialPolicyRule[]): FinancialPolicyRule {
  return {
    id: nextRuleId(rules),
    enabled: true,
    priority: 100,
    match: emptyMatch(),
    limits: {},
    outcome: "use_checks",
    approval_mode: null,
    reason: "Apply configured financial approval policy.",
    remediation: "Review the rule, proposed action, and current portfolio state.",
  }
}

function MultiSelectPills({
  label,
  options,
  values,
  onChange,
}: {
  label: string
  options: string[]
  values: string[]
  onChange: (values: string[]) => void
}) {
  const selected = new Set(values)
  return (
    <div>
      <p className="theme-field-label">{label}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {options.map(option => {
          const active = selected.has(option)
          return (
            <button
              key={option}
              type="button"
              onClick={() => {
                const next = new Set(selected)
                if (active) next.delete(option)
                else next.add(option)
                onChange(Array.from(next))
              }}
              className={cn(
                "rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                active
                  ? "border-[hsl(var(--accent))] bg-selected text-app"
                  : "border-app bg-card-muted text-muted hover:bg-hover",
              )}
            >
              {labelFor(option)}
            </button>
          )
        })}
        {!options.length && <span className="text-xs text-subtle">No options</span>}
      </div>
    </div>
  )
}

function ruleSummary(rule: FinancialPolicyRule) {
  const matchCount = Object.values(rule.match).reduce((count, value) => count + (Array.isArray(value) ? value.length : 0), 0)
  const limitCount = Object.keys(rule.limits).length
  return `${matchCount || "all"} match${matchCount === 1 ? "" : "es"} · ${limitCount} limit${limitCount === 1 ? "" : "s"}`
}

export function PolicyMatrixSettings() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useApiQuery<FinancialPolicyMatrixSettings>(
    QUERY_KEY,
    fetchFinancialPolicyMatrixSettings,
    30_000,
  )
  const [draft, setDraft] = useState<FinancialPolicyMatrix | null>(null)
  const [note, setNote] = useState("")

  useEffect(() => {
    if (data && !draft) setDraft(clonePolicy(data.policy))
  }, [data, draft])

  const hasChanges = useMemo(
    () => Boolean(data && draft && JSON.stringify(data.policy) !== JSON.stringify(draft)),
    [data, draft],
  )

  const saveMutation = useMutation({
    mutationFn: ({ policy, changeNote }: { policy: FinancialPolicyMatrix; changeNote: string }) =>
      updateFinancialPolicyMatrix({ policy, note: changeNote }),
    onSuccess: settings => {
      queryClient.setQueryData(QUERY_KEY, settings)
      setDraft(clonePolicy(settings.policy))
      setNote("")
    },
  })
  const validateMutation = useMutation({
    mutationFn: (policy: FinancialPolicyMatrix) => validateFinancialPolicyMatrix(policy),
  })

  if (isLoading) return <LoadingSpinner message="Loading policy matrix..." />
  if (error || !data || !draft) return <ErrorMessage message={String(error) || "Failed to load policy matrix"} />

  const setPolicy = (updater: (policy: FinancialPolicyMatrix) => FinancialPolicyMatrix) => {
    setDraft(prev => updater(clonePolicy(prev ?? data.policy)))
    validateMutation.reset()
  }

  const setRule = (index: number, updater: (rule: FinancialPolicyRule) => FinancialPolicyRule) => {
    setPolicy(policy => ({
      ...policy,
      rules: policy.rules.map((rule, ruleIndex) => ruleIndex === index ? updater({ ...rule, match: { ...rule.match }, limits: { ...rule.limits } }) : rule),
    }))
  }

  const sortedRules = draft.rules
    .map((rule, index) => ({ rule, index }))
    .sort((a, b) => b.rule.priority - a.rule.priority || a.rule.id.localeCompare(b.rule.id))
  const canSave = hasChanges && note.trim().length > 0 && !saveMutation.isPending

  return (
    <div className="max-w-6xl">
      <PageHeader
        title="Policy Matrix"
        subtitle="Financial approval rules for policy gates, self-apply, and break-glass requests."
        actions={(
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => validateMutation.mutate(draft)}
              disabled={validateMutation.isPending}
              className="theme-button-base theme-button-secondary px-3"
            >
              <CheckCircle2 size={15} aria-hidden="true" />
              Validate
            </button>
            <button
              type="button"
              onClick={() => setPolicy(() => clonePolicy(data.default_policy))}
              className="theme-button-base theme-button-secondary px-3"
            >
              <RotateCcw size={15} aria-hidden="true" />
              Reset
            </button>
            <button
              type="button"
              onClick={() => saveMutation.mutate({ policy: draft, changeNote: note.trim() })}
              disabled={!canSave}
              className="theme-button-base theme-button-primary px-3 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Save size={15} aria-hidden="true" />
              Save
            </button>
          </div>
        )}
      />

      <div className="mb-5 grid gap-3 md:grid-cols-[1fr_18rem]">
        <label>
          <span className="theme-field-label">Policy ID</span>
          <input
            value={draft.policy_id}
            onChange={event => setPolicy(policy => ({ ...policy, policy_id: event.target.value }))}
            className="theme-input mt-2 w-full px-3 py-2 text-sm"
          />
        </label>
        <label>
          <span className="theme-field-label">Change Note</span>
          <input
            value={note}
            onChange={event => setNote(event.target.value)}
            className="theme-input mt-2 w-full px-3 py-2 text-sm"
            placeholder="Required before saving"
          />
        </label>
      </div>

      {validateMutation.data && (
        <div className={cn(
          "mb-5 rounded-md border px-3 py-2 text-sm",
          validateMutation.data.valid
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200"
            : "border-red-500/30 bg-red-500/10 text-red-800 dark:text-red-200",
        )}>
          {validateMutation.data.valid ? "Policy matrix is valid." : validateMutation.data.errors.join(" ")}
        </div>
      )}
      {saveMutation.isError && <ErrorMessage message={String(saveMutation.error)} />}

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted">
          {draft.rules.length} rule{draft.rules.length === 1 ? "" : "s"} · evaluated highest priority first
        </p>
        <button
          type="button"
          onClick={() => setPolicy(policy => ({ ...policy, rules: [...policy.rules, newRule(policy.rules)] }))}
          className="theme-button-base theme-button-secondary px-3"
        >
          <Plus size={15} aria-hidden="true" />
          Add Rule
        </button>
      </div>

      <div className="space-y-4">
        {sortedRules.map(({ rule, index }) => (
          <SurfaceCard key={`${rule.id}-${index}`} className="p-5">
            <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="section-title">{rule.id || "Untitled Rule"}</h2>
                <p className="mt-1 text-xs text-muted">{ruleSummary(rule)}</p>
              </div>
              <div className="flex items-center gap-3">
                <Toggle
                  label="Enabled"
                  checked={rule.enabled}
                  onChange={enabled => setRule(index, current => ({ ...current, enabled }))}
                />
                <button
                  type="button"
                  onClick={() => setPolicy(policy => ({ ...policy, rules: policy.rules.filter((_rule, ruleIndex) => ruleIndex !== index) }))}
                  className="theme-button-base theme-button-secondary px-3"
                  aria-label={`Remove rule ${rule.id}`}
                  title="Remove rule"
                >
                  <Trash2 size={15} aria-hidden="true" />
                </button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-[1.4fr_8rem_12rem_12rem]">
              <TextInput
                label="Rule ID"
                value={rule.id}
                onChange={id => setRule(index, current => ({ ...current, id }))}
              />
              <TextInput
                label="Priority"
                value={String(rule.priority)}
                type="number"
                onChange={value => setRule(index, current => ({ ...current, priority: Number(value || 0) }))}
              />
              <SelectInput
                label="Outcome"
                value={rule.outcome}
                onChange={value => setRule(index, current => ({ ...current, outcome: value as FinancialPolicyOutcome }))}
                options={data.metadata.outcomes.map(value => ({ value, label: labelFor(value) }))}
              />
              <SelectInput
                label="Approval Mode"
                value={rule.approval_mode ?? ""}
                onChange={value => setRule(index, current => ({
                  ...current,
                  approval_mode: value ? value as FinancialPolicyApprovalMode : null,
                }))}
                options={[
                  { value: "", label: "Request Default" },
                  ...data.metadata.approval_modes.map(value => ({ value, label: labelFor(value) })),
                ]}
              />
            </div>

            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              <div className="space-y-4">
                {MATCH_FIELDS.map(field => {
                  const options = field.optionsKey ? data.metadata[field.optionsKey] as string[] : []
                  return (
                    <MultiSelectPills
                      key={field.key}
                      label={field.label}
                      options={options}
                      values={rule.match[field.key]}
                      onChange={values => setRule(index, current => ({
                        ...current,
                        match: { ...current.match, [field.key]: values },
                      }))}
                    />
                  )
                })}
              </div>

              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-2">
                  {TEXT_MATCH_FIELDS.map(field => (
                    <TextInput
                      key={field.key}
                      label={field.label}
                      value={listToText(rule.match[field.key])}
                      placeholder={field.placeholder}
                      onChange={value => setRule(index, current => ({
                        ...current,
                        match: { ...current.match, [field.key]: textToList(value) },
                      }))}
                    />
                  ))}
                </div>

                <div>
                  <p className="theme-field-label">Limit Overrides</p>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    {data.metadata.limit_keys.map(limitKey => (
                      <label key={limitKey} className="block">
                        <span className="text-xs font-medium text-muted">{labelFor(limitKey)}</span>
                        <input
                          type="number"
                          step="0.01"
                          value={rule.limits[limitKey] ?? ""}
                          placeholder={String(data.limit_defaults[limitKey] ?? "")}
                          onChange={event => setRule(index, current => {
                            const limits = { ...current.limits }
                            const raw = event.target.value
                            if (raw === "") delete limits[limitKey]
                            else limits[limitKey] = Number(raw)
                            return { ...current, limits }
                          })}
                          className="theme-input mt-1 w-full px-3 py-2 text-sm"
                        />
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 grid gap-3 md:grid-cols-2">
              <label>
                <span className="theme-field-label">Reason</span>
                <textarea
                  value={rule.reason}
                  onChange={event => setRule(index, current => ({ ...current, reason: event.target.value }))}
                  className="theme-input mt-2 min-h-[5rem] w-full px-3 py-2 text-sm"
                />
              </label>
              <label>
                <span className="theme-field-label">Remediation</span>
                <textarea
                  value={rule.remediation}
                  onChange={event => setRule(index, current => ({ ...current, remediation: event.target.value }))}
                  className="theme-input mt-2 min-h-[5rem] w-full px-3 py-2 text-sm"
                />
              </label>
            </div>
          </SurfaceCard>
        ))}
      </div>
    </div>
  )
}
