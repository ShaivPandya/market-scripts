import { useEffect, useState } from "react"

import { ActionButton, SelectInput, TextInput } from "@/components/shared/FormControls"
import { Dialog } from "@/components/shared/Dialog"
import type { TriggerMutationBody } from "@/lib/api"

export interface EditableWatchTrigger {
  id?: number | string
  condition?: string | null
  trigger_type?: string | null
  ticker?: string | null
  expires_at?: string | null
  definition?: Record<string, unknown> | null
}

const WATCH_TRIGGER_TYPE_OPTIONS = [
  { value: "price_level", label: "Price level" },
  { value: "technical", label: "Technical" },
  { value: "fundamental", label: "Fundamental" },
  { value: "fundamental_news", label: "Fundamental news" },
  { value: "event", label: "Event" },
  { value: "news_event", label: "News event" },
  { value: "macro", label: "Macro" },
  { value: "custom", label: "Custom" },
]

function formatDefinition(value: Record<string, unknown> | null | undefined): string {
  if (!value || !Object.keys(value).length) return ""
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return ""
  }
}

function parseDefinition(value: string): Record<string, unknown> | null {
  const text = value.trim()
  if (!text) return null
  const parsed = JSON.parse(text)
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Definition must be a JSON object.")
  }
  return parsed as Record<string, unknown>
}

interface WatchTriggerEditDialogProps {
  open: boolean
  trigger: EditableWatchTrigger | null
  title: string
  description: string
  submitLabel: string
  loading?: boolean
  error?: string | null
  onOpenChange: (open: boolean) => void
  onSubmit: (body: TriggerMutationBody) => void | Promise<void>
}

export function WatchTriggerEditDialog({
  open,
  trigger,
  title,
  description,
  submitLabel,
  loading,
  error,
  onOpenChange,
  onSubmit,
}: WatchTriggerEditDialogProps) {
  const [condition, setCondition] = useState("")
  const [triggerType, setTriggerType] = useState("custom")
  const [ticker, setTicker] = useState("")
  const [expiresAt, setExpiresAt] = useState("")
  const [definition, setDefinition] = useState("")
  const [definitionError, setDefinitionError] = useState<string | null>(null)

  useEffect(() => {
    if (!open || !trigger) return
    setCondition(String(trigger.condition ?? ""))
    setTriggerType(String(trigger.trigger_type || "custom"))
    setTicker(String(trigger.ticker ?? ""))
    setExpiresAt(String(trigger.expires_at ?? ""))
    setDefinition(formatDefinition(trigger.definition))
    setDefinitionError(null)
  }, [open, trigger])

  async function submit() {
    setDefinitionError(null)
    let parsedDefinition: Record<string, unknown> | null
    try {
      parsedDefinition = parseDefinition(definition)
    } catch (err) {
      setDefinitionError(err instanceof Error ? err.message : "Definition must be valid JSON.")
      return
    }
    await onSubmit({
      condition: condition.trim(),
      trigger_type: triggerType,
      ticker: ticker.trim() || null,
      expires_at: expiresAt.trim() || null,
      definition: parsedDefinition,
    })
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      maxWidth="max-w-2xl"
    >
      <div className="space-y-4">
        <TextInput
          id="watch-trigger-condition"
          label="Condition"
          value={condition}
          onChange={setCondition}
          placeholder="Condition to monitor"
        />
        <div className="grid gap-3 sm:grid-cols-2">
          <SelectInput
            id="watch-trigger-type"
            label="Trigger type"
            value={triggerType}
            onChange={setTriggerType}
            options={WATCH_TRIGGER_TYPE_OPTIONS}
          />
          <TextInput
            id="watch-trigger-ticker"
            label="Ticker"
            value={ticker}
            onChange={setTicker}
            uppercase
            placeholder="Optional"
          />
        </div>
        <TextInput
          id="watch-trigger-expires"
          label="Expires"
          value={expiresAt}
          onChange={setExpiresAt}
          placeholder="Optional ISO timestamp"
        />
        <div>
          <label htmlFor="watch-trigger-definition" className="theme-field-label">
            Definition JSON
          </label>
          <textarea
            id="watch-trigger-definition"
            value={definition}
            onChange={event => setDefinition(event.target.value)}
            className="theme-input mt-1 min-h-[130px] w-full font-mono text-xs"
            placeholder="Optional machine-readable JSON object"
          />
          {definitionError && <p className="theme-field-caption theme-field-error mt-1">{definitionError}</p>}
        </div>
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-lg border border-app px-3 py-2 text-sm font-medium text-muted hover:text-app"
          >
            Cancel
          </button>
          <ActionButton
            onClick={submit}
            loading={loading}
            loadingText="Staging..."
            disabled={!condition.trim()}
            className="w-auto px-4"
          >
            {submitLabel}
          </ActionButton>
        </div>
      </div>
    </Dialog>
  )
}
