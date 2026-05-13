import { Save } from "lucide-react"
import { Toggle } from "@/components/shared/FormControls"
import type { AgentPreferenceLevel, AgentResponsePreferences } from "@/lib/api"

const LEVEL_OPTIONS: { value: AgentPreferenceLevel; label: string }[] = [
  { value: "less", label: "Less" },
  { value: "balanced", label: "Balanced" },
  { value: "more", label: "More" },
]

type LevelPreferenceKey = "warmth" | "enthusiasm" | "headers_lists" | "emoji"

interface AgentPreferencesPanelProps {
  draftPreferences: AgentResponsePreferences
  onChange: (updater: (prev: AgentResponsePreferences) => AgentResponsePreferences) => void
  onSave: () => void
  isSaving: boolean
  saveError: string | null
  preferencesUnavailable: boolean
}

function PreferenceSelect({
  label,
  value,
  onChange,
}: {
  label: string
  value: AgentPreferenceLevel
  onChange: (value: AgentPreferenceLevel) => void
}) {
  return (
    <label className="flex items-center justify-between gap-3 py-2">
      <span className="text-sm font-medium text-app">{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value as AgentPreferenceLevel)}
        className="theme-input h-10 min-h-10 w-32 rounded-lg px-2 py-1.5 text-sm"
      >
        {LEVEL_OPTIONS.map(option => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  )
}

export function AgentPreferencesPanel({
  draftPreferences,
  onChange,
  onSave,
  isSaving,
  saveError,
  preferencesUnavailable,
}: AgentPreferencesPanelProps) {
  function updateLevel(key: LevelPreferenceKey, value: AgentPreferenceLevel) {
    onChange(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="flex-1 overflow-y-auto bg-app px-4 py-4">
      <div className="mx-auto w-full max-w-[42rem] space-y-4">
        <section className="rounded-xl border border-app bg-card p-4">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-app">Response Profile</h3>
            <p className="mt-1 text-xs leading-5 text-muted">Tone and structure defaults for future turns.</p>
          </div>
          <label className="flex items-center justify-between gap-3">
            <span>
              <span className="block text-sm font-medium text-app">Personality</span>
              <span className="block text-xs text-muted">Default response posture</span>
            </span>
            <select
              value={draftPreferences.personality}
              onChange={event => onChange(prev => ({
                ...prev,
                personality: event.target.value as AgentResponsePreferences["personality"],
              }))}
              className="theme-input h-10 min-h-10 w-36 rounded-lg px-3 py-1.5 text-sm"
            >
              <option value="pragmatic">Pragmatic</option>
              <option value="friendly">Friendly</option>
            </select>
          </label>
        </section>

        <section className="rounded-xl border border-app bg-card p-4">
          <h3 className="mb-2 text-sm font-semibold text-app">Style Controls</h3>
          <div className="divide-y divide-app">
            <PreferenceSelect
              label="Warmth"
              value={draftPreferences.warmth}
              onChange={value => updateLevel("warmth", value)}
            />
            <PreferenceSelect
              label="Enthusiasm"
              value={draftPreferences.enthusiasm}
              onChange={value => updateLevel("enthusiasm", value)}
            />
            <PreferenceSelect
              label="Headers and lists"
              value={draftPreferences.headers_lists}
              onChange={value => updateLevel("headers_lists", value)}
            />
            <PreferenceSelect
              label="Emoji"
              value={draftPreferences.emoji}
              onChange={value => updateLevel("emoji", value)}
            />
          </div>
        </section>

        <section className="rounded-xl border border-app bg-card p-4">
          <h3 className="mb-3 text-sm font-semibold text-app">Reasoning</h3>
          <div className="space-y-3">
            <Toggle
              label="Fast Answers"
              description="Prefer direct answers for simple questions"
              checked={draftPreferences.fast_answers}
              onChange={checked => onChange(prev => ({ ...prev, fast_answers: checked }))}
            />
            <Toggle
              label="Thinking"
              description="Use deeper model reasoning for complex turns"
              checked={draftPreferences.thinking_enabled}
              onChange={checked => onChange(prev => ({ ...prev, thinking_enabled: checked }))}
            />
          </div>
        </section>

        <section className="rounded-xl border border-app bg-card p-4">
          <label htmlFor="agent-custom-instructions" className="text-sm font-semibold text-app">
            Custom Instructions
          </label>
          <textarea
            id="agent-custom-instructions"
            value={draftPreferences.custom_instructions ?? ""}
            onChange={event => onChange(prev => ({ ...prev, custom_instructions: event.target.value }))}
            placeholder="End responses after answering. Do not ask follow-up questions."
            rows={6}
            maxLength={2000}
            className="theme-input mt-2 min-h-[9rem] w-full resize-y rounded-xl text-sm"
          />
        </section>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-app bg-card-muted px-4 py-3">
          <div className="min-w-0">
            {saveError ? (
              <p className="text-xs font-medium text-negative">{saveError}</p>
            ) : preferencesUnavailable ? (
              <p className="text-xs text-muted">Saved preferences unavailable. Showing this browser's cache.</p>
            ) : (
              <p className="text-xs text-muted">Preferences are saved to your account.</p>
            )}
          </div>
          <button
            type="button"
            onClick={onSave}
            disabled={isSaving}
            className="theme-button-base theme-button-primary min-h-10 px-4 text-sm disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Save size={14} aria-hidden="true" />
            {isSaving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  )
}
