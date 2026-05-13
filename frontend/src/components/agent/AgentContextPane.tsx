import { Activity, MapPin, Route, Sparkles } from "lucide-react"
import { AgentWorkflowLauncher } from "./AgentWorkflowLauncher"
import type { QuickPromptGroup } from "./AgentChatTypes"
import type { AgentWorkflow } from "@/lib/api"
import type { ScreenContext } from "@/contexts/ScreenContext"

interface AgentContextPaneProps {
  screenContext?: ScreenContext | null
  workflows: AgentWorkflow[]
  workflowsLoading: boolean
  workflowsError: boolean
  isStreaming: boolean
  workflowTicker: string
  onTickerChange: (value: string) => void
  onWorkflow: (workflow: AgentWorkflow) => void
  promptGroups: QuickPromptGroup[]
  onPrompt: (prompt: string) => void
}

function FieldRow({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null
  return (
    <div className="rounded-lg border border-app bg-card px-3 py-2">
      <dt className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">{label}</dt>
      <dd className="mt-1 break-words text-sm text-app">{value}</dd>
    </div>
  )
}

function compactEntries(value: Record<string, string> | undefined, limit: number) {
  return Object.entries(value ?? {}).slice(0, limit)
}

export function AgentContextPane({
  screenContext,
  workflows,
  workflowsLoading,
  workflowsError,
  isStreaming,
  workflowTicker,
  onTickerChange,
  onWorkflow,
  promptGroups,
  onPrompt,
}: AgentContextPaneProps) {
  const metricEntries = compactEntries(screenContext?.metrics, 4)
  const filterEntries = compactEntries(screenContext?.filters, 3)
  const tools = screenContext?.correspondingTools?.slice(0, 5) ?? []

  return (
    <aside className="hidden w-[23rem] shrink-0 border-l border-app bg-card-muted lg:flex lg:flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <section className="rounded-xl border border-app bg-card-muted p-3">
          <div className="mb-3 flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-app bg-card text-link">
              <Activity size={15} aria-hidden="true" />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-app">Current Context</h3>
              <p className="text-xs text-subtle">Included with the next turn</p>
            </div>
          </div>

          {screenContext ? (
            <dl className="space-y-2">
              <FieldRow label="Page" value={screenContext.pageName} />
              <FieldRow label="Ticker" value={screenContext.ticker} />
              <FieldRow label="Route" value={screenContext.route} />
              <FieldRow label="Summary" value={screenContext.summary} />
              {metricEntries.length > 0 && (
                <div className="rounded-lg border border-app bg-card px-3 py-2">
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">Metrics</dt>
                  <dd className="mt-2 space-y-1">
                    {metricEntries.map(([key, value]) => (
                      <div key={key} className="flex items-baseline justify-between gap-3 text-xs">
                        <span className="text-muted">{key}</span>
                        <span className="text-right font-medium text-app">{value}</span>
                      </div>
                    ))}
                  </dd>
                </div>
              )}
              {filterEntries.length > 0 && (
                <div className="rounded-lg border border-app bg-card px-3 py-2">
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">Filters</dt>
                  <dd className="mt-2 flex flex-wrap gap-1.5">
                    {filterEntries.map(([key, value]) => (
                      <span key={key} className="rounded-md border border-app bg-card-muted px-2 py-1 text-[11px] text-muted">
                        {key}: {value}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              {tools.length > 0 && (
                <div className="rounded-lg border border-app bg-card px-3 py-2">
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">Relevant Tools</dt>
                  <dd className="mt-2 flex flex-wrap gap-1.5">
                    {tools.map(tool => (
                      <span key={tool} className="rounded-md border border-app bg-card-muted px-2 py-1 font-mono text-[11px] text-muted">
                        {tool}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
            </dl>
          ) : (
            <div className="rounded-lg border border-app bg-card px-3 py-3 text-sm text-muted">
              No page context is attached.
            </div>
          )}
        </section>

        <AgentWorkflowLauncher
          workflows={workflows}
          isLoading={workflowsLoading}
          isError={workflowsError}
          isStreaming={isStreaming}
          workflowTicker={workflowTicker}
          onTickerChange={onTickerChange}
          onWorkflow={onWorkflow}
        />

        <section className="rounded-xl border border-app bg-card-muted p-3">
          <div className="mb-3 flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-app bg-card text-link">
              <Sparkles size={15} aria-hidden="true" />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-app">Prompt Starters</h3>
              <p className="text-xs text-subtle">Analyst-ready starting points</p>
            </div>
          </div>
          <div className="space-y-3">
            {promptGroups.map(group => (
              <div key={group.title} className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-subtle">
                  <MapPin size={12} aria-hidden="true" />
                  <span>{group.title}</span>
                </div>
                <div className="grid grid-cols-1 gap-2">
                  {group.prompts.map(prompt => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => onPrompt(prompt)}
                      disabled={isStreaming}
                      className="theme-button-secondary min-h-10 justify-start rounded-lg px-3 py-2 text-left text-xs font-medium leading-5 text-muted hover:text-app disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
      <div className="border-t border-app px-4 py-3 text-[11px] text-subtle">
        <div className="flex items-center gap-1.5">
          <Route size={12} aria-hidden="true" />
          <span>Context refreshes as the workspace changes.</span>
        </div>
      </div>
    </aside>
  )
}
