import { AlertCircle, BriefcaseBusiness, Loader2, Target, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AgentWorkflow } from "@/lib/api"

interface AgentWorkflowLauncherProps {
  workflows: AgentWorkflow[]
  isLoading: boolean
  isError: boolean
  isStreaming: boolean
  workflowTicker: string
  onTickerChange: (value: string) => void
  onWorkflow: (workflow: AgentWorkflow) => void
  variant?: "panel" | "compact"
}

function WorkflowButton({
  workflow,
  disabled,
  onWorkflow,
}: {
  workflow: AgentWorkflow
  disabled: boolean
  onWorkflow: (workflow: AgentWorkflow) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onWorkflow(workflow)}
      disabled={disabled}
      className="theme-button-secondary min-h-9 justify-start rounded-lg px-3 py-2 text-left text-xs font-medium text-muted transition-colors hover:text-app disabled:cursor-not-allowed disabled:opacity-45"
      title={workflow.description}
    >
      {workflow.label}
    </button>
  )
}

export function AgentWorkflowLauncher({
  workflows,
  isLoading,
  isError,
  isStreaming,
  workflowTicker,
  onTickerChange,
  onWorkflow,
  variant = "panel",
}: AgentWorkflowLauncherProps) {
  const portfolioWorkflows = workflows.filter(workflow => !workflow.requiresTicker)
  const positionWorkflows = workflows.filter(workflow => workflow.requiresTicker)
  const hasTicker = Boolean(workflowTicker.trim())

  return (
    <section
      className={cn(
        "rounded-xl border border-app bg-card-muted p-3",
        variant === "panel" ? "space-y-4" : "space-y-3",
      )}
      aria-labelledby="agent-workflows-title"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-app bg-card text-link">
            <Zap size={15} aria-hidden="true" />
          </span>
          <div>
            <h3 id="agent-workflows-title" className="text-sm font-semibold text-app">Workflows</h3>
            <p className="text-xs text-subtle">Run governed portfolio actions</p>
          </div>
        </div>
        {isLoading && <Loader2 size={15} className="animate-spin text-muted" aria-label="Loading workflows" />}
      </div>

      {isError ? (
        <div className="flex items-start gap-2 rounded-lg border border-app bg-card px-3 py-2 text-xs text-muted">
          <AlertCircle size={14} className="mt-0.5 text-negative" aria-hidden="true" />
          <span>Workflows are unavailable right now.</span>
        </div>
      ) : !isLoading && workflows.length === 0 ? (
        <div className="rounded-lg border border-app bg-card px-3 py-2 text-xs text-muted">
          No workflows are configured.
        </div>
      ) : (
        <div className="space-y-4">
          {portfolioWorkflows.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-subtle">
                <BriefcaseBusiness size={13} aria-hidden="true" />
                <span>Portfolio</span>
              </div>
              <div className="grid grid-cols-1 gap-2">
                {portfolioWorkflows.map(workflow => (
                  <WorkflowButton
                    key={workflow.name}
                    workflow={workflow}
                    disabled={isStreaming}
                    onWorkflow={onWorkflow}
                  />
                ))}
              </div>
            </div>
          )}

          {positionWorkflows.length > 0 && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-subtle">
                  <Target size={13} aria-hidden="true" />
                  <span>Position</span>
                </div>
                <input
                  type="text"
                  value={workflowTicker}
                  onChange={event => onTickerChange(event.target.value.toUpperCase())}
                  placeholder="TICKER"
                  aria-label="Position workflow ticker"
                  autoCapitalize="characters"
                  spellCheck={false}
                  className="theme-input mono-text h-9 min-h-9 w-24 rounded-lg px-2 py-1 text-xs uppercase"
                />
              </div>
              <div className="grid grid-cols-1 gap-2">
                {positionWorkflows.map(workflow => (
                  <WorkflowButton
                    key={workflow.name}
                    workflow={workflow}
                    disabled={isStreaming || !hasTicker}
                    onWorkflow={onWorkflow}
                  />
                ))}
              </div>
              {!hasTicker && (
                <p className="text-[11px] leading-4 text-subtle">Enter a ticker to enable position workflows.</p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
