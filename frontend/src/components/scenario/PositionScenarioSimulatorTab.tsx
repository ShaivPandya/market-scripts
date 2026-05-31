import { useMemo, useState } from "react"
import { useMutation } from "@tanstack/react-query"

import { ActionButton } from "@/components/shared/FormControls"
import { ErrorMessage, LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { cn } from "@/lib/utils"
import {
  evaluateScenarioSimulator,
  type ScenarioSimulatorEvaluateRequest,
  type ScenarioSimulatorEvaluateResponse,
  type ScenarioSimulatorOutcome,
} from "@/lib/api"

function formatMoney(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function formatPct(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return `${value.toFixed(2)}%`
}

function uncertaintyTone(level: string | undefined): string {
  if (level === "high") return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950"
  if (level === "medium") return "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
  return "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950"
}

function buildSimulatorRequest(
  ticker: string,
  position: Record<string, unknown> | null | undefined,
): ScenarioSimulatorEvaluateRequest {
  const direction = String(position?.direction || "long")
  const quantity = Number(position?.quantity ?? position?.shares ?? 0)
  const currentPrice = Number(position?.current_price ?? position?.price ?? 0)
  const notional = Number(
    position?.notional_base ?? (quantity && currentPrice ? quantity * currentPrice : 0),
  )
  const positionPayload = {
    ticker,
    direction,
    quantity: quantity || undefined,
    current_price: currentPrice || undefined,
    notional_base: notional || undefined,
    position_uid: String(position?.position_uid || position?.object_uid || `position:${ticker}`),
    average_daily_volume_notional: position?.average_daily_volume_notional,
  }
  return {
    portfolio: {
      portfolio_id: "default-portfolio",
      account_id: "default-account",
      base_currency: "USD",
      book_value: 100000,
      positions: [positionPayload],
    },
    position: positionPayload,
    candidates: [
      { action: "hold", candidate_id: "hold" },
      { action: "add", candidate_id: "add", delta: { pct_position: 0.25 } },
      { action: "trim", candidate_id: "trim", delta: { pct_position: 0.25 } },
      { action: "exit", candidate_id: "exit" },
    ],
    scenarios: [
      {
        scenario_id: "downside",
        name: "Downside",
        price_move_pct: -10,
        probability: 0.5,
      },
      {
        scenario_id: "base",
        name: "Base",
        price_move_pct: 0,
        probability: 0.25,
      },
      {
        scenario_id: "upside",
        name: "Upside",
        price_move_pct: 10,
        probability: 0.25,
      },
    ],
    assumptions: [
      {
        name: "execution_friction",
        value: {
          transaction_cost_bps: 5,
          slippage_bps: 10,
          market_impact_bps: 5,
          max_adv_participation: 0.2,
        },
        confidence: 0.6,
      },
    ],
    enrich_from_risk_snapshot: true,
    persist: false,
  }
}

function OutcomeCard({ outcome }: { outcome: ScenarioSimulatorOutcome }) {
  const uncertainty = outcome.uncertainty
  const friction = outcome.execution_friction
  const risk = outcome.risk
  const liquidity = outcome.liquidity
  const scenarioRows = outcome.scenario_outcomes || []

  return (
    <article className="rounded-lg border border-app bg-card-muted p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold capitalize text-app">{outcome.action}</h3>
          <p className="text-xs text-subtle">Score {formatPct((outcome.ranking_score ?? 0) * 100)}</p>
        </div>
        <span className={cn("rounded px-2 py-1 text-xs font-medium capitalize", uncertaintyTone(uncertainty?.level))}>
          {uncertainty?.level || "unknown"} uncertainty
        </span>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <div className="rounded border border-app px-3 py-2">
          <p className="text-xs text-subtle">Expected P&L</p>
          <p className="mt-1 text-sm font-semibold text-app">{formatMoney(risk?.expected_pnl_base)}</p>
        </div>
        <div className="rounded border border-app px-3 py-2">
          <p className="text-xs text-subtle">Worst Loss</p>
          <p className="mt-1 text-sm font-semibold text-app">{formatMoney(risk?.worst_loss_base)}</p>
        </div>
        <div className="rounded border border-app px-3 py-2">
          <p className="text-xs text-subtle">Execution Friction</p>
          <p className="mt-1 text-sm font-semibold text-app">{formatMoney(friction?.total_friction_base)}</p>
        </div>
      </div>

      {liquidity?.status === "missing" || liquidity?.status === "constrained" ? (
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          Liquidity {liquidity.status}.{" "}
          {(liquidity.notes as string[] | undefined)?.join(" ") || "Exit timing may be understated."}
        </div>
      ) : null}

      {uncertainty?.notes?.length ? (
        <ul className="mt-3 space-y-1 text-xs text-muted">
          {uncertainty.notes.map(note => (
            <li key={note} className="rounded border border-app px-2 py-1">{note}</li>
          ))}
        </ul>
      ) : null}

      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="text-left text-subtle">
              <th className="px-2 py-1">Scenario</th>
              <th className="px-2 py-1">Gross P&L</th>
              <th className="px-2 py-1">Net P&L</th>
              <th className="px-2 py-1">Incremental</th>
            </tr>
          </thead>
          <tbody>
            {scenarioRows.map(row => (
              <tr key={String(row.scenario_id)} className="border-t border-app">
                <td className="px-2 py-1">{String(row.name || row.scenario_id)}</td>
                <td className="px-2 py-1 mono-text">{formatMoney(row.target_pnl_gross_base)}</td>
                <td className="px-2 py-1 mono-text">{formatMoney(row.target_pnl_net_base ?? row.target_pnl_base)}</td>
                <td className="px-2 py-1 mono-text">{formatMoney(row.incremental_pnl_net_base ?? row.incremental_pnl_base)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-subtle">
        Policy gate: {outcome.policy_gate?.decision || "n/a"}
        {outcome.policy_gate?.review_required ? " · review required" : ""}
      </p>
    </article>
  )
}

export function PositionScenarioSimulatorTab({
  ticker,
  position,
}: {
  ticker: string
  position: Record<string, unknown> | null | undefined
}) {
  const request = useMemo(() => buildSimulatorRequest(ticker, position), [ticker, position])
  const [result, setResult] = useState<ScenarioSimulatorEvaluateResponse | null>(null)

  const runMutation = useMutation({
    mutationFn: () => evaluateScenarioSimulator(request),
    onSuccess: data => setResult(data),
  })

  const disclosure = String(
    result?.execution_assumptions?.disclosure
      || "Scenario simulation is decision support only. Rankings do not authorize execution.",
  )

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="section-title">Action Scenario Comparison</h2>
          <p className="mt-1 text-sm text-subtle">
            Compare hold, add, trim, and exit under downside/base/upside scenarios with explicit friction assumptions.
          </p>
        </div>
        <ActionButton
          onClick={() => runMutation.mutate()}
          disabled={runMutation.isPending}
          loading={runMutation.isPending}
          loadingText="Running..."
        >
          Run comparison
        </ActionButton>
      </div>

      <div
        className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200"
        role="note"
      >
        {disclosure}
      </div>

      {runMutation.isPending && <LoadingSpinner message="Running scenario simulator..." />}
      {runMutation.error && <ErrorMessage message={String(runMutation.error)} />}
      {!runMutation.isPending && !result && !runMutation.error && (
        <p className="text-sm text-muted">Run the comparison to inspect ranked outcomes and uncertainty disclosure.</p>
      )}

      {result && (
        <>
          <div className="rounded-lg border border-app px-4 py-3 text-sm text-muted">
            <p>
              Version {result.calculation_version || "unknown"} · Generated{" "}
              {result.generated_at ? new Date(result.generated_at).toLocaleString() : "just now"}
            </p>
            <p className="mt-1">{result.comparison?.selection_policy}</p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {(result.outcomes || []).map(outcome => (
              <OutcomeCard key={outcome.candidate_id} outcome={outcome} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
