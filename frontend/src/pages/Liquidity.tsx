import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchLiquidity } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorZscore, colorPolarityChange } from "@/lib/colors"

export function Liquidity() {
  const [skipEcb, setSkipEcb] = useState(false)
  const { data, isLoading, error } = useApiQuery(
    ["liquidity", skipEcb],
    () => fetchLiquidity(skipEcb),
  )

  const REGION_LABELS: Record<string, string> = {
    us: "United States",
    europe: "Europe",
    japan: "Japan",
  }

  const regimeSignal = (color: string): "success" | "info" | "warning" | "error" => {
    if (color === "green") return "success"
    if (color === "cyan") return "info"
    if (color === "yellow") return "warning"
    return "error"
  }

  const componentCols: ColumnDef[] = [
    { key: "region", header: "Region" },
    { key: "label", header: "Component" },
    { key: "value_str", header: "Value" },
    { key: "z_score_str", header: "Z-Score", colorFn: (_, row) => colorZscore(row["z_score"]) },
    { key: "weight_str", header: "Weight" },
    { key: "contribution_str", header: "Contribution", colorFn: (_, row) => colorZscore(row["contribution"]) },
    { key: "signal", header: "Signal", colorFn: v => v === "supportive" ? "#00c853; font-weight: bold" : v === "tightening" ? "#ff1744; font-weight: bold" : "gray" },
  ]

  const changesCols: ColumnDef[] = [
    { key: "series", header: "Series" },
    { key: "1w", header: "1W", colorFn: (v, row) => colorPolarityChange(v, (row["polarity"] as number) ?? 1) },
    { key: "1m", header: "1M", colorFn: (v, row) => colorPolarityChange(v, (row["polarity"] as number) ?? 1) },
    { key: "3m", header: "3M", colorFn: (v, row) => colorPolarityChange(v, (row["polarity"] as number) ?? 1) },
  ]

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Liquidity Dashboard</h1>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-500 select-none cursor-pointer">
            <input
              type="checkbox"
              checked={skipEcb}
              onChange={e => setSkipEcb(e.target.checked)}
              className="rounded"
            />
            Skip ECB data
          </label>
          <RefreshButton queryKeys={[["liquidity", skipEcb]]} />
        </div>
      </div>

      {isLoading && <LoadingSpinner message="Fetching liquidity data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {data.composite_score == null ? (
            <p className="text-yellow-600">Insufficient data to compute liquidity score.</p>
          ) : (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                <MetricCard
                  title="Composite Score"
                  value={`${Number(data.composite_score) >= 0 ? "+" : ""}${Number(data.composite_score).toFixed(2)}`}
                />
                <MetricCard
                  title="Regime"
                  value={String(data.regime ?? "unknown").toUpperCase()}
                  signal={regimeSignal(data.regime_color ?? "red")}
                  signalLabel={String(data.regime ?? "").toUpperCase()}
                />
                <MetricCard title="As of" value={String(data.latest_date ?? "N/A")} />
              </div>

              <section className="mb-8">
                <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Regional Liquidity Scores</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {Object.entries(data.regional_scores ?? {}).map(([key, region]) => {
                    const r = region as { score: number; regime: string; color: string }
                    return (
                      <MetricCard
                        key={key}
                        title={REGION_LABELS[key] ?? key}
                        value={`${r.score >= 0 ? "+" : ""}${r.score.toFixed(2)}`}
                        signal={regimeSignal(r.color)}
                        signalLabel={r.regime.toUpperCase()}
                      />
                    )
                  })}
                </div>
              </section>

              <section className="mb-8">
                <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Components</h2>
                <DataTable
                  columns={componentCols}
                  rows={(data.components ?? []).map((c: Record<string, unknown>) => {
                    const v = c["value"] as number | null
                    const z = c["z_score"] as number | null
                    const contrib = c["contribution"] as number | null
                    const kind = c["value_kind"] as string
                    let value_str = "N/A"
                    if (v != null) {
                      if (kind === "billions") value_str = `$${(v / 1000).toFixed(2)}B`
                      else if (kind === "percent") value_str = `${v.toFixed(2)}%`
                      else if (kind === "ratio") value_str = v.toFixed(3)
                      else value_str = v.toFixed(2)
                    }
                    return {
                      ...c,
                      value_str,
                      z_score_str: z != null ? `${z >= 0 ? "+" : ""}${z.toFixed(2)}` : "N/A",
                      weight_str: `${((c["weight"] as number) * 100).toFixed(0)}%`,
                      contribution_str: contrib != null ? `${contrib >= 0 ? "+" : ""}${contrib.toFixed(2)}` : "N/A",
                      signal: z != null ? (z >= 0 ? "supportive" : "tightening") : "N/A",
                    }
                  })}
                />
              </section>

              <section className="mb-8">
                <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Historical Changes</h2>
                <DataTable
                  columns={changesCols}
                  rows={Object.entries(data.changes ?? {}).map(([label, d]) => {
                    const info = d as Record<string, unknown>
                    const kind = info["value_kind"] as string
                    const fmtVal = (v: unknown) => {
                      if (v == null) return "N/A"
                      const n = Number(v)
                      if (kind === "billions") return `${n >= 0 ? "+" : ""}${(n / 1000).toFixed(2)}B`
                      if (kind === "percent") return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`
                      return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`
                    }
                    return {
                      series: label,
                      "1w": fmtVal(info["1w"]),
                      "1m": fmtVal(info["1m"]),
                      "3m": fmtVal(info["3m"]),
                      polarity: info["polarity"] ?? 1,
                    }
                  })}
                />
                <p className="text-xs text-gray-400 mt-1.5">
                  Green = liquidity-supportive changes · Red = liquidity-tightening changes
                </p>
              </section>
            </>
          )}
        </>
      )}
    </div>
  )
}
