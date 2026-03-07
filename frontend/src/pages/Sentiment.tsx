import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchSentimentPutCall, fetchSentimentSurveys, fetchSentimentVolatility } from "@/lib/api"
import { TimeSeriesChart } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { DataTable } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"

type Tab = "Put/Call" | "Surveys" | "Volatility"
const TABS: Tab[] = ["Put/Call", "Surveys", "Volatility"]

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt2(v: unknown): string {
  return v != null ? Number(v).toFixed(2) : "N/A"
}

function mkSeries(rows: Record<string, unknown>[], dateKey: string, valueKey: string) {
  return rows
    .filter(r => r[dateKey] && r[valueKey] != null)
    .map(r => ({ date: String(r[dateKey]), value: Number(r[valueKey]) }))
}

// ─── Put/Call Tab ─────────────────────────────────────────────────────────────

type PCEntry = { ticker: string; calls: number; puts: number; ratio: number; as_of: string; breakdown?: { expiry: string; calls: number; puts: number; ratio: number | null }[] }

function PutCallTab() {
  const { data, isLoading, error } = useApiQuery(
    ["sentiment-put-call"],
    () => fetchSentimentPutCall(180),
    5 * 60 * 1000,
  )

  if (isLoading) return <LoadingSpinner message="Computing Put/Call ratios from options chains..." />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const equity = data?.equity as PCEntry | undefined
  const spy = data?.spy as PCEntry | undefined
  const qqq = data?.qqq as PCEntry | undefined
  const iwm = data?.iwm as PCEntry | undefined
  const asOf = spy?.as_of ?? equity?.as_of ?? ""

  const breakdownRows: Record<string, unknown>[] = (spy?.breakdown ?? []).map(b => ({
    expiry: b.expiry,
    calls: b.calls.toLocaleString(),
    puts: b.puts.toLocaleString(),
    ratio: b.ratio != null ? b.ratio.toFixed(3) : "N/A",
  }))

  return (
    <div>
      <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-1">
        Put/Call Ratio — Live Snapshot
      </p>
      <p className="text-xs text-gray-400 mb-4">
        Computed from Yahoo Finance options chains (SPY, QQQ, IWM). Ratio &gt; 1.0 = more
        puts than calls (bearish tilt). As of: {asOf || "today"}.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Equity Aggregate"
          value={equity ? equity.ratio.toFixed(3) : "N/A"}
          subtitle="SPY + QQQ + IWM"
        />
        <MetricCard
          title="SPY P/C"
          value={spy ? spy.ratio.toFixed(3) : "N/A"}
          subtitle={spy ? `${(spy.puts / 1000).toFixed(0)}K puts / ${(spy.calls / 1000).toFixed(0)}K calls` : ""}
        />
        <MetricCard
          title="QQQ P/C"
          value={qqq ? qqq.ratio.toFixed(3) : "N/A"}
          subtitle={qqq ? `${(qqq.puts / 1000).toFixed(0)}K puts / ${(qqq.calls / 1000).toFixed(0)}K calls` : ""}
        />
        <MetricCard
          title="IWM P/C"
          value={iwm ? iwm.ratio.toFixed(3) : "N/A"}
          subtitle={iwm ? `${(iwm.puts / 1000).toFixed(0)}K puts / ${(iwm.calls / 1000).toFixed(0)}K calls` : ""}
        />
      </div>

      {breakdownRows.length > 0 && (
        <>
          <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2">
            SPY Breakdown by Expiry
          </p>
          <DataTable
            columns={[
              { key: "expiry", header: "Expiry" },
              { key: "calls", header: "Calls Volume" },
              { key: "puts", header: "Puts Volume" },
              { key: "ratio", header: "P/C Ratio" },
            ]}
            rows={breakdownRows}
          />
        </>
      )}
    </div>
  )
}

// ─── Surveys Tab ─────────────────────────────────────────────────────────────

function SurveysTab() {
  const { data, isLoading, error } = useApiQuery(
    ["sentiment-surveys"],
    fetchSentimentSurveys,
    60 * 60 * 1000,
  )

  if (isLoading) return <LoadingSpinner message="Fetching AAII and NAAIM data..." />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const aaii: Record<string, unknown>[] = Array.isArray(data?.aaii) ? data.aaii : []
  const naaim: Record<string, unknown>[] = Array.isArray(data?.naaim) ? data.naaim : []

  const latestAaii = aaii[aaii.length - 1] ?? {}
  const latestNaaim = naaim[naaim.length - 1] ?? {}

  const spreadSeries = mkSeries(aaii, "date", "spread")
  const naaimSeries = mkSeries(naaim, "date", "exposure")

  return (
    <div className="space-y-8">
      {/* AAII */}
      <div>
        <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-1">
          AAII Investor Sentiment Survey
        </p>
        <p className="text-xs text-gray-400 mb-4">
          Weekly survey of individual investors. Bull-Bear spread &gt; +30 historically signals
          elevated bullish sentiment; &lt; -10 signals elevated fear.
        </p>

        {aaii.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <MetricCard
                title="Bullish"
                value={latestAaii["bull"] != null ? `${Number(latestAaii["bull"]).toFixed(1)}%` : "N/A"}
              />
              <MetricCard
                title="Bearish"
                value={latestAaii["bear"] != null ? `${Number(latestAaii["bear"]).toFixed(1)}%` : "N/A"}
              />
              <MetricCard
                title="Neutral"
                value={latestAaii["neutral"] != null ? `${Number(latestAaii["neutral"]).toFixed(1)}%` : "N/A"}
              />
              <MetricCard
                title="Bull-Bear Spread"
                value={latestAaii["spread"] != null ? `${Number(latestAaii["spread"]) >= 0 ? "+" : ""}${Number(latestAaii["spread"]).toFixed(1)}%` : "N/A"}
              />
            </div>

            <TimeSeriesChart
              multiData={aaii as Record<string, unknown>[]}
              series={[
                { key: "bull", color: "#10b981", strokeWidth: 1.5 },
                { key: "bear", color: "#ef4444", strokeWidth: 1.5 },
                { key: "neutral", color: "#9ca3af", strokeWidth: 1.5 },
              ]}
              height={180}
              label="Bull / Bear / Neutral (%)"
              zeroLine={false}
              yFormatter={v => `${v.toFixed(0)}%`}
            />

            {spreadSeries.length > 0 && (
              <div className="mt-4">
                <TimeSeriesChart
                  data={spreadSeries}
                  height={140}
                  label="Bull-Bear Spread"
                  zeroLine
                  yFormatter={v => `${v >= 0 ? "+" : ""}${v.toFixed(0)}%`}
                />
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-gray-400">No AAII data available.</p>
        )}
      </div>

      {/* NAAIM */}
      <div>
        <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-1">
          NAAIM Exposure Index
        </p>
        <p className="text-xs text-gray-400 mb-4">
          Weekly equity exposure reported by active investment managers. Readings above 100
          indicate leveraged long exposure; below 0 indicate net short.
        </p>

        {naaim.length > 0 ? (
          <>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <MetricCard
                title="Exposure (latest)"
                value={latestNaaim["exposure"] != null ? `${Number(latestNaaim["exposure"]).toFixed(1)}` : "N/A"}
              />
              <MetricCard
                title="Week of"
                value={latestNaaim["date"] != null ? String(latestNaaim["date"]) : "N/A"}
              />
            </div>

            <TimeSeriesChart
              data={naaimSeries}
              height={180}
              label="NAAIM Exposure Index"
              zeroLine
              yFormatter={v => v.toFixed(0)}
            />
          </>
        ) : (
          <p className="text-sm text-gray-400">No NAAIM data available.</p>
        )}
      </div>
    </div>
  )
}

// ─── Volatility Tab ──────────────────────────────────────────────────────────

function VolatilityTab() {
  const { data, isLoading, error } = useApiQuery(
    ["sentiment-volatility"],
    () => fetchSentimentVolatility(365),
    5 * 60 * 1000,
  )

  if (isLoading) return <LoadingSpinner message="Fetching VIX, VXN, VVIX..." />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const rows: Record<string, unknown>[] = Array.isArray(data) ? data : []
  const latest = rows[rows.length - 1] ?? {}

  const vixSeries = mkSeries(rows, "date", "vix")
  const vxnSeries = mkSeries(rows, "date", "vxn")
  const vvixSeries = mkSeries(rows, "date", "vvix")

  return (
    <div>
      <p className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-1">
        Volatility Indices
      </p>
      <p className="text-xs text-gray-400 mb-4">
        VIX = S&P 500 implied vol (1M). VXN = NASDAQ 100 implied vol. VVIX = volatility of VIX
        (options demand on VIX itself).
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <MetricCard title="VIX" value={fmt2(latest["vix"])} />
        <MetricCard title="VXN" value={fmt2(latest["vxn"])} />
        <MetricCard title="VVIX" value={fmt2(latest["vvix"])} />
      </div>

      {vixSeries.length > 0 && (
        <div className="space-y-4">
          <TimeSeriesChart
            data={vixSeries}
            height={160}
            color="#ef4444"
            label="VIX (S&P 500 Implied Vol)"
            zeroLine={false}
            yFormatter={v => v.toFixed(1)}
          />
          <TimeSeriesChart
            data={vxnSeries}
            height={160}
            color="#f97316"
            label="VXN (NASDAQ 100 Implied Vol)"
            zeroLine={false}
            yFormatter={v => v.toFixed(1)}
          />
          <TimeSeriesChart
            data={vvixSeries}
            height={160}
            color="#8b5cf6"
            label="VVIX (Volatility of VIX)"
            zeroLine={false}
            yFormatter={v => v.toFixed(1)}
          />
        </div>
      )}
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function Sentiment() {
  const [tab, setTab] = useState<Tab>("Put/Call")

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Sentiment</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Options market, investor surveys, and volatility signals
          </p>
        </div>
        <RefreshButton />
      </div>

      <div className="mb-6">
        <SegmentedControl
          options={TABS.map(t => ({ value: t, label: t }))}
          value={tab}
          onChange={setTab}
        />
      </div>

      {tab === "Put/Call" && <PutCallTab />}
      {tab === "Surveys" && <SurveysTab />}
      {tab === "Volatility" && <VolatilityTab />}
    </div>
  )
}
