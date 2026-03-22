import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { ChevronDown, Sparkles } from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { fetchSentimentPutCall, fetchSentimentSurveys, fetchSentimentVolatility, analyzeSentiment } from "@/lib/api"
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

function toFiniteNumber(v: unknown): number | null {
  const n = typeof v === "number" ? v : Number(v)
  return Number.isFinite(n) ? n : null
}

function fmt3(v: unknown): string {
  const n = toFiniteNumber(v)
  return n != null ? n.toFixed(3) : "N/A"
}

function fmtK(puts: unknown, calls: unknown): string {
  const p = toFiniteNumber(puts)
  const c = toFiniteNumber(calls)
  if (p == null || c == null) return ""
  return `${(p / 1000).toFixed(0)}K puts / ${(c / 1000).toFixed(0)}K calls`
}

function mkSeries(rows: Record<string, unknown>[], dateKey: string, valueKey: string) {
  return rows
    .filter(r => r[dateKey] && r[valueKey] != null)
    .map(r => ({ date: String(r[dateKey]), value: Number(r[valueKey]) }))
}

const SURVEYS_CACHE_KEY = "sentiment-surveys-cache-v1"

type SurveysPayload = {
  aaii: Record<string, unknown>[]
  naaim: Record<string, unknown>[]
  errors?: Record<string, unknown>
}

function isSurveysPayload(v: unknown): v is SurveysPayload {
  if (!v || typeof v !== "object") return false
  const rec = v as Record<string, unknown>
  return Array.isArray(rec.aaii) && Array.isArray(rec.naaim)
}

function loadSurveysCache(): { payload: SurveysPayload; savedAt: string } | null {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(SURVEYS_CACHE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { payload?: unknown; savedAt?: unknown }
    if (!isSurveysPayload(parsed.payload) || typeof parsed.savedAt !== "string") return null
    return { payload: parsed.payload, savedAt: parsed.savedAt }
  } catch {
    return null
  }
}

function saveSurveysCache(payload: SurveysPayload) {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(
      SURVEYS_CACHE_KEY,
      JSON.stringify({ payload, savedAt: new Date().toISOString() }),
    )
  } catch {
    // Ignore localStorage write failures.
  }
}

function formatSavedAt(iso: string): string {
  const dt = new Date(iso)
  if (Number.isNaN(dt.getTime())) return iso
  return dt.toLocaleString("en-US")
}

// ─── Put/Call Tab ─────────────────────────────────────────────────────────────

type PCEntry = { ticker: string; calls: number | null; puts: number | null; ratio: number | null; as_of: string; breakdown: { expiry: string; calls: number; puts: number; ratio: number | null }[] }

function isRecord(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === "object"
}

function parseBreakdown(v: unknown): PCEntry["breakdown"] {
  if (!Array.isArray(v)) return []
  return v
    .map((row) => {
      if (!isRecord(row)) return null
      const expiry = String(row.expiry ?? "")
      const calls = toFiniteNumber(row.calls)
      const puts = toFiniteNumber(row.puts)
      if (!expiry || calls == null || puts == null) return null
      return {
        expiry,
        calls,
        puts,
        ratio: toFiniteNumber(row.ratio),
      }
    })
    .filter((row): row is PCEntry["breakdown"][number] => row != null)
}

function parsePCEntry(v: unknown): PCEntry | null {
  if (!isRecord(v)) return null
  return {
    ticker: String(v.ticker ?? ""),
    calls: toFiniteNumber(v.calls),
    puts: toFiniteNumber(v.puts),
    ratio: toFiniteNumber(v.ratio),
    as_of: typeof v.as_of === "string" ? v.as_of : "",
    breakdown: parseBreakdown(v.breakdown),
  }
}

function PutCallTab() {
  const { data, isLoading, error } = useApiQuery(
    ["sentiment-put-call"],
    () => fetchSentimentPutCall(180),
    5 * 60 * 1000,
  )

  const payload = isRecord(data) ? data : {}
  const equity = parsePCEntry(payload.equity)
  const spy = parsePCEntry(payload.spy)
  const qqq = parsePCEntry(payload.qqq)
  const iwm = parsePCEntry(payload.iwm)
  const asOf = spy?.as_of ?? equity?.as_of ?? ""

  const expiryCurveData = useMemo(() => {
    const byExpiry = new Map<string, {
      date: string
      SPY: number | null
      QQQ: number | null
      IWM: number | null
      Equity: number | null
      equityCalls: number
      equityPuts: number
    }>()

    const ingest = (key: "SPY" | "QQQ" | "IWM", rows: PCEntry["breakdown"]) => {
      for (const row of rows ?? []) {
        const current = byExpiry.get(row.expiry) ?? {
          date: row.expiry,
          SPY: null,
          QQQ: null,
          IWM: null,
          Equity: null,
          equityCalls: 0,
          equityPuts: 0,
        }
        current[key] = row.ratio
        current.equityCalls += row.calls
        current.equityPuts += row.puts
        byExpiry.set(row.expiry, current)
      }
    }

    ingest("SPY", spy?.breakdown ?? [])
    ingest("QQQ", qqq?.breakdown ?? [])
    ingest("IWM", iwm?.breakdown ?? [])

    return Array.from(byExpiry.values())
      .map((r) => ({
        date: r.date,
        SPY: r.SPY,
        QQQ: r.QQQ,
        IWM: r.IWM,
        Equity: r.equityCalls > 0 ? Number((r.equityPuts / r.equityCalls).toFixed(3)) : null,
      }))
      .sort((a, b) => String(a.date).localeCompare(String(b.date)))
  }, [spy?.breakdown, qqq?.breakdown, iwm?.breakdown])

  if (isLoading) return <LoadingSpinner message="Computing Put/Call ratios from options chains..." />
  if (error || !data) return <ErrorMessage message={String(error)} />

  const breakdownRows: Record<string, unknown>[] = (spy?.breakdown ?? []).map(b => ({
    expiry: b.expiry,
    calls: b.calls.toLocaleString(),
    puts: b.puts.toLocaleString(),
    ratio: fmt3(b.ratio),
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
          value={fmt3(equity?.ratio)}
          subtitle="SPY + QQQ + IWM"
        />
        <MetricCard
          title="SPY P/C"
          value={fmt3(spy?.ratio)}
          subtitle={fmtK(spy?.puts, spy?.calls)}
        />
        <MetricCard
          title="QQQ P/C"
          value={fmt3(qqq?.ratio)}
          subtitle={fmtK(qqq?.puts, qqq?.calls)}
        />
        <MetricCard
          title="IWM P/C"
          value={fmt3(iwm?.ratio)}
          subtitle={fmtK(iwm?.puts, iwm?.calls)}
        />
      </div>

      {expiryCurveData.length > 0 && (
        <div className="mb-6">
          <TimeSeriesChart
            multiData={expiryCurveData}
            series={[
              { key: "Equity", color: "#111827", strokeWidth: 2 },
              { key: "SPY", color: "#2563eb", strokeWidth: 1.6 },
              { key: "QQQ", color: "#16a34a", strokeWidth: 1.6 },
              { key: "IWM", color: "#f59e0b", strokeWidth: 1.6 },
            ]}
            height={200}
            label="Put/Call Curve by Expiry (Current Snapshot)"
            zeroLine={false}
            yFormatter={v => v.toFixed(2)}
            tooltipFormatter={v => v.toFixed(3)}
          />
          <p className="text-[11px] text-gray-400 mt-1">
            This is a per-expiry curve from today&apos;s options chains.
          </p>
        </div>
      )}

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
  const cached = useMemo(() => loadSurveysCache(), [])
  const { data, isLoading, error } = useApiQuery(
    ["sentiment-surveys"],
    fetchSentimentSurveys,
    60 * 60 * 1000,
  )

  useEffect(() => {
    if (isSurveysPayload(data)) {
      saveSurveysCache(data)
    }
  }, [data])

  const hasLivePayload = isSurveysPayload(data)
  const hasCachedPayload = cached?.payload != null
  const payload: SurveysPayload = hasLivePayload
    ? data
    : cached?.payload ?? { aaii: [], naaim: [] }
  const usingCachedFallback = !hasLivePayload && hasCachedPayload
  const feedErrors = payload.errors && typeof payload.errors === "object"
    ? payload.errors as Record<string, unknown>
    : {}
  const aaiiError = typeof feedErrors.aaii === "string" ? feedErrors.aaii : null
  const naaimError = typeof feedErrors.naaim === "string" ? feedErrors.naaim : null
  const hasSourceError = Boolean(aaiiError || naaimError)

  if (isLoading && !hasLivePayload && !hasCachedPayload) {
    return <LoadingSpinner message="Fetching AAII and NAAIM data..." />
  }

  const aaii: Record<string, unknown>[] = payload.aaii
  const naaim: Record<string, unknown>[] = payload.naaim

  const latestAaii = aaii[aaii.length - 1] ?? {}
  const latestNaaim = naaim[naaim.length - 1] ?? {}

  const spreadSeries = mkSeries(aaii.slice(-52), "date", "spread")
  const naaimSeries = mkSeries(naaim, "date", "exposure")

  return (
    <div className="space-y-8">
      {(usingCachedFallback || error || hasSourceError) && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
          Surveys data is partially unavailable.
          {usingCachedFallback
            ? ` Showing cached data${cached?.savedAt ? ` (last updated ${formatSavedAt(cached.savedAt)}).` : "."}`
            : hasSourceError
              ? ` ${[
                  aaiiError ? "AAII feed failed" : null,
                  naaimError ? "NAAIM feed failed" : null,
                ].filter(Boolean).join("; ")}.`
              : " Data will appear when the endpoint recovers."}
        </div>
      )}

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

            <p className="mb-1 text-xs font-medium text-muted">Bull / Bear / Neutral (%)</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={aaii.slice(-52) as Record<string, unknown>[]}
                margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                barSize={8}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(d: string) => new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                  tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
                  tickLine={false}
                  axisLine={{ stroke: "hsl(var(--chart-grid))" }}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                  tick={{ fontSize: 10, fill: "hsl(var(--chart-axis))" }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--chart-tooltip-bg))",
                    borderColor: "hsl(var(--chart-tooltip-border))",
                    borderRadius: "0.75rem",
                    color: "hsl(var(--foreground))",
                  }}
                  labelFormatter={(l: unknown) => new Date(String(l)).toLocaleDateString()}
                  formatter={(v: unknown) => `${Number(v).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }} />
                <Bar dataKey="bull" stackId="a" fill="#10b981" name="bull" />
                <Bar dataKey="neutral" stackId="a" fill="#9ca3af" name="neutral" />
                <Bar dataKey="bear" stackId="a" fill="#ef4444" name="bear" />
              </BarChart>
            </ResponsiveContainer>

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
          <p className="text-sm text-gray-400">
            {aaiiError ? "AAII feed is currently unavailable." : "No AAII data available."}
          </p>
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
          <p className="text-sm text-gray-400">
            {naaimError ? "NAAIM feed is currently unavailable." : "No NAAIM data available."}
          </p>
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
  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } = useSessionAiOverview("ai-overview:sentiment")
  const [prepError, setPrepError] = useState<string | null>(null)
  const [isPreparingOverview, setIsPreparingOverview] = useState(false)
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: analyzeSentiment,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })
  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError || isPreparingOverview || prepError)

  // Register screen context for agent chat
  const screenCtx = useMemo(() => ({
    pageName: "Sentiment",
    metrics: { "Active Tab": tab },
    filters: { tab },
    summary: `Sentiment dashboard, viewing ${tab} tab`,
    correspondingTools: ["get_sentiment"],
  }), [tab])
  useRegisterScreenContext(screenCtx)

  async function handleAnalyzeClick() {
    setIsOpen(true)
    setPrepError(null)
    setIsPreparingOverview(true)

    try {
      const [putCall, surveys, volatility] = await Promise.all([
        queryClient.fetchQuery({ queryKey: ["sentiment-put-call"], queryFn: () => fetchSentimentPutCall(180), staleTime: 5 * 60 * 1000 }),
        queryClient.fetchQuery({ queryKey: ["sentiment-surveys"], queryFn: fetchSentimentSurveys, staleTime: 60 * 60 * 1000 }),
        queryClient.fetchQuery({ queryKey: ["sentiment-volatility"], queryFn: () => fetchSentimentVolatility(365), staleTime: 5 * 60 * 1000 }),
      ])

      mutation.mutate({
        put_call: putCall ?? {},
        surveys: surveys ?? {},
        volatility: Array.isArray(volatility) ? volatility : [],
      })
    } catch (err) {
      setPrepError(String(err))
    } finally {
      setIsPreparingOverview(false)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Sentiment</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Options market, investor surveys, and volatility signals
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleAnalyzeClick}
            disabled={mutation.isPending || isPreparingOverview}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton />
        </div>
      </div>

      {showPanel && (
        <div className="mb-6 rounded-xl border border-blue-200 bg-white overflow-hidden">
          <button
            onClick={() => setIsOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 bg-blue-50 hover:bg-blue-100 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-blue-500" />
              <span className="text-sm font-semibold text-blue-700">AI Overview</span>
            </div>
            <ChevronDown
              size={16}
              className={`text-blue-500 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            />
          </button>

          {isOpen && (
            <div className="px-4 py-4">
              {(isPreparingOverview || mutation.isPending) && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  {isPreparingOverview ? "Loading sentiment datasets..." : "Analyzing sentiment data..."}
                </div>
              )}
              {prepError && <p className="text-sm text-red-600">{prepError}</p>}
              {mutation.isError && (
                <p className="text-sm text-red-600">
                  {String(mutation.error) || "Analysis failed. Please try again."}
                </p>
              )}
              {analysisText && (
                <p className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                  {analysisText}
                </p>
              )}
            </div>
          )}
        </div>
      )}

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
