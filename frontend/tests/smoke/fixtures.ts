import { expect, test as base, type Page, type Route } from "@playwright/test"

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

interface ApiMockState {
  unknownRequests: string[]
  agentHistoryTitle: string
  approvalsDismissed: boolean
  dismissedPressureKeys: Set<string>
  ontologyQueryRequest: Record<string, JsonValue> | null
}

const smokeApproval = {
  id: "smoke-approval",
  status: "pending",
  entity_type: "action_item",
  action_id: "create_action_item",
  ticker: "MSFT",
  reason: "Follow up on AI infrastructure risk after the weekly portfolio review.",
  created_at: "2026-05-14T14:00:00Z",
  application_status: "pending",
  proposed_change: {
    ticker: "MSFT",
    description: "Review MSFT AI infrastructure concentration before adding exposure.",
    action_type: "research",
    urgency: "high",
  },
  approval_requirements: [
    {
      id: "research_lead",
      label: "Research lead",
      min_count: 1,
      actor_roles: ["research_lead"],
      actor_ids: [],
      scope_type: "ticker",
      scope_id: "MSFT",
      allow_requester: false,
      allow_actor_reuse: false,
      approved_count: 1,
      remaining_count: 0,
      satisfied: true,
    },
    {
      id: "portfolio_manager",
      label: "Portfolio manager",
      min_count: 1,
      actor_roles: ["portfolio_manager"],
      actor_ids: [],
      scope_type: "portfolio",
      scope_id: "default",
      allow_requester: false,
      allow_actor_reuse: false,
      approved_count: 0,
      remaining_count: 1,
      satisfied: false,
    },
  ],
  approval_decisions: [
    {
      requirement_id: "research_lead",
      actor_id: "research@example.com",
      actor_type: "user",
      actor_roles: ["research_lead"],
      decision: "approved",
      note: "Research lead reviewed.",
      decided_at: "2026-05-14T14:05:00Z",
    },
  ],
  approval_progress: {
    total_required: 2,
    recorded_count: 1,
    remaining_count: 1,
    completed: false,
    requirements: [
      {
        id: "research_lead",
        label: "Research lead",
        min_count: 1,
        actor_roles: ["research_lead"],
        actor_ids: [],
        scope_type: "ticker",
        scope_id: "MSFT",
        allow_requester: false,
        allow_actor_reuse: false,
        approved_count: 1,
        remaining_count: 0,
        satisfied: true,
      },
      {
        id: "portfolio_manager",
        label: "Portfolio manager",
        min_count: 1,
        actor_roles: ["portfolio_manager"],
        actor_ids: [],
        scope_type: "portfolio",
        scope_id: "default",
        allow_requester: false,
        allow_actor_reuse: false,
        approved_count: 0,
        remaining_count: 1,
        satisfied: false,
      },
    ],
    remaining_requirements: [
      {
        id: "portfolio_manager",
        label: "Portfolio manager",
        min_count: 1,
        actor_roles: ["portfolio_manager"],
        actor_ids: [],
        scope_type: "portfolio",
        scope_id: "default",
        allow_requester: false,
        allow_actor_reuse: false,
        approved_count: 0,
        remaining_count: 1,
        satisfied: false,
      },
    ],
  },
  remaining_approval_requirements: [
    {
      id: "portfolio_manager",
      label: "Portfolio manager",
      min_count: 1,
      actor_roles: ["portfolio_manager"],
      actor_ids: [],
      scope_type: "portfolio",
      scope_id: "default",
      allow_requester: false,
      allow_actor_reuse: false,
      approved_count: 0,
      remaining_count: 1,
      satisfied: false,
    },
  ],
  decision_state: "pending_approval",
  effect_scope: "internal_state",
  policy_state: "pass",
  quality_state: "ok",
  base_state_status: "valid",
  base_state_valid: true,
  base_state_message: null,
  can_approve: true,
  can_reject: true,
  can_retry_apply: false,
  can_restage: false,
  review_route: "/workspace",
  source_health_review: {
    status: "warning",
    blockers: [],
    warnings: [
      {
        id: "signal_aggregator:current:v1",
        source_name: "market_regime",
        domain: "market",
        status: "stale",
        quality_state: "stale",
        required: true,
        reliability_tier: "standard",
        sla_breach: true,
        gate_action: "warn",
        reason: "standard source needs review",
        freshness_timestamp: "2026-05-11",
      },
    ],
    generated_at: "2026-05-14T14:10:00Z",
  },
} satisfies JsonValue

function approvalSummaryResponse(state: ApiMockState) {
  return {
    count: state.approvalsDismissed ? 0 : 1,
    items: state.approvalsDismissed ? [] : [smokeApproval],
    recommendation_approval_count: 0,
    has_more: false,
    status: "pending",
    ticker: null,
    application_status: null,
    limit: 50,
  } satisfies JsonValue
}

const portfolioSeries = [
  { date: "2026-05-10T14:30:00Z", value: 100 },
  { date: "2026-05-11T14:30:00Z", value: 102 },
  { date: "2026-05-12T14:30:00Z", value: 101 },
  { date: "2026-05-13T14:30:00Z", value: 105 },
  { date: "2026-05-14T14:30:00Z", value: 106 },
]

function portfolioResponse(timeframe: string) {
  const nvdaOffset = timeframe === "Monthly" ? 12 : timeframe === "Weekly" ? 8 : 4

  return {
    holdings: [
      { ticker: "MSFT", role: "position" },
      { ticker: "NVDA", role: "position" },
      { ticker: "SPY", role: "hedge" },
    ],
    position_order: ["MSFT", "NVDA", "SPY"],
    positions: {
      MSFT: portfolioSeries,
      NVDA: portfolioSeries.map(point => ({ ...point, value: Number(point.value) + nvdaOffset })),
      SPY: portfolioSeries.map(point => ({ ...point, value: Number(point.value) - 2 })),
    },
    warning: null,
    group_exposures: [
      {
        group_name: "AI Infrastructure",
        group_conviction: 4,
        direction: "long",
        tickers: ["MSFT", "NVDA"],
        current_notional: 250000,
      },
    ],
  } satisfies JsonValue
}

const smokePressureKey = "MSFT:pressure-smoke"

const baseWorkspaceResponse = {
  regime: {
    regime: "Risk-on",
    composite_score: 0.72,
    signal: "bullish",
    snapshot: {
      as_of: "2026-05-14",
      stale: false,
      refresh_status: "ok",
      error: null,
    },
  },
  portfolio: {
    position_count: 2,
    total_pnl: 8400,
    total_pnl_pct: 0.034,
    risk: {
      result_id: "risk-smoke",
      as_of: "2026-05-14",
      computed_at: "2026-05-14T14:10:00Z",
      quality: "ok",
      confidence: 0.91,
      average_risk_score: 0.42,
      max_risk_score: 0.68,
      risk_level: "medium",
      risk_buckets: { high: 1, medium: 1, low: 0 },
      top_contributors: [
        { ticker: "MSFT", risk_score: 0.68 },
        { ticker: "NVDA", risk_score: 0.51 },
      ],
    },
  },
  source_health: {
    generated_at: "2026-05-14T14:10:00Z",
    overall_quality: "stale",
    counts: {
      total: 3,
      ok: 2,
      stale: 1,
      degraded: 0,
      failed: 0,
      missing: 0,
      required_stale: 1,
      required_failed: 0,
      optional_degraded: 0,
      critical_stale: 1,
      critical_failed: 0,
      sla_breach: 1,
    },
    tier_counts: {
      ad_hoc: 0,
      critical: 2,
      standard: 1,
      supplemental: 0,
    },
    domains: [
      {
        domain: "market",
        label: "Market",
        overall_quality: "stale",
        counts: {
          total: 2,
          ok: 1,
          stale: 1,
          degraded: 0,
          failed: 0,
          missing: 0,
          required_stale: 1,
          required_failed: 0,
          optional_degraded: 0,
          critical_stale: 1,
          critical_failed: 0,
          sla_breach: 1,
        },
        sources: [
          {
            id: "market_breadth:sp500:1y",
            domain: "market",
            source_name: "market_breadth",
            snapshot_key: "market_breadth:sp500:1y",
            status: "stale",
            quality_state: "stale",
            required: true,
            reliability_tier: "critical",
            sla_seconds: 129600,
            sla_breach: true,
            gate_action: "block",
            as_of: "2026-05-11",
            fetched_at: "2026-05-11T14:10:00Z",
            freshness_timestamp: "2026-05-11",
            stale: true,
            detail: "snapshot is stale",
          },
          {
            id: "signal_aggregator:current:v1",
            domain: "market",
            source_name: "market_regime",
            snapshot_key: "signal_aggregator:current:v1",
            status: "ok",
            quality_state: "ok",
            required: true,
            reliability_tier: "standard",
            sla_seconds: 129600,
            sla_breach: false,
            gate_action: "ok",
            as_of: "2026-05-14",
            fetched_at: "2026-05-14T14:10:00Z",
            freshness_timestamp: "2026-05-14",
            stale: false,
          },
        ],
      },
      {
        domain: "portfolio",
        label: "Portfolio",
        overall_quality: "ok",
        counts: {
          total: 1,
          ok: 1,
          stale: 0,
          degraded: 0,
          failed: 0,
          missing: 0,
          required_stale: 0,
          required_failed: 0,
          optional_degraded: 0,
          critical_stale: 0,
          critical_failed: 0,
          sla_breach: 0,
        },
        sources: [
          {
            id: "portfolio",
            domain: "portfolio",
            source_name: "portfolio",
            status: "ok",
            quality_state: "ok",
            required: true,
            reliability_tier: "critical",
            sla_seconds: 129600,
            sla_breach: false,
            gate_action: "ok",
            as_of: "2026-05-14",
            fetched_at: "2026-05-14T14:10:00Z",
            freshness_timestamp: "2026-05-14",
            stale: false,
          },
        ],
      },
    ],
  },
  thesis_pressure: [
    {
      ticker: "MSFT",
      status: "watching",
      action: "watch",
      confidence: "medium",
      risk_flag: "AI capex concentration",
      evaluated_at: "2026-05-14T14:00:00Z",
      pressure_key: smokePressureKey,
    },
  ],
  pending_approvals: {
    count: 1,
    items: [smokeApproval],
  },
  recommendations: {
    latest_daily: null,
    latest_weekly: null,
    pending_actionable: {
      count: 0,
      items: [],
    },
    blocked_warnings: [],
    pending_approval_count: 0,
  },
  open_actions: {
    count: 0,
    items: [],
  },
  active_triggers: {
    count: 0,
    items: [],
  },
  monitor_hits: {
    count: 0,
    items: [],
  },
  recent_workflow_runs: [
    {
      run_id: "workflow-smoke",
      workflow_name: "weekly_portfolio_review",
      status: "completed",
      started_at: "2026-05-14T13:00:00Z",
      completed_at: "2026-05-14T13:02:00Z",
      summary: { result: "No immediate rebalance required." },
    },
  ],
} satisfies JsonValue

function workspaceResponse(state: ApiMockState) {
  return {
    ...(baseWorkspaceResponse as Record<string, JsonValue>),
    thesis_pressure: state.dismissedPressureKeys.has(smokePressureKey)
      ? []
      : (baseWorkspaceResponse as { thesis_pressure: JsonValue[] }).thesis_pressure,
    pending_approvals: {
      count: state.approvalsDismissed ? 0 : 1,
      items: state.approvalsDismissed ? [] : [smokeApproval],
    },
  } satisfies JsonValue
}

function ontologyResultForRequest(body: Record<string, JsonValue> | null = null) {
  const asOf = typeof body?.as_of === "string" && body.as_of ? body.as_of : "2026-05-14T14:00:00Z"
  const txAsOf = typeof body?.tx_as_of === "string" && body.tx_as_of ? body.tx_as_of : null
  const includeHistory = body?.include_history === true

  return {
    run_id: "ontology-smoke-run",
    as_of: asOf,
    aggregate: {
      confidence: 0.88,
      position_count: 2,
      risk_buckets: { high: 1, medium: 1, low: 0 },
    },
    results: [
      {
        ticker: "MSFT",
        asset: "equity",
        direction: "long",
        sector: "Information Technology",
        risk_score: 0.68,
        risk_level: "high",
        evidence: [
          {
            source: "macro",
            name: "Liquidity impulse",
            contribution: 0.42,
          },
        ],
      },
      {
        ticker: "NVDA",
        asset: "equity",
        direction: "long",
        sector: "Semiconductors",
        risk_score: 0.51,
        risk_level: "medium",
        evidence: [
          {
            source: "positioning",
            name: "Crowding",
            contribution: 0.31,
          },
        ],
      },
    ],
    source_status: {
      portfolio: { status: "ok", detail: "smoke fixture" },
      macro: { status: "ok", detail: "smoke fixture" },
    },
    _meta: {
      pagination: {
        page: 1,
        page_size: 25,
        total_results: 2,
        returned_results: 2,
        total_pages: 1,
        has_prev: false,
        has_next: false,
      },
      temporal: {
        as_of: asOf,
        tx_as_of: txAsOf,
        include_history: includeHistory,
        mode: "temporal_read_model",
      },
    },
  } satisfies JsonValue
}

const liquidityResponse = {
  composite_score: -0.03,
  regime: "normal",
  regime_color: "cyan",
  latest_date: "2026-05-13",
  regional_scores: {
    us: { score: 0.05, regime: "normal", color: "cyan" },
    europe: { score: -0.37, regime: "normal", color: "cyan" },
    japan: { score: 0.47, regime: "normal", color: "cyan" },
  },
  components: [
    {
      region: "US",
      key: "net_liquidity_change_4w",
      label: "Net Liquidity (4w change)",
      value: -67920,
      value_kind: "billions",
      z_score: -0.48,
      weight: 0.25,
      contribution: -0.12,
      polarity: 1,
    },
  ],
  changes: {
    "Net Liquidity": {
      value_kind: "billions",
      polarity: 1,
      "1w": 56088,
      "1m": -67920,
      "3m": 120000,
    },
  },
  component_as_of: {
    net_liquidity_change_4w: "2026-05-13",
    jpn_m3_yoy: "2025-11-01",
  },
  data_quality: {
    status: "degraded",
    warnings: [
      "Suppressed partial weekly bucket ending 2026-05-20; using latest completed week 2026-05-13.",
      "M3 YoY is lagged: as of 2025-11-01 (193d old, limit 120d).",
    ],
  },
  _meta: {
    snapshot: {
      key: "liquidity:current:v1",
      as_of: "2026-05-13",
      stale: false,
      refresh_status: "ok",
    },
  },
} satisfies JsonValue

const responsePreferences = {
  personality: "pragmatic",
  warmth: "less",
  enthusiasm: "less",
  headers_lists: "less",
  emoji: "less",
  fast_answers: true,
  thinking_enabled: false,
  custom_instructions: "",
} satisfies JsonValue

const dossierResponse = {
  ticker: "MSFT",
  position: { ticker: "MSFT", asset: "equity", instrument_type: "security", direction: "long" },
  overview_content: null,
  overview_parsed: null,
  management_quality: { content: null, parsed: null },
  what_changed: { counts: { total: 0 }, items: [] },
  evidence_ledger: {
    ticker: "MSFT",
    generated_at: "2026-05-31T00:00:00Z",
    claims: [
      {
        claim_id: "thesis_claim:MSFT:ai-capex",
        claim: "AI capex remains durable",
        status: "active",
        expected_evidence_text: "Azure growth re-accelerated",
        disconfirming_evidence_text: null,
        supporting_evidence: [
          {
            evidence: {
              evidence_id: "ev-1",
              title: "Azure earnings",
              summary: "Azure growth re-accelerated in latest quarter",
              source_record_id: "report:weekly:2026-05-10",
              observed_at: "2026-05-10T00:00:00Z",
            },
            citations: [{ citation_id: "cit-1", title: "Weekly report", url: "https://example.com/report" }],
            source_record: {
              source_record_id: "report:weekly:2026-05-10",
              source_name: "weekly_report_sync",
              vendor: "github_actions",
              quality: "ok",
              as_of: "2026-05-10T00:00:00Z",
            },
          },
        ],
        disconfirming_evidence: [],
      },
    ],
    recommendations: [],
    counts: { claims: 1, recommendations: 0, evidence_items: 1 },
  },
  thesis: { meta: { ticker: "MSFT", status: "active", last_evaluated: null }, content: null, status_history: [] },
  evaluations: [],
  thesis_claims: [],
  catalysts: [],
  kill_conditions: [],
  ontology_risk: null,
  workflow_runs: [],
  action_items: [],
  watch_triggers: [],
  pending_approvals: [],
} satisfies JsonValue

function agentHistorySessions(state: ApiMockState) {
  const now = new Date().toISOString()
  return [
    {
      session_id: "history-session-nvda",
      started_at: now,
      ended_at: now,
      message_count: 4,
      key_tickers: ["NVDA"],
      key_topics: ["earnings prep", "AI capex"],
      summary: "Prepared an NVDA earnings discussion focused on AI infrastructure demand and margin risk.",
      title: state.agentHistoryTitle,
      title_source: "generated",
      title_updated_at: now,
    },
  ] satisfies JsonValue
}

function json(route: Route, body: JsonValue, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  })
}

async function handleApiRoute(route: Route, state: ApiMockState) {
  const request = route.request()
  const url = new URL(request.url())
  const path = url.pathname
  const method = request.method()

  if (method === "POST" && path === "/api/auth/logout") return json(route, { status: "ok" })
  if (method === "POST" && path === "/api/auth/login") return json(route, { status: "ok" })
  if (method === "GET" && path === "/api/auth/me") {
    return json(route, { email: "smoke@example.com", authenticated: true })
  }

  if (method === "GET" && path === "/api/portfolio") {
    return json(route, portfolioResponse(url.searchParams.get("timeframe") ?? "This Week"))
  }

  if (method === "GET" && path === "/api/liquidity") return json(route, liquidityResponse)
  if (method === "GET" && path === "/api/dossier/MSFT") return json(route, dossierResponse)
  if (method === "GET" && path === "/api/thesis/status") return json(route, { MSFT: "uploaded" })
  if (method === "GET" && path === "/api/valuation/MSFT") {
    return json(route, { ticker: "MSFT", status: "unavailable", valuation: null })
  }
  if (method === "GET" && path === "/api/workspace") return json(route, workspaceResponse(state))
  if (method === "POST" && path === "/api/workspace/thesis-pressure/dismiss") {
    const body = JSON.parse(request.postData() || "{}") as { ticker?: string; pressure_key?: string }
    if (body.pressure_key) state.dismissedPressureKeys.add(body.pressure_key)
    return json(route, {
      status: "dismissed",
      ticker: body.ticker ?? "MSFT",
      pressure_key: body.pressure_key ?? smokePressureKey,
    })
  }
  if (method === "GET" && path === "/api/approvals") {
    return json(route, {
      approvals: state.approvalsDismissed ? [] : [smokeApproval],
      count: state.approvalsDismissed ? 0 : 1,
    })
  }
  if (method === "GET" && path === "/api/approvals/summary") return json(route, approvalSummaryResponse(state))
  if (method === "POST" && path === "/api/approvals/bulk-reject") {
    const body = JSON.parse(request.postData() || "{}") as { ids?: string[] }
    state.approvalsDismissed = true
    return json(route, {
      results: (body.ids ?? []).map(id => ({ id, status: "rejected" })),
    })
  }
  if (method === "POST" && path === "/api/approvals/smoke-approval/approve") {
    return json(route, {
      ...(smokeApproval as Record<string, JsonValue>),
      status: "approved",
      application_status: "applied",
    })
  }

  if (method === "GET" && path === "/api/ontology/runs") {
    return json(route, {
      runs: [
        {
          run_id: "ontology-smoke-run",
          as_of: "2026-05-14T14:00:00Z",
          created_at: "2026-05-14T14:05:00Z",
          required_modules_ok: true,
        },
      ],
    })
  }
  if (method === "POST" && path === "/api/ontology/query/async") {
    state.ontologyQueryRequest = JSON.parse(request.postData() || "{}") as Record<string, JsonValue>
    return json(route, {
      job_id: "ontology-smoke-job",
      status: "done",
      result: ontologyResultForRequest(state.ontologyQueryRequest),
    })
  }
  if (method === "GET" && path === "/api/ontology/query/async/ontology-smoke-job") {
    return json(route, {
      job_id: "ontology-smoke-job",
      status: "done",
      result: ontologyResultForRequest(state.ontologyQueryRequest),
    })
  }

  if (method === "GET" && path === "/api/agent/workflows") {
    return json(route, [
      {
        name: "weekly_portfolio_review",
        label: "Weekly Portfolio Review",
        description: "Review portfolio risk and staged actions.",
        requires_ticker: false,
      },
      {
        name: "position_risk_review",
        label: "Position Risk Review",
        description: "Review a single position.",
        requires_ticker: true,
      },
    ])
  }
  if (method === "GET" && path === "/api/settings/agent-response-preferences") {
    return json(route, responsePreferences)
  }
  if (method === "PUT" && path === "/api/settings/agent-response-preferences") {
    return json(route, responsePreferences)
  }
  if (method === "GET" && path === "/api/memory/sessions") return json(route, agentHistorySessions(state))
  if (method === "GET" && path === "/api/memory/sessions/history-session-nvda") {
    const session = (agentHistorySessions(state) as JsonValue[])[0] as Record<string, JsonValue>
    return json(route, {
      ...session,
      transcript: [
        {
          id: "user-1",
          role: "user",
          content: "Prep NVDA earnings",
          timestamp: Date.now() - 1000,
        },
        {
          id: "assistant-1",
          role: "assistant",
          content: "NVDA earnings prep should focus on AI infrastructure demand.",
          timestamp: Date.now(),
        },
      ],
    })
  }
  if (method === "PATCH" && path === "/api/memory/sessions/history-session-nvda") {
    const body = JSON.parse(request.postData() || "{}") as { title?: string }
    state.agentHistoryTitle = String(body.title || state.agentHistoryTitle).trim()
    const session = (agentHistorySessions(state) as JsonValue[])[0] as Record<string, JsonValue>
    return json(route, { ...session, title_source: "manual" })
  }

  const signature = `${method} ${path}${url.search}`
  state.unknownRequests.push(signature)
  return json(route, { detail: `Unmocked API request in smoke test: ${signature}` }, 599)
}

export async function authenticate(page: Page) {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("auth_session", "1")
  })
}

export const test = base.extend<{ apiMocks: ApiMockState }>({
  apiMocks: [
    async ({ page }, use) => {
      const state: ApiMockState = {
        unknownRequests: [],
        agentHistoryTitle: "NVDA Earnings Prep",
        approvalsDismissed: false,
        dismissedPressureKeys: new Set(),
        ontologyQueryRequest: null,
      }
      await page.route("**/api/**", route => handleApiRoute(route, state))
      await use(state)
      expect(state.unknownRequests, "Unexpected /api requests").toEqual([])
    },
    { auto: true },
  ],
})

export { expect }
