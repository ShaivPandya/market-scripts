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
} satisfies JsonValue

const approvalSummary = {
  count: 1,
  items: [smokeApproval],
  recommendation_approval_count: 0,
  has_more: false,
  status: "pending",
  ticker: null,
  application_status: null,
  limit: 50,
} satisfies JsonValue

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

const workspaceResponse = {
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
  source_health: null,
  thesis_pressure: [
    {
      ticker: "MSFT",
      status: "watching",
      action: "watch",
      confidence: "medium",
      risk_flag: "AI capex concentration",
      evaluated_at: "2026-05-14T14:00:00Z",
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

const ontologyResult = {
  run_id: "ontology-smoke-run",
  as_of: "2026-05-14T14:00:00Z",
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
  },
} satisfies JsonValue

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
  if (method === "GET" && path === "/api/workspace") return json(route, workspaceResponse)
  if (method === "GET" && path === "/api/approvals/summary") return json(route, approvalSummary)
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
    return json(route, {
      job_id: "ontology-smoke-job",
      status: "done",
      result: ontologyResult,
    })
  }
  if (method === "GET" && path === "/api/ontology/query/async/ontology-smoke-job") {
    return json(route, {
      job_id: "ontology-smoke-job",
      status: "done",
      result: ontologyResult,
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
      }
      await page.route("**/api/**", route => handleApiRoute(route, state))
      await use(state)
      expect(state.unknownRequests, "Unexpected /api requests").toEqual([])
    },
    { auto: true },
  ],
})

export { expect }
