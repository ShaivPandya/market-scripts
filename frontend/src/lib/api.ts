import axios from "axios"

import { getAuthMode } from "@/lib/authMode"
import type { DecisionState, DecisionStateFields, EffectScope } from "@/lib/decisionState"
import type { ParsedManagementQuality } from "@/lib/managementQualityTypes"
import type { ParsedOverview } from "@/lib/overviewTypes"

const client = axios.create({
  baseURL: (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/+$/, ""),
  withCredentials: true,
  // Avoid "spinning forever" when the backend (or an upstream like Cloudflare) hangs.
  timeout: 60_000,
})

function schemaHeaders(method: string, url: string): Record<string, string> {
  const base = new URL(client.defaults.baseURL ?? "/api/v1", window.location.origin)
  const parsed = url.startsWith("http")
    ? new URL(url)
    : new URL(`${base.pathname.replace(/\/+$/, "")}/${url.replace(/^\/+/, "")}`, window.location.origin)
  return {
    "X-Request-Schema-Name": `${method.toLowerCase()}:${parsed.pathname}`,
    "X-Request-Schema-Version": "1",
  }
}

client.interceptors.request.use(config => {
  const method = (config.method ?? "get").toLowerCase()
  if (!["post", "put", "patch", "delete"].includes(method) || config.data == null || !config.url) return config
  config.headers.set(schemaHeaders(method, config.url))
  return config
})

function _truncate(s: string, maxLen: number) {
  if (s.length <= maxLen) return s
  return s.slice(0, maxLen - 1) + "…"
}

function _extractDetail(data: unknown): unknown {
  if (data == null) return undefined
  if (typeof data === "string") return data
  if (typeof data !== "object") return undefined
  const rec = data as Record<string, unknown>
  if ("detail" in rec) return rec.detail
  if ("message" in rec) return rec.message
  if ("error" in rec) return rec.error
  return undefined
}

function formatApiError(err: unknown): string | null {
  if (!axios.isAxiosError(err)) return null

  const isTimeout =
    err.code === "ECONNABORTED" ||
    (typeof err.message === "string" && err.message.toLowerCase().includes("timeout"))

  const status = err.response?.status
  const data = err.response?.data
  const detail = _extractDetail(data)

  const prefix =
    status === 424 ? "Dependency error: " : status ? `${status}: ` : isTimeout ? "Timeout: " : ""

  if (typeof detail === "string" && detail.trim()) return prefix + _truncate(detail.trim(), 500)

  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>
    const msg = typeof d.message === "string" ? d.message.trim() : ""
    const failed = Array.isArray(d.failed) ? d.failed.filter(x => typeof x === "string") as string[] : []
    if (msg) {
      const extra = failed.length ? ` (failed: ${_truncate(failed.join(", "), 220)})` : ""
      return prefix + _truncate(msg + extra, 500)
    }
    try {
      return prefix + _truncate(JSON.stringify(detail), 500)
    } catch {
      // fall through
    }
  }

  if (typeof data === "string" && data.trim()) return prefix + _truncate(data.trim(), 500)
  if (err.message && err.message.trim()) return prefix + _truncate(err.message.trim(), 500)
  return status ? `${status}: Request failed` : "Request failed"
}

export interface StagedMutationOptions {
  reason?: string
  apply?: boolean
  approval_note?: string
}

export interface StagedMutationResponse {
  status: "pending_approval_created" | "applied" | string
  approval_id: number
  application_status: "pending" | "applying" | "applied" | "failed" | "not_applicable" | string
  action_id: string
  entity_type: string
  ticker: string | null
  proposed_change: Record<string, unknown>
  decision_state?: DecisionState | string
  decision_kind?: string
  effect_scope?: EffectScope | string
  execution_capability?: string
  review_route?: string | null
  summary?: {
    reason?: string | null
    risk_class?: string | null
    approval_mode?: string | null
    approval_note_required?: boolean
  }
}

export interface PolicyGateReason {
  code?: string
  check?: string
  message?: string
  observed?: unknown
  limit?: unknown
}

export interface PolicyGateResult {
  decision?: string
  review_required?: boolean
  failure_reasons?: PolicyGateReason[]
  warnings?: PolicyGateReason[]
  disclosures?: string[]
}

export interface ApprovalRecord extends DecisionStateFields {
  id: number
  status?: string | null
  entity_type: string
  action_id?: string | null
  ticker: string | null
  reason: string | null
  created_at: string
  application_status?: string | null
  application_error?: string | null
  application_attempts?: number | null
  base_state_status?: "valid" | "stale" | "untracked" | "unknown" | string | null
  base_state_valid?: boolean | null
  base_state_message?: string | null
  source_type?: string | null
  source_id?: string | null
  proposed_change: Record<string, unknown>
  policy_gate?: PolicyGateResult | null
  can_approve?: boolean
  can_reject?: boolean
  can_retry_apply?: boolean
  can_restage?: boolean
  review_route?: string | null
}

export interface RejectAndRestageResponse {
  status: "replacement_created" | string
  original: ApprovalRecord
  replacement: ApprovalRecord
}

export interface ApprovalSummaryResponse {
  count: number
  items: ApprovalRecord[]
  recommendation_approval_count: number
  has_more: boolean
  status: string | null
  ticker: string | null
  application_status: string | null
  limit: number
}

export interface ApprovalSummaryParams {
  status?: string
  ticker?: string
  application_status?: string
  limit?: number
}

export interface RecommendationRecord extends DecisionStateFields {
  id: number
  report_type: string
  as_of: string
  stance: string
  recommendation_status: string
  critical_data_quality: string
  action: string
  ticker: string | null
  instrument: string
  rationale: string
  confidence: number | null
  approval_status: string
  blocked_reasons_json?: string[]
  policy_gate?: PolicyGateResult | null
  policy_gate_decision?: string | null
  policy_gate_review_required?: boolean | number | null
  policy_gate_failures_json?: PolicyGateReason[]
  policy_gate_warnings_json?: PolicyGateReason[]
  policy_gate_disclosures_json?: string[]
  risk_snapshot_id?: string | null
  portfolio_risk_snapshot_id?: string | null
  risk_quality?: string | null
  risk_confidence?: number | null
  risk_score?: number | null
  risk_level?: string | null
  risk_source_status?: Record<string, unknown> | null
  risk_bindings?: Record<string, unknown> | null
}

export type IdeaStatus = "watching" | "researching" | "ready_for_review" | "accepted" | "rejected" | "archived"
export type IdeaAction = "buy" | "watch" | "avoid" | "do_nothing"

export interface InvestmentIdea {
  id: number
  ticker: string
  company_name: string | null
  status: IdeaStatus | string
  user_notes: string
  tags: string[]
  tags_json?: string[] | string
  created_at: string
  updated_at: string
  source_type?: string | null
  source_id?: string | null
  latest_evaluation_id: number | null
  latest_job_id: string | null
  accepted_recommendation_id: number | null
  metadata?: Record<string, unknown>
  latest_evaluation?: IdeaEvaluation | null
}

export interface IdeaFactorScore {
  score?: number
  status?: string
  rationale?: string
  missing?: string[]
}

export interface IdeaMissingInformation {
  field: string
  severity: "critical" | "high" | "medium" | "low" | string
  reason?: string
}

export interface IdeaEvidenceItem {
  source?: string
  url?: string
  summary?: string
  [key: string]: unknown
}

export interface IdeaEvaluation {
  id: number
  idea_id: number
  ticker: string
  job_id: string | null
  evaluated_at: string
  action: IdeaAction | string
  recommendation_status: string
  score: number | null
  confidence: number | null
  thesis_statement: string | null
  rationale: string
  factor_scores: Record<string, IdeaFactorScore>
  missing_information: IdeaMissingInformation[]
  data_quality: Record<string, unknown>
  evidence: IdeaEvidenceItem[]
  disconfirming_evidence: IdeaEvidenceItem[]
  catalyst: string | null
  invalidation: string | null
  portfolio_fit: Record<string, unknown>
  recommendation_record?: Partial<RecommendationRecord> & Record<string, unknown>
  recommendation_id: number | null
  approval_id: number | null
  action_approval_id: number | null
  accepted_at: string | null
  accepted_by: string | null
  created_at: string
}

export interface IdeaDetailResponse {
  idea: InvestmentIdea
  evaluations: IdeaEvaluation[]
  documents?: {
    overview_present?: boolean
    overview_content?: string | null
    overview_parsed?: ParsedOverview | null
    overview_error?: string | null
    thesis_present?: boolean
    thesis_content?: string | null
    thesis_error?: string | null
    management_quality_present?: boolean
    management_quality_content?: string | null
    management_quality_parsed?: ParsedManagementQuality | null
    management_quality_error?: string | null
  }
}

export interface IdeaListResponse {
  ideas: InvestmentIdea[]
  count: number
}

export interface IdeaEvaluationJobResult {
  idea: InvestmentIdea
  evaluation: IdeaEvaluation
  result?: Record<string, unknown>
}

export type IdeaEvaluationJobResponse =
  | { job_id: string; status: "queued" | "running"; timeout_s?: number; progress?: Record<string, unknown> }
  | { job_id: string; status: "done"; timeout_s?: number; result?: IdeaEvaluationJobResult; progress?: Record<string, unknown> }
  | { job_id: string; status: "error" | "cancelled"; timeout_s?: number; error?: string; progress?: Record<string, unknown> }

export interface IdeaAcceptResponse {
  status: string
  idea: InvestmentIdea
  evaluation: IdeaEvaluation
  recommendation: RecommendationRecord
  action_proposal: StagedMutationResponse | null
  action_error?: string | null
}

export interface IdeaComparisonRanking {
  id: number
  run_id: string
  idea_id: number
  evaluation_id: number
  ticker: string
  rank: number
  action: IdeaAction | string
  score: number | null
  confidence: number | null
  confidence_level: "high" | "medium" | "low" | string
  rationale: string
  created_at: string
}

export interface IdeaComparisonRun {
  id: number
  run_id: string
  job_id: string | null
  scope_statuses: string[]
  summary: string
  ranking_count: number
  raw_result?: Record<string, unknown>
  created_at: string
  rankings: IdeaComparisonRanking[]
}

export interface IdeaComparisonRunListResponse {
  runs: IdeaComparisonRun[]
  count: number
}

export interface IdeaComparisonJobResult {
  run: IdeaComparisonRun
  rankings: IdeaComparisonRanking[]
  evaluations?: IdeaEvaluation[]
}

export type IdeaComparisonJobResponse =
  | { job_id: string; status: "queued" | "running"; timeout_s?: number; progress?: Record<string, unknown> }
  | { job_id: string; status: "done"; timeout_s?: number; result?: IdeaComparisonJobResult; progress?: Record<string, unknown> }
  | { job_id: string; status: "error" | "cancelled"; timeout_s?: number; error?: string; progress?: Record<string, unknown> }

client.interceptors.response.use(
  res => res,
  err => {
    if (
      getAuthMode() !== "cloudflare" &&
      err.response?.status === 401 &&
      !window.location.pathname.startsWith("/login")
    ) {
      window.location.href = "/login"
    }
    const msg = formatApiError(err)
    if (msg) err.message = msg
    return Promise.reject(err)
  },
)

export const authApi = {
  login: (password: string) => client.post("/auth/login", { password }).then(r => r.data),
  logout: () => client.post("/auth/logout").then(r => r.data),
  me: () => client.get("/auth/me").then(r => r.data),
}

export type LLMProvider = "anthropic" | "openai"
export type LLMModelTier = "low" | "mid" | "high"
export type LLMReasoningEffort = "none" | "medium" | "high" | "xhigh" | "max"
export type LLMModelTierMap = Record<LLMModelTier, string>
export type LLMReasoningEffortMap = Record<LLMModelTier, LLMReasoningEffort>

export interface LLMProviderStatus {
  provider: LLMProvider
  label: string
  configured: boolean
  api_key_env: string
}

export interface LLMReasoningEffortOption {
  effort: LLMReasoningEffort
  label: string
}

export interface LLMSettings {
  provider: LLMProvider
  available_providers: LLMProviderStatus[]
  models: LLMModelTierMap
  models_by_provider: Record<LLMProvider, LLMModelTierMap>
  reasoning_efforts: Record<LLMProvider, LLMReasoningEffortMap>
  reasoning_options: Record<LLMProvider, Record<LLMModelTier, LLMReasoningEffortOption[]>>
}

export const fetchLLMSettings = () =>
  client.get("/settings/llm").then(r => r.data as LLMSettings)

export const updateLLMSettings = (settings: {
  provider: LLMProvider
  reasoning_efforts?: LLMReasoningEffortMap
}) =>
  client.put("/settings/llm", settings).then(r => r.data as LLMSettings)

export type AgentPreferenceLevel = "less" | "balanced" | "more"
export type AgentPersonality = "friendly" | "pragmatic"

export interface AgentResponsePreferences {
  personality: AgentPersonality
  warmth: AgentPreferenceLevel
  enthusiasm: AgentPreferenceLevel
  headers_lists: AgentPreferenceLevel
  emoji: AgentPreferenceLevel
  fast_answers: boolean
  thinking_enabled: boolean
  custom_instructions?: string | null
}

export const fetchAgentResponsePreferences = () =>
  client
    .get("/settings/agent-response-preferences")
    .then(r => r.data as AgentResponsePreferences)

export const updateAgentResponsePreferences = (preferences: AgentResponsePreferences) =>
  client
    .put("/settings/agent-response-preferences", preferences)
    .then(r => r.data as AgentResponsePreferences)

export interface AgentToolGovernanceMetadata {
  required_scopes: string[]
  account_scope: string | null
  portfolio_scope: string | null
  data_sensitivity: string
  provider_egress: string
  timeout_s: number
  retry_policy: Record<string, unknown>
  token_budget: number | null
  cost_budget_usd: number | null
  rate_limit: Record<string, unknown>
  audit_level: string
  failure_mode: string
}

export interface AgentCapability {
  name: string
  category: string
  access_mode: "read" | "compute" | "proposal" | string
  description: string
  aliases: string[]
  schema_safe: boolean
  selectable: boolean
  governance: AgentToolGovernanceMetadata
}

export interface AgentCapabilitiesResponse {
  capabilities: AgentCapability[]
  count: number
}

export const fetchAgentCapabilities = () =>
  client.get("/agent/capabilities").then(r => r.data as AgentCapabilitiesResponse)

// ─── GET endpoints ───────────────────────────────────────────────────────────

export const fetchPortfolio = (timeframe: string) =>
  client.get(`/portfolio?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchPortfolioAllTimeframes = () =>
  client.get("/portfolio?all_timeframes=true").then(r => r.data)

export type InstrumentType = "security" | "future"
export type PortfolioAsset = "equity" | "commodity" | "fx" | "bond"

export interface PortfolioPosition {
  ticker: string
  asset: PortfolioAsset
  direction: "long" | "short"
  contrarian: boolean
  conviction: number
  cost_basis: number | null
  shares: number | null
  quantity?: number | null
  instrument_type?: InstrumentType | null
  price_symbol?: string | null
  contract_multiplier?: number | null
  currency?: string | null
  country?: string | null
  exchange?: string | null
  base_currency?: string | null
  fx_rate_to_base?: number | null
  fx_rate_as_of?: string | null
  cost_basis_base?: number | null
  notional_base?: number | null
  valuation_status?: string | null
  role?: "position" | "hedge"
}

export const fetchPortfolioPositions = (includeHedges = false) =>
  client
    .get("/portfolio-positions", { params: includeHedges ? { include_hedges: true } : undefined })
    .then(r => r.data as { positions: PortfolioPosition[] })

export const savePortfolioPositions = (positions: PortfolioPosition[], options?: StagedMutationOptions) =>
  client.put("/portfolio-positions", { positions, ...options }).then(r => r.data as StagedMutationResponse)

export interface HedgePosition {
  ticker: string
  asset?: PortfolioAsset | null
  direction: "long" | "short"
  cost_basis: number | null
  shares: number | null
  quantity?: number | null
  instrument_type?: InstrumentType | null
  price_symbol?: string | null
  contract_multiplier?: number | null
  currency?: string | null
  country?: string | null
  exchange?: string | null
  base_currency?: string | null
  fx_rate_to_base?: number | null
  fx_rate_as_of?: string | null
  cost_basis_base?: number | null
  notional_base?: number | null
  valuation_status?: string | null
}

export const fetchHedgePositions = () =>
  client.get("/hedge-positions").then(r => r.data as { positions: HedgePosition[] })

export const saveHedgePositions = (positions: HedgePosition[], options?: StagedMutationOptions) =>
  client.put("/hedge-positions", { positions, ...options }).then(r => r.data as StagedMutationResponse)

export type ThesisStatus = "populated" | "empty" | "missing"

export const fetchThesisStatus = () =>
  client.get("/thesis/status").then(r => r.data as Record<string, ThesisStatus>)

export const fetchThesis = (ticker: string) =>
  client
    .get(`/thesis/${encodeURIComponent(ticker)}`)
    .then(r => r.data as { status: "ok"; ticker: string; content: string })

export const saveThesisContent = (ticker: string, content: string, options?: StagedMutationOptions) =>
  client
    .put(`/thesis/${encodeURIComponent(ticker)}`, { content, ...options })
    .then(r => r.data as StagedMutationResponse)

export const uploadThesisDocument = (ticker: string, file: File) => {
  const formData = new FormData()
  formData.append("ticker", ticker)
  formData.append("file", file)
  return client
    .post("/thesis/generate", formData, { timeout: 120_000 })
    .then(r => r.data as StagedMutationResponse)
}

// --- Overview ---

export const saveOverviewContent = (ticker: string, content: string) =>
  client
    .put(`/overview/${encodeURIComponent(ticker)}`, { content })
    .then(r => r.data as { status: "ok"; ticker: string; content: string })

export const uploadOverviewDocument = (ticker: string, file: File) => {
  const formData = new FormData()
  formData.append("ticker", ticker)
  formData.append("file", file)
  return client
    .post("/overview/generate", formData, { timeout: 120_000 })
    .then(r => r.data as { status: "ok"; ticker: string; content: string })
}

// --- Management Quality ---

export const saveManagementQualityContent = (ticker: string, content: string, options?: StagedMutationOptions) =>
  client
    .put(`/management-quality/${encodeURIComponent(ticker)}`, { content, ...options })
    .then(r => r.data as StagedMutationResponse)

export const uploadManagementQualityDocument = (ticker: string, file: File) => {
  const formData = new FormData()
  formData.append("ticker", ticker)
  formData.append("file", file)
  return client
    .post("/management-quality/generate", formData, { timeout: 120_000 })
    .then(r => r.data as StagedMutationResponse)
}

export const fetchManagementQuality = (ticker: string) =>
  client
    .get(`/management-quality/${encodeURIComponent(ticker)}`)
    .then(r => r.data as { status: "ok"; ticker: string; content: string; parsed: ParsedManagementQuality | null })

// --- Thesis metadata types ---

export type ThesisStatusValue = "active" | "under_review" | "invalidated" | "missing"

export interface ThesisEvaluation {
  id: number
  ticker: string
  evaluated_at: string
  thesis_status: string
  technical_read: string
  fundamental_read: string
  action: string
  confidence: string
  key_developments: string[]
  earnings_note: string | null
  risk_flag: string | null
}

export interface ThesisMeta {
  ticker: string
  status: ThesisStatusValue
  created_at: string | null
  updated_at: string | null
  direction?: "long" | "short" | string | null
  asset?: string | null
  conviction?: number | string | null
  last_evaluated?: string | null
  latest_evaluation?: ThesisEvaluation | null
}

export interface ThesisDetail {
  meta: ThesisMeta
  content: string | null
  status_history: Array<{
    id: number
    ticker: string
    old_status: string | null
    new_status: string
    reason: string | null
    changed_at: string
  }>
  evaluations: ThesisEvaluation[]
}

export const fetchThesisMeta = () =>
  client.get("/thesis/meta").then(r => r.data as ThesisMeta[])

export const fetchThesisDetail = (ticker: string) =>
  client
    .get(`/thesis/${encodeURIComponent(ticker)}/detail`)
    .then(r => r.data as ThesisDetail)

export const updateThesisStatus = (
  ticker: string,
  status: ThesisStatusValue,
  reason: string,
  options?: StagedMutationOptions,
) =>
  client
    .put(`/thesis/${encodeURIComponent(ticker)}/status`, { status, reason, ...options })
    .then(r => r.data as StagedMutationResponse)

export const fetchMomentum = () =>
  client.get("/momentum").then(r => r.data)

export const fetchSignalAggregator = (params?: {
  lookback_weeks?: number
  positioning_instruments?: string
  include_raw_modules?: boolean
  force_refresh?: boolean
}) => client.get("/signal-aggregator", { params, timeout: 180_000 }).then(r => r.data)

export const fetchIndexDashboard = (timeframe: string) =>
  client.get(`/index-dashboard?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchFxDashboard = (timeframe: string) =>
  client.get(`/fx-dashboard?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchCommodities = (timeframe: string) =>
  client.get(`/commodities?timeframe=${encodeURIComponent(timeframe)}`).then(r => r.data)

export const fetchMarketBreadth = () =>
  client.get("/market-breadth").then(r => r.data)

export const fetchTop50Breadth = () =>
  client.get("/top50-breadth").then(r => r.data)

export const fetchPriceVolumeSignals = () =>
  client.get("/price-volume-signals").then(r => r.data)

export const fetchVixTermStructure = () =>
  client.get("/vix-term-structure").then(r => r.data)

export type OntologyQueryBody = {
  query?: string
  intent?:
    | "portfolio_risk_exposure"
    | "positions_in_deteriorating_macro"
    | "entity_context"
    | "thesis_review"
    | "temporal_comparison"
  filters?: {
    tickers?: string[]
    sectors?: string[]
    assets?: string[]
    min_risk_score?: number
  }
  timeframe?: "This Week" | "Daily" | "Weekly" | "Monthly"
  include_graph?: boolean
  run_id?: string
  as_of?: string
  tx_as_of?: string
  include_history?: boolean
  refresh_snapshot?: boolean
  page?: number
  page_size?: number
  schema_mode?: "stored" | "upgraded"
}

export interface OntologyEvidence {
  component?: string
  source?: string
  name?: string
  value?: number | string | null
  threshold?: string
  direction?: string
  contribution?: number | null
}

export interface OntologyRow {
  ticker: string
  asset?: string
  direction?: string
  sector?: string
  risk_score?: number | null
  risk_level?: string
  evidence?: OntologyEvidence[]
  _meta?: {
    temporal?: {
      object_uid?: string
      version_id?: string
      valid_from?: string
      valid_to?: string | null
      tx_from?: string
      tx_to?: string | null
      temporal_confidence?: string
    }
  }
}

export interface OntologySourceStatus {
  status?: string
  detail?: string
}

export interface PositionRiskSourceStatus {
  status?: string
  quality?: string
  detail?: string
  error?: string
  required?: boolean
  accepted?: boolean
  used?: boolean
  fallback_used?: boolean
  refreshed?: boolean
  snapshot_key?: string
  snapshot_status?: string
  payload_hash?: string
  as_of?: string | null
  fetched_at?: string | null
  freshness?: {
    policy?: string
    fresh?: boolean
    basis?: string
    expected_market_date?: string
    observed_as_of_date?: string | null
    reason?: string | null
  }
  [key: string]: unknown
}

export interface PositionRiskEvidence {
  component?: string
  source?: string
  name?: string
  value?: number | string | null
  threshold?: string
  direction?: string
  contribution?: number | null
}

export interface PositionRiskDegradedModule {
  module: string
  required?: boolean
  status?: string
  quality?: string
  reason?: string
  as_of?: string | null
  fetched_at?: string | null
}

export interface PositionRiskSnapshot {
  result_id: string
  run_id?: string
  ticker: string
  as_of?: string | null
  computed_at: string
  market_snapshot_as_of?: string | null
  freshness_policy?: string
  risk_score?: number | null
  risk_level?: string
  confidence?: number | null
  quality?: string
  asset?: string
  direction?: string
  sector?: string
  component_scores?: {
    volatility_cluster?: number
    breadth_stress?: number
    sector_stress?: number
    macro_regime?: number
  }
  evidence?: PositionRiskEvidence[]
  drivers?: PositionRiskEvidence[]
  degraded_modules?: PositionRiskDegradedModule[]
  missing_modules?: string[]
  stale_modules?: string[]
  source_status?: Record<string, PositionRiskSourceStatus>
  input_snapshots?: Record<string, unknown>
  position?: Record<string, unknown>
  aggregate?: {
    confidence?: number
    position_count?: number
    average_risk_score?: number
    exact?: boolean
    risk_buckets?: {
      high?: number
      medium?: number
      low?: number
    }
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface PortfolioRiskSnapshot extends Omit<PositionRiskSnapshot, "ticker"> {
  position_count?: number
  average_risk_score?: number | null
  max_risk_score?: number | null
  risk_buckets?: {
    high?: number
    medium?: number
    low?: number
  }
  top_contributors?: Array<Record<string, unknown>>
  position_snapshot_ids?: Record<string, string>
  position_snapshots?: PositionRiskSnapshot[]
}

export interface OntologyResponse {
  run_id?: string
  intent?: string
  as_of?: string
  source_status?: Record<string, OntologySourceStatus>
  results?: OntologyRow[]
  aggregate?: {
    confidence?: number
    position_count?: number
    average_risk_score?: number
    exact?: boolean
    risk_buckets?: {
      high?: number
      medium?: number
      low?: number
    }
    [key: string]: unknown
  }
  _meta?: {
    pagination?: {
      page?: number
      page_size?: number
      returned_results?: number
      total_results?: number
      total_pages?: number
      has_prev?: boolean
      has_next?: boolean
      sort?: string
      exact_total?: boolean
    }
    graph?: {
      scope?: string
      node_count?: number
      edge_count?: number
      truncated?: boolean
      max_nodes?: number
      max_edges?: number
    }
    temporal?: {
      as_of?: string | null
      tx_as_of?: string | null
      include_history?: boolean
      mode?: string
    }
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface OntologyRunSummary {
  run_id: string
  as_of: string
  created_at: string
  required_modules_ok: boolean
}

export const fetchOntologyRuns = (limit = 100) =>
  client
    .get("/ontology/runs", { params: { limit } })
    .then(r => r.data as { runs: OntologyRunSummary[] })

export type OntologyTemporalParams = {
  as_of?: string
  tx_as_of?: string
  include_history?: boolean
  limit?: number
  offset?: number
}

export const fetchOntologyObjects = (params: OntologyTemporalParams & {
  object_type?: string
  business_key?: string
  object_uid?: string
} = {}) =>
  client.get("/ontology/objects", { params }).then(r => r.data as { objects: Array<Record<string, unknown>> })

export const fetchOntologyObject = (object_uid: string, params: Pick<OntologyTemporalParams, "as_of" | "tx_as_of"> = {}) =>
  client
    .get(`/ontology/objects/${encodeURIComponent(object_uid)}`, { params })
    .then(r => r.data as Record<string, unknown>)

export const fetchOntologyRelations = (params: OntologyTemporalParams & {
  relation_type?: string
  source_object_uid?: string
  target_object_uid?: string
} = {}) =>
  client.get("/ontology/relations", { params }).then(r => r.data as { relations: Array<Record<string, unknown>> })

export const fetchOntologySourceRecords = (params: OntologyTemporalParams & {
  vendor?: string
  source_name?: string
  record_kind?: string
} = {}) =>
  client
    .get("/ontology/source-records", { params })
    .then(r => r.data as { source_records: Array<Record<string, unknown>> })

export const queryOntology = (body: OntologyQueryBody) =>
  runOntologyQueryAsync(body)

export const fetchPositionRiskLatest = async (ticker: string) => {
  try {
    const r = await client.get(`/risk/positions/${encodeURIComponent(ticker)}/latest`)
    return r.data as PositionRiskSnapshot
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) return null
    throw err
  }
}

export const refreshPositionRisk = (ticker: string) =>
  client
    .post(`/risk/positions/${encodeURIComponent(ticker)}/refresh`, undefined, { timeout: 120_000 })
    .then(r => r.data as PositionRiskSnapshot)

export const fetchPortfolioRiskLatest = async () => {
  try {
    const r = await client.get("/risk/portfolio/latest")
    return r.data as PortfolioRiskSnapshot
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) return null
    throw err
  }
}

export const refreshPortfolioRisk = () =>
  client
    .post("/risk/portfolio/refresh", undefined, { timeout: 180_000 })
    .then(r => r.data as PortfolioRiskSnapshot)

type OntologyJobResponse =
  | { job_id: string; status: "queued" | "running"; timeout_s?: number }
  | { job_id: string; status: "error"; error?: string; timeout_s?: number }
  | { job_id: string; status: "done"; result?: OntologyResponse; timeout_s?: number }

const DEFAULT_ONTOLOGY_JOB_TIMEOUT_MS = 300_000
const ONTOLOGY_JOB_TIMEOUT_BUFFER_MS = 45_000
const ONTOLOGY_JOB_POLL_INTERVAL_MS = 2_000

function ontologyJobDeadline(started: OntologyJobResponse): number {
  const timeoutMs = typeof started.timeout_s === "number" && Number.isFinite(started.timeout_s)
    ? Math.max(0, started.timeout_s * 1000)
    : DEFAULT_ONTOLOGY_JOB_TIMEOUT_MS
  return Date.now() + timeoutMs + ONTOLOGY_JOB_TIMEOUT_BUFFER_MS
}

export const startOntologyQueryJob = (body: OntologyQueryBody) =>
  client
    .post("/ontology/query/async", { ...body, schema_mode: body.schema_mode ?? "upgraded" }, { timeout: 30_000 })
    .then(r => r.data as OntologyJobResponse)

export const fetchOntologyQueryJob = (job_id: string) =>
  client.get(`/ontology/query/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as OntologyJobResponse)

export async function runOntologyQueryAsync(body: OntologyQueryBody, signal?: AbortSignal): Promise<OntologyResponse> {
  const started = await startOntologyQueryJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Ontology query failed")

  const job_id = started.job_id
  const deadline = ontologyJobDeadline(started)

  for (; ;) {
    if (signal?.aborted) throw new DOMException("Query cancelled", "AbortError")
    if (Date.now() > deadline) throw new Error("Timeout: Ontology query is still running. Try again shortly.")

    await new Promise(r => setTimeout(r, ONTOLOGY_JOB_POLL_INTERVAL_MS))
    const job = await fetchOntologyQueryJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      throw new Error("Ontology query returned no result")
    }
    if (job.status === "error") throw new Error(job.error || "Ontology query failed")
  }
}

export const fetchEconomicGrowth = () =>
  client.get("/economic-growth").then(r => r.data)

export const uploadEconomicGrowthCrbFile = (file: File) => {
  const formData = new FormData()
  formData.append("file", file)
  return client
    .post("/economic-growth/crb-file", formData, { timeout: 120_000 })
    .then(r => r.data as {
      status: "ok"
      crb: {
        filename: string
        uploaded_at: string
        rows: number
        latest_date: string
        latest_value: number
        size_bytes: number
      }
    })
}

export const fetchWeeklyReport = () =>
  client.get("/weekly-report", { timeout: 180_000 }).then(r => r.data)

export const fetchWeeklyReportCached = () =>
  client
    .get("/weekly-report", { params: { cached_only: true }, timeout: 30_000 })
    .then(r => r.data)
    .catch(err => {
      if (axios.isAxiosError(err) && err.response?.status === 404) return null
      throw err
    })

export const generateWeeklyReport = () =>
  client.get("/weekly-report", { params: { refresh: true }, timeout: 180_000 }).then(r => r.data)

export const analyzeMarketTechnicals = (body: {
  market_breadth: Record<string, unknown>
  top50_breadth: Record<string, unknown>
  vix_term_structure: Record<string, unknown>
  price_volume_signals: Record<string, unknown>
}) => client.post("/market-technicals/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const analyzeEconomicGrowth = (body: {
  commodities: Record<string, Record<string, number | null>>
  equities: Record<string, Record<string, number | null>>
  currencies: Record<string, Record<string, number | null>>
  equity_periods: string[]
  currency_periods: string[]
}) => client.post("/economic-growth/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchLaborMarket = () =>
  client.get("/labor-market").then(r => r.data)

export const analyzeLaborMarket = (body: {
  latest: Record<string, { value: number | null; date: string | null; change: number | null }>
  series_labels: Record<string, string>
  series_units: Record<string, string>
  timestamp?: string | null
}) => client.post("/labor-market/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchHousing = () =>
  client.get("/housing").then(r => r.data)

export const analyzeHousing = (body: {
  latest: Record<string, { value: number | null; date: string | null; change: number | null }>
  series_labels: Record<string, string>
  series_units: Record<string, string>
  timestamp?: string | null
}) => client.post("/housing/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchLiquidity = () =>
  client.get("/liquidity").then(r => r.data)

export const analyzeLiquidity = (body: {
  composite_score?: number | null
  regime?: string | null
  latest_date?: string | null
  regional_scores?: Record<string, unknown>
  components?: Record<string, unknown>[]
  changes?: Record<string, Record<string, unknown>>
}) => client.post("/liquidity/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchCountryDashboard = (metric: string) =>
  client.get(`/country-dashboard?metric=${encodeURIComponent(metric)}`).then(r => r.data)

export const fetchBreakout = () =>
  client.get("/breakout").then(r => r.data)

export const fetchCentralBanks = (refresh = false) =>
  client.get(`/central-banks?refresh=${refresh}`).then(r => r.data)

export interface NewsDigestStory {
  id: string
  section: string
  headline: string
  notes: string[]
  digest_id?: string
  digest_title?: string
  generated_date?: string
}

export interface NewsDigestSummary {
  id: string
  title: string
  slug: string
  filename: string
  generated_date: string
  uploaded_at: string
  updated_at: string
  content_hash: string
  story_count: number
  section_count: number
  sections: Array<{ name: string; story_count: number }>
}

export interface NewsDigestListResponse {
  items: NewsDigestSummary[]
  stories: NewsDigestStory[]
  counts: { digests: number; stories: number }
}

export interface NewsDigestDetail extends NewsDigestSummary {
  content: string
  parsed: {
    title: string
    slug: string
    generated_date: string
    story_count: number
    section_count: number
    sections: Array<{ name: string; stories: NewsDigestStory[] }>
    stories: NewsDigestStory[]
  }
}

export type NewsDigestUploadResponse =
  | { status: "ok"; digest: NewsDigestDetail }
  | StagedMutationResponse

export type NewsDigestDeleteResponse =
  | { status: "ok"; deleted: boolean; id: string }
  | StagedMutationResponse

export const fetchPortfolioNews = () =>
  client.get("/portfolio-news").then(r => r.data as NewsDigestListResponse)

export const fetchPortfolioNewsDigest = (digestId: string) =>
  client
    .get(`/portfolio-news/${encodeURIComponent(digestId)}`)
    .then(r => r.data as NewsDigestDetail)

export const uploadPortfolioNewsDigest = (file: File) => {
  const formData = new FormData()
  formData.append("file", file)
  return client
    .post("/portfolio-news", formData, { timeout: 120_000 })
    .then(r => r.data as NewsDigestUploadResponse)
}

export const deletePortfolioNewsDigest = (digestId: string) =>
  client
    .delete(`/portfolio-news/${encodeURIComponent(digestId)}`)
    .then(r => r.data as NewsDigestDeleteResponse)

export const fetchSectorMetrics = () =>
  client.get("/sector-metrics").then(r => r.data)

export const analyzeSectorMetrics = (body: {
  rows: Record<string, unknown>[]
  timestamp?: string | null
}) => client.post("/sector-metrics/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchIndustryMonitor = (refresh = false) =>
  client.get(`/industry-monitor?refresh=${refresh}`).then(r => r.data)

export const industryTranscriptPdfUrl = (ticker: string) =>
  `${client.defaults.baseURL}/industry-monitor/transcripts/${encodeURIComponent(ticker)}/pdf`

export const fetchYieldCurve = (lookback_days = 90) =>
  client.get(`/yield-curve?lookback_days=${lookback_days}`).then(r => r.data)

export const fetchBondDashboard = () =>
  client.get("/bond-dashboard").then(r => r.data)

export const fetchCommoditiesCurve = (commodity = "CL", lookback_days = 30) =>
  client
    .get(`/commodities-curve?commodity=${encodeURIComponent(commodity)}&lookback_days=${lookback_days}`)
    .then(r => r.data)

export const fetchCommodityResearch = () =>
  client.get("/commodity-research", { timeout: 120_000 }).then(r => r.data)

export const fetchPositioningSummary = (params: Record<string, string>) =>
  client.get("/positioning/summary", { params }).then(r => r.data)

export const fetchPositioningTimeseries = (params: Record<string, string>) =>
  client.get("/positioning/timeseries", { params }).then(r => r.data)

export const fetchPositioningInstruments = () =>
  client.get("/positioning/instruments").then(r => r.data)

export const analyzePositioning = (body: { rows: Record<string, unknown>[] }) =>
  client.post("/positioning/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const fetchFxModelPairs = () =>
  client.get("/fx-model/pairs").then(r => r.data)

export const fetchHedgingToolPrefill = () =>
  client.get("/hedging-tool/prefill").then(r => r.data)

export const fetchHedgingPortfolioWeights = (book?: number) =>
  client
    .get("/hedging-tool/portfolio-weights", { params: book ? { book } : undefined })
    .then(r => r.data as {
      positions: { ticker: string; weight: number }[]
      metadata: {
        ticker: string
        direction: string
        conviction: number
        shares: number | null
        quantity?: number | null
        instrument_type?: InstrumentType | null
        price_symbol?: string | null
        contract_multiplier?: number | null
        cost_basis: number | null
        weight: number
      }[]
      book: number
      source: string
    })

export const fetchHedgingRecommendations = (body: Record<string, unknown>) =>
  client.post("/hedging-tool/recommend", body, { timeout: 180_000 }).then(r => r.data as { analysis: string })

export const fetchSizerPrefill = () =>
  client.get("/portfolio-sizer/prefill").then(r => r.data)

// ─── POST endpoints ───────────────────────────────────────────────────────────

export const runChart = (body: { ticker: string; lookback: string }) =>
  client.post("/chart", body).then(r => r.data)

export const downloadPriceHistory = (ticker: string) =>
  client
    .get(`/chart/price-history/${encodeURIComponent(ticker)}`, {
      responseType: "blob",
      timeout: 120_000,
    })
    .then(r => r.data as Blob)

export const runPriceRatioChart = (body: {
  symbol_a: string
  symbol_b: string
  method?: string
  start_date?: string
  end_date?: string
}) => client.post("/chart/ratio", body).then(r => r.data)

export type AnalyzerScenarioRequest = {
  preset?: string
  metric_scores?: {
    quality: number
    price_momentum: number
    revenue: number
    eps: number
    price_sales: number
    price_operating_income: number
    price_fcf: number
    price_earnings: number
  }
  factor_weights?: {
    quality: number
    price_momentum: number
    fundamental_momentum: number
    valuation: number
  }
  fundamental_momentum_weights?: {
    revenue: number
    eps: number
  }
  valuation_weights?: {
    price_sales: number
    price_operating_income: number
    price_fcf: number
    price_earnings: number
  }
  brakes?: {
    drawdown_sensitivity: number
    contrarian_penalty: number
    short_squeeze_brake: number
  }
}

export interface AnalyzerFactorBreakdown {
  factor: string
  label: string
  weight: number | null
  value: number | null
  contribution: number | null
  status: "available" | "missing" | "not_applicable" | "disabled" | string
  reason?: string | null
}

export interface AnalyzerCourseAction {
  ticker: string
  asset: string
  direction: string
  action: string
  conviction_band: "none" | "small" | "medium" | "large" | string
  priority_score: number
  scenario_score: number
  score_delta: number
  baseline_score?: number | null
  confidence: number
  gate_status: "pass" | "review" | "watch" | string
  gate_reasons?: string[]
  deterministic_rationale: string
  warnings?: string[]
  data_coverage?: {
    ratio: number
    available: number
    applicable: number
  }
  factor_conflict?: boolean
  factor_breakdown?: AnalyzerFactorBreakdown[]
  sizing_implication?: {
    implication: string
    conviction_band: string
    note?: string
  }
}

export interface AnalyzerCourseOfAction {
  summary?: {
    mission?: string
    as_of?: string
    action_counts?: Record<string, number>
    data_quality_counts?: Record<string, number>
    strongest_opportunities?: { ticker: string; action: string; priority_score: number }[]
    largest_risks?: { ticker: string; action: string; priority_score: number }[]
    analysis_only?: boolean
  }
  action_queue?: AnalyzerCourseAction[]
  factor_breakdown?: Record<string, AnalyzerFactorBreakdown[]>
}

type AnalyzerRequest = {
  book?: number
  target_leverage?: number
  beta_neutral?: boolean
  scenario?: AnalyzerScenarioRequest
}

export const runPortfolioAnalyzer = (body: AnalyzerRequest = {}) =>
  runPortfolioAnalyzerAsync(body)

type AnalyzerJobBase = { job_id: string; timeout_s?: number }

type AnalyzerJobResponse =
  | (AnalyzerJobBase & { status: "queued" | "running" })
  | (AnalyzerJobBase & { status: "error"; error?: string })
  | (AnalyzerJobBase & { status: "done"; result?: unknown })

export const startPortfolioAnalyzerJob = (body: AnalyzerRequest = {}) =>
  client.post("/portfolio-analyzer/async", body, { timeout: 30_000 }).then(r => r.data as AnalyzerJobResponse)

export const fetchPortfolioAnalyzerJob = (job_id: string) =>
  client.get(`/portfolio-analyzer/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as AnalyzerJobResponse)

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isRetryableAnalyzerError(err: unknown) {
  if (!axios.isAxiosError(err)) return false
  if (!err.response) return true
  return [408, 429, 500, 502, 503, 504].includes(err.response.status)
}

async function withAnalyzerRetry<T>(operation: () => Promise<T>, attempts = 3): Promise<T> {
  let lastError: unknown
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation()
    } catch (err) {
      lastError = err
      if (attempt >= attempts || !isRetryableAnalyzerError(err)) throw err
      await sleep(750 * attempt)
    }
  }
  throw lastError
}

export async function runPortfolioAnalyzerAsync(body: AnalyzerRequest = {}) {
  const started = await withAnalyzerRetry(() => startPortfolioAnalyzerJob(body))
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Portfolio analyzer failed")

  const job_id = started.job_id
  const serverTimeoutMs = Number.isFinite(started.timeout_s) ? Math.max(180, Number(started.timeout_s)) * 1000 : 180_000
  const deadline = Date.now() + serverTimeoutMs + 30_000
  let transientPollErrors = 0

  // Poll until completion; each request is short to avoid edge proxy timeouts.
  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Portfolio analyzer is taking too long. Try again.")

    await sleep(2000)
    let job: AnalyzerJobResponse
    try {
      job = await fetchPortfolioAnalyzerJob(job_id)
      transientPollErrors = 0
    } catch (err) {
      if (!isRetryableAnalyzerError(err) || transientPollErrors >= 5) throw err
      transientPollErrors += 1
      continue
    }

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      // cached:* jobs return done without result in the poll endpoint
      // (the initial response already carried the payload).
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Portfolio analyzer failed")
  }
}

// Backward-compatible exports.
export const runPortfolioOptimizer = runPortfolioAnalyzer
export const startPortfolioOptimizerJob = startPortfolioAnalyzerJob
export const fetchPortfolioOptimizerJob = fetchPortfolioAnalyzerJob
export async function runPortfolioOptimizerAsync(body: AnalyzerRequest = {}) {
  return runPortfolioAnalyzerAsync(body)
}

export const generatePortfolioAnalyzerBrief = (action: AnalyzerCourseAction) =>
  client
    .post("/portfolio-analyzer/course-of-action/brief", { action }, { timeout: 120_000 })
    .then(r => r.data as { brief: string })

export const runHedgingTool = (body: { book: number; positions: { ticker: string; weight: number }[] }) =>
  runHedgingToolAsync(body)

type HedgingJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startHedgingToolJob = (body: { book: number; positions: { ticker: string; weight: number }[] }) =>
  client.post("/hedging-tool/async", body, { timeout: 30_000 }).then(r => r.data as HedgingJobResponse)

export const fetchHedgingToolJob = (job_id: string) =>
  client.get(`/hedging-tool/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as HedgingJobResponse)

export async function runHedgingToolAsync(body: { book: number; positions: { ticker: string; weight: number }[] }) {
  const started = await startHedgingToolJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Hedging tool failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Hedging tool is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchHedgingToolJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Hedging tool failed")
  }
}

type SizerJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startSizerJob = (body: {
  book: number
  target_leverage: number
  positions: { ticker: string; conviction: number }[]
}) =>
  client.post("/portfolio-sizer/async", body, { timeout: 30_000 }).then(r => r.data as SizerJobResponse)

export const fetchSizerJob = (job_id: string) =>
  client.get(`/portfolio-sizer/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as SizerJobResponse)

export async function runPortfolioSizerAsync(body: {
  book: number
  target_leverage: number
  positions: { ticker: string; conviction: number }[]
}) {
  const started = await startSizerJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Sizer failed")

  const job_id = started.job_id
  const deadline = Date.now() + 180_000

  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: Sizer is taking too long. Try again.")

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchSizerJob(job_id)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started ? started.result : undefined
    }
    if (job.status === "error") throw new Error(job.error || "Sizer failed")
  }
}

export const runQualityScreen = (body: {
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}) => {
  const controller = new AbortController()
  const timeoutMs = 180_000
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  return client
    .post("/quality-screen", body, { signal: controller.signal, timeout: timeoutMs })
    .then(r => r.data)
    .catch(err => {
      if (axios.isAxiosError(err) && err.code === "ERR_CANCELED") {
        throw new Error("Timeout: Quality screen exceeded 180s. Try a smaller universe or custom tickers.")
      }
      throw err
    })
    .finally(() => clearTimeout(timer))
}

export type ScreenJobProgress = {
  phase?: string
  done?: number
  total?: number
}

type ScreenJobResponse<T> =
  | { job_id: string; status: "queued" | "running"; progress?: ScreenJobProgress }
  | { job_id: string; status: "error"; error?: string; progress?: ScreenJobProgress }
  | { job_id: string; status: "done"; result?: T; progress?: ScreenJobProgress }

export type ScreenResult = {
  results_df?: Record<string, unknown>[]
  failed_tickers?: string[]
  input_count?: number
  scored_count?: number
  benchmark_name?: string
  date?: string | null
  phase1_count?: number
  phase1_pass_count?: number | null
  phase3_pass_count?: number
  final_count?: number
  [key: string]: unknown
}

export type ShortScreenRequest = {
  input_mode: string
  universe: string
  tickers: string
  pb_threshold: number | null
  loss_type: string | null
  check_issuance: boolean
  check_revenue: boolean
  max_revenue_growth: number
  check_eps: boolean
  max_eps_growth: number
  check_52w_positive: boolean
  check_min_drawdown: boolean
  min_drawdown_pct: number
  check_max_drawdown: boolean
  max_drawdown_pct: number
  check_3m_neg_momentum: boolean
  check_2m_neg_rel_momentum: boolean
  rel_momentum_benchmark: string
}

export type LongScreenRequest = {
  input_mode: string
  universe: string
  tickers: string
  pb_threshold: number | null
  profit_type: string | null
  check_issuance: boolean
  check_revenue: boolean
  min_revenue_growth: number
  check_eps: boolean
  min_eps_growth: number
  check_ebit_multiple: boolean
  max_ebit_multiple: number
  check_52w_positive: boolean
  check_min_drawdown: boolean
  min_drawdown_pct: number
  check_max_drawdown: boolean
  max_drawdown_pct: number
  check_3m_pos_momentum: boolean
  check_2m_pos_rel_momentum: boolean
  rel_momentum_benchmark: string
}

export type PriceMomentumRequest = {
  input_mode: string
  universe: string
  tickers: string
  benchmark: string
}

const startShortScreenJob = (body: ShortScreenRequest) =>
  client.post("/short-screen/async", body, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

const fetchShortScreenJob = (job_id: string) =>
  client.get(`/short-screen/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

const startLongScreenJob = (body: LongScreenRequest) =>
  client.post("/long-screen/async", body, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

const fetchLongScreenJob = (job_id: string) =>
  client.get(`/long-screen/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

const startPriceMomentumJob = (body: PriceMomentumRequest) =>
  client.post("/price-momentum/async", body, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

const fetchPriceMomentumJob = (job_id: string) =>
  client.get(`/price-momentum/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as ScreenJobResponse<ScreenResult>)

async function runScreenJob<TBody>(
  body: TBody,
  startJob: (body: TBody) => Promise<ScreenJobResponse<ScreenResult>>,
  fetchJob: (job_id: string) => Promise<ScreenJobResponse<ScreenResult>>,
  label: string,
  onProgress?: (progress: ScreenJobProgress | undefined) => void,
): Promise<ScreenResult> {
  const started = await startJob(body)
  onProgress?.(started.progress)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || `${label} failed`)

  const job_id = started.job_id
  const deadline = Date.now() + 45 * 60_000

  for (; ;) {
    if (Date.now() > deadline) {
      throw new Error(`Timeout: ${label} is still running after 45 minutes. Try a smaller universe or fewer filters.`)
    }

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchJob(job_id)
    onProgress?.(job.progress)

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return {}
    }
    if (job.status === "error") throw new Error(job.error || `${label} failed`)
  }
}

export const runShortScreen = (body: ShortScreenRequest, onProgress?: (progress: ScreenJobProgress | undefined) => void) =>
  runScreenJob(body, startShortScreenJob, fetchShortScreenJob, "Short screen", onProgress)

export const runLongScreen = (body: LongScreenRequest, onProgress?: (progress: ScreenJobProgress | undefined) => void) =>
  runScreenJob(body, startLongScreenJob, fetchLongScreenJob, "Long screen", onProgress)

export const runPriceMomentum = (body: PriceMomentumRequest, onProgress?: (progress: ScreenJobProgress | undefined) => void) =>
  runScreenJob(body, startPriceMomentumJob, fetchPriceMomentumJob, "Price momentum", onProgress)

type FundamentalMomentumRequest = {
  screen_type: string
  universe: string
  tickers: string
  benchmark: string
  input_mode: string
}

type FundamentalMomentumResponse = {
  screen_type?: string
  eps?: { results_df?: Record<string, unknown>[]; [key: string]: unknown }
  rev?: { results_df?: Record<string, unknown>[]; [key: string]: unknown }
  [key: string]: unknown
}

type FundamentalMomentumJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: FundamentalMomentumResponse }

export const startFundamentalMomentumJob = (body: FundamentalMomentumRequest) =>
  client.post("/fundamental-momentum/async", body, { timeout: 30_000 }).then(r => r.data as FundamentalMomentumJobResponse)

export const fetchFundamentalMomentumJob = (job_id: string) =>
  client.get(`/fundamental-momentum/async/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as FundamentalMomentumJobResponse)

export async function runFundamentalMomentumAsync(body: FundamentalMomentumRequest): Promise<FundamentalMomentumResponse> {
  const started = await startFundamentalMomentumJob(body)
  if (started.status === "done" && "result" in started && started.result != null) return started.result
  if (started.status === "error") throw new Error(started.error || "Fundamental momentum failed")

  let job_id = started.job_id
  const deadline = Date.now() + 300_000
  let restartedAfterUnknownJob = false

  for (; ;) {
    if (Date.now() > deadline) {
      throw new Error("Timeout: Fundamental momentum is taking too long. Try a smaller universe or custom tickers.")
    }

    await new Promise(r => setTimeout(r, 2000))
    let job: FundamentalMomentumJobResponse
    try {
      job = await fetchFundamentalMomentumJob(job_id)
    } catch (err) {
      const isUnknownJob =
        axios.isAxiosError(err) &&
        err.response?.status === 404 &&
        typeof err.message === "string" &&
        err.message.includes("Unknown job_id")

      if (isUnknownJob && !restartedAfterUnknownJob) {
        restartedAfterUnknownJob = true
        const restarted = await startFundamentalMomentumJob(body)
        if (restarted.status === "done" && "result" in restarted && restarted.result != null) return restarted.result
        if (restarted.status === "error") throw new Error(restarted.error || "Fundamental momentum failed")
        job_id = restarted.job_id
        continue
      }
      throw err
    }

    if (job.status === "done") {
      if ("result" in job && job.result != null) return job.result
      return "result" in started && started.result != null ? started.result : {}
    }
    if (job.status === "error") throw new Error(job.error || "Fundamental momentum failed")
  }
}

// Keep existing import call-sites unchanged.
export const runFundamentalMomentum = runFundamentalMomentumAsync

export const runFinancials = (body: { ticker: string }) =>
  client.post("/financials", body).then(r => r.data)

// DCF Model
export const fetchDCFHistorical = (ticker: string) =>
  client.get(`/dcf/historical/${encodeURIComponent(ticker)}`).then(r => r.data)

export interface DCFValuationRequest {
  ticker: string
  revenue_growth_rates: number[]
  ebitda_margin: number
  tax_rate: number
  da_pct_revenue: number
  nwc_pct_revenue: number
  capex_pct_revenue: number
  wacc: number
  terminal_growth_rates: { bear: number; base: number; bull: number }
  exit_ev_ebitda: { bear: number; base: number; bull: number }
  exit_ev_revenue: { bear: number; base: number; bull: number }
}

export const runDCFValuation = (body: DCFValuationRequest) =>
  client.post("/dcf/valuation", body).then(r => r.data)

export const runFxModel = (body: {
  pair: string
  bootstrap: number
  skip_bis: boolean
  horizons: string
}) => client.post("/fx-model", body).then(r => r.data)

export const fetchSentimentPutCall = (lookback_days = 180) =>
  client.get(`/sentiment/put-call?lookback_days=${lookback_days}`).then(r => r.data)

export const fetchSentimentSurveys = () =>
  client.get("/sentiment/surveys").then(r => r.data)

export const fetchSentimentVolatility = (lookback_days = 365) =>
  client.get(`/sentiment/volatility?lookback_days=${lookback_days}`).then(r => r.data)

export const analyzeSentiment = (body: {
  put_call: Record<string, unknown>
  surveys: Record<string, unknown>
  volatility: unknown[]
}) => client.post("/sentiment/analyze", body, { timeout: 180_000 }).then(r => r.data)

export const clearCache = () => client.delete("/cache").then(r => r.data)

type AdminJobResponse =
  | { job_id: string; status: "queued" | "running" }
  | { job_id: string; status: "error"; error?: string }
  | { job_id: string; status: "done"; result?: unknown }

export const startMarketSnapshotRefresh = () =>
  client.post("/admin/jobs/enqueue-market-snapshot-refresh", {}, { timeout: 30_000 }).then(r => r.data as AdminJobResponse)

export const fetchAdminJob = (job_id: string) =>
  client.get(`/admin/jobs/${encodeURIComponent(job_id)}`, { timeout: 30_000 }).then(r => r.data as AdminJobResponse)

export async function refreshMarketSnapshots(): Promise<unknown> {
  const started = await startMarketSnapshotRefresh()
  if (started.status === "done" && "result" in started) return started.result
  if (started.status === "error") throw new Error(started.error || "Market snapshot refresh failed")

  const deadline = Date.now() + 15 * 60_000
  const job_id = started.job_id

  for (; ;) {
    if (Date.now() > deadline) {
      throw new Error("Timeout: market snapshot refresh is still running. Try again in a few minutes.")
    }

    await new Promise(r => setTimeout(r, 2000))
    const job = await fetchAdminJob(job_id)
    if (job.status === "done") return "result" in job ? job.result : undefined
    if (job.status === "error") throw new Error(job.error || "Market snapshot refresh failed")
  }
}

// ---------------------------------------------------------------------------
// Investing OS APIs
// ---------------------------------------------------------------------------

// Workspace
export const fetchWorkspace = () => client.get("/workspace").then(r => r.data)

// Continuous Optimization
export interface OptimizationMission {
  id: number
  name: string
  status: string
  schedule_label?: string | null
  scenario?: Record<string, unknown>
  source_config?: Record<string, unknown>
  thresholds?: Record<string, unknown>
  created_at?: string
  updated_at?: string
}

export interface OptimizationRun {
  run_id: string
  mission_id: number
  mission_name: string
  status: string
  started_at: string
  completed_at?: string | null
  summary?: Record<string, unknown>
  source_freshness?: Record<string, unknown>
  error?: string | null
  snapshots?: OptimizationSnapshot[]
}

export interface OptimizationSnapshot {
  id: number
  run_id: string
  mission_id: number
  ticker?: string | null
  asset?: string | null
  direction?: string | null
  action: string
  conviction_band?: string | null
  priority_score?: number | null
  confidence?: number | null
  gate_status?: string | null
  severity?: string | null
  state_hash?: string
  evidence?: Record<string, unknown>
  source_links?: Record<string, unknown>
  created_at?: string
}

export interface OptimizationAlert {
  id: number
  mission_id: number
  run_id: string
  ticker?: string | null
  alert_type: string
  severity: "low" | "normal" | "high" | "urgent" | string
  status: string
  change_summary: string
  approval_id?: number | null
  recommendation_id?: number | null
  action_item_approval_id?: number | null
  evidence?: Record<string, unknown>
  previous_snapshot?: OptimizationSnapshot | null
  current_snapshot?: OptimizationSnapshot | null
  created_at: string
}

export const fetchOptimizationMissions = () =>
  client.get("/optimization/missions").then(r => r.data as { missions: OptimizationMission[]; count: number })

export const runOptimizationMission = (missionId: number, body?: { source?: string; force?: boolean }) =>
  client.post(`/optimization/missions/${missionId}/run`, body ?? { source: "manual" }).then(r => r.data as AdminJobResponse)

export async function runOptimizationMissionAsync(missionId: number, body?: { source?: string; force?: boolean }) {
  const started = await runOptimizationMission(missionId, body)
  if (started.status === "done" && "result" in started) return started.result
  if (started.status === "error") throw new Error(started.error || "Continuous optimizer failed")

  const deadline = Date.now() + 20 * 60_000
  for (; ;) {
    if (Date.now() > deadline) throw new Error("Timeout: continuous optimizer is still running. Try again in a few minutes.")
    await sleep(2000)
    const job = await fetchAdminJob(started.job_id)
    if (job.status === "done") return "result" in job ? job.result : undefined
    if (job.status === "error") throw new Error(job.error || "Continuous optimizer failed")
  }
}

export const fetchOptimizationRuns = (params?: { mission_id?: number; limit?: number }) =>
  client.get("/optimization/runs", { params }).then(r => r.data as { runs: OptimizationRun[]; count: number })

export const fetchOptimizationRun = (runId: string) =>
  client.get(`/optimization/runs/${encodeURIComponent(runId)}`).then(r => r.data as OptimizationRun)

export const fetchOptimizationAlerts = (params?: { status?: string; mission_id?: number; limit?: number }) =>
  client.get("/optimization/alerts", { params }).then(r => r.data as { alerts: OptimizationAlert[]; count: number })

export const dismissOptimizationAlert = (id: number, note?: string) =>
  client.put(`/optimization/alerts/${id}/dismiss`, note ? { note } : {}).then(r => r.data as OptimizationAlert)

// Idea Watchlist
export const fetchIdeas = (params?: { status?: IdeaStatus | string; include_archived?: boolean; limit?: number }) =>
  client.get("/ideas", { params }).then(r => r.data as IdeaListResponse)

export const fetchIdea = (id: number) =>
  client.get(`/ideas/${id}`).then(r => r.data as IdeaDetailResponse)

export const createIdea = (body: {
  ticker: string
  company_name?: string | null
  user_notes?: string | null
  tags?: string[]
  status?: IdeaStatus
}) => client.post("/ideas", body).then(r => r.data as IdeaDetailResponse)

export const updateIdea = (id: number, body: {
  ticker?: string
  company_name?: string | null
  user_notes?: string | null
  tags?: string[]
  status?: IdeaStatus
}) => client.put(`/ideas/${id}`, body).then(r => r.data as IdeaDetailResponse)

export const archiveIdea = (id: number) =>
  client.delete(`/ideas/${id}`).then(r => r.data as { status: string; idea: InvestmentIdea })

export const startIdeaEvaluationJob = (id: number, body?: { force_refresh?: boolean }) =>
  client
    .post(`/ideas/${id}/evaluate/async`, body ?? {}, { timeout: 30_000 })
    .then(r => r.data as IdeaEvaluationJobResponse)

export const fetchIdeaEvaluationJob = (jobId: string) =>
  client
    .get(`/ideas/evaluate/async/${encodeURIComponent(jobId)}`, { timeout: 30_000 })
    .then(r => r.data as IdeaEvaluationJobResponse)

export const startIdeaComparisonEvaluationJob = () =>
  client
    .post("/ideas/evaluate-all/async", {}, { timeout: 30_000 })
    .then(r => r.data as IdeaComparisonJobResponse)

export const fetchIdeaComparisonEvaluationJob = (jobId: string) =>
  client
    .get(`/ideas/evaluate-all/async/${encodeURIComponent(jobId)}`, { timeout: 30_000 })
    .then(r => r.data as IdeaComparisonJobResponse)

export const fetchIdeaComparisonRuns = (params?: { limit?: number }) =>
  client.get("/ideas/comparison-runs", { params }).then(r => r.data as IdeaComparisonRunListResponse)

export const fetchIdeaComparisonRun = (runId: string) =>
  client.get(`/ideas/comparison-runs/${encodeURIComponent(runId)}`).then(r => r.data as IdeaComparisonRun)

export const acceptIdeaEvaluation = (ideaId: number, evaluationId: number, body?: { note?: string }) =>
  client
    .post(`/ideas/${ideaId}/evaluations/${evaluationId}/accept`, body ?? {})
    .then(r => r.data as IdeaAcceptResponse)

export const rejectIdea = (id: number, note?: string) =>
  client.post(`/ideas/${id}/reject`, note ? { note } : {}).then(r => r.data as { status: string; idea: InvestmentIdea })

// Recommendations
export const fetchRecommendations = (params?: {
  report_type?: string
  status?: string
  ticker?: string
  approval_status?: string
  outcome_status?: string
  limit?: number
}) => client.get("/recommendations", { params }).then(r => r.data)
export const fetchLatestRecommendations = () =>
  client.get("/recommendations/latest").then(r => r.data)
export const fetchRecommendation = (id: number) =>
  client.get(`/recommendations/${id}`).then(r => r.data)

// Dossier
export const fetchDossier = (ticker: string) =>
  client.get(`/dossier/${encodeURIComponent(ticker)}`).then(r => r.data)

// Approvals
export const fetchApprovals = (status?: string) =>
  client.get("/approvals", { params: status ? { status } : undefined }).then(r => r.data)
export const fetchApprovalSummary = (params?: ApprovalSummaryParams) =>
  client.get("/approvals/summary", { params }).then(r => r.data as ApprovalSummaryResponse)
export const approveItem = (id: number, note: string) =>
  client.post(`/approvals/${id}/approve`, { note }).then(r => r.data as ApprovalRecord)
export const rejectItem = (id: number, note?: string) =>
  client.post(`/approvals/${id}/reject`, note ? { note } : {}).then(r => r.data as ApprovalRecord)
export const rejectAndRestageApproval = (id: number, note?: string) =>
  client
    .post(`/approvals/${id}/reject-and-restage`, note ? { note } : {})
    .then(r => r.data as RejectAndRestageResponse)
export const bulkApprove = (ids: number[], note: string) =>
  client.post("/approvals/bulk-approve", { ids, note }).then(r => r.data)
export const bulkReject = (ids: number[], note?: string) =>
  client.post("/approvals/bulk-reject", { ids, note }).then(r => r.data)

// Action Items
export const fetchActions = (params?: { status?: string; ticker?: string }) =>
  client.get("/actions", { params }).then(r => r.data)
export const createAction = (body: { description: string; action_type?: string; ticker?: string; urgency?: string } & StagedMutationOptions) =>
  client.post("/actions", body).then(r => r.data as StagedMutationResponse)
export const completeAction = (id: number, resolution_note?: string, options?: StagedMutationOptions) =>
  client.put(`/actions/${id}/complete`, { resolution_note: resolution_note ?? "", ...options }).then(r => r.data as StagedMutationResponse)
export const dismissAction = (id: number, options?: StagedMutationOptions) =>
  client.put(`/actions/${id}/dismiss`, options ?? {}).then(r => r.data as StagedMutationResponse)

// Watch Triggers
export const fetchTriggers = (params?: { status?: string; ticker?: string }) =>
  client.get("/triggers", { params }).then(r => r.data)
export const createTrigger = (body: { condition: string; trigger_type?: string; ticker?: string; expires_at?: string; definition?: Record<string, unknown> } & StagedMutationOptions) =>
  client.post("/triggers", body).then(r => r.data as StagedMutationResponse)
export const fireTrigger = (id: number, options?: StagedMutationOptions) =>
  client.put(`/triggers/${id}/fire`, options ?? {}).then(r => r.data as StagedMutationResponse)
export const cancelTrigger = (id: number, options?: StagedMutationOptions) =>
  client.put(`/triggers/${id}/cancel`, options ?? {}).then(r => r.data as StagedMutationResponse)

// Catalysts
export const fetchCatalysts = (ticker: string) =>
  client.get("/catalysts", { params: { ticker } }).then(r => r.data)
export const createCatalyst = (body: { ticker: string; description: string; category?: string; target_date?: string } & StagedMutationOptions) =>
  client.post("/catalysts", body).then(r => r.data as StagedMutationResponse)
export const updateCatalystStatus = (id: number, status: string, evidence?: string, options?: StagedMutationOptions) =>
  client.put(`/catalysts/${id}/status`, { status, evidence, ...options }).then(r => r.data as StagedMutationResponse)

// Thesis Claims
export interface SourceRequirement {
  type: string
  description: string
  required: boolean
  freshness_days: number | null
}

export type ThesisClaimStatus = "active" | "supported" | "challenged" | "disconfirmed" | "retired"

export interface ThesisClaim {
  id: number
  ticker: string
  claim: string
  expected_evidence: string | null
  disconfirming_evidence: string | null
  source_requirements: SourceRequirement[]
  source_requirements_json?: SourceRequirement[]
  cadence: string | null
  confidence: number | null
  status: ThesisClaimStatus
  linked_catalyst_ids: number[]
  linked_catalyst_ids_json?: number[]
  linked_kill_condition_ids: number[]
  linked_kill_condition_ids_json?: number[]
  source_type: string
  source_id: string | null
  created_at: string
  updated_at: string
}

export type ThesisClaimPayload = {
  ticker?: string
  claim?: string
  expected_evidence?: string | null
  disconfirming_evidence?: string | null
  source_requirements?: SourceRequirement[]
  cadence?: string | null
  confidence?: number | null
  status?: ThesisClaimStatus
  linked_catalyst_ids?: number[]
  linked_kill_condition_ids?: number[]
}

export const createThesisClaim = (body: ThesisClaimPayload & { ticker: string; claim: string } & StagedMutationOptions) =>
  client.post("/thesis-claims", body).then(r => r.data as StagedMutationResponse)
export const updateThesisClaim = (id: number, body: ThesisClaimPayload & StagedMutationOptions) =>
  client.put(`/thesis-claims/${id}`, body).then(r => r.data as StagedMutationResponse)

// Kill Conditions
export const fetchKillConditions = (ticker: string) =>
  client.get("/kill-conditions", { params: { ticker } }).then(r => r.data)
export const createKillCondition = (body: { ticker: string; condition: string; metric?: string; threshold?: string } & StagedMutationOptions) =>
  client.post("/kill-conditions", body).then(r => r.data as StagedMutationResponse)
export const updateKillConditionStatus = (id: number, status: string, options?: StagedMutationOptions) =>
  client.put(`/kill-conditions/${id}/status`, { status, ...options }).then(r => r.data as StagedMutationResponse)

// Research Notes
export const fetchResearchNotes = (params?: { ticker?: string; limit?: number }) =>
  client.get("/research-notes", { params }).then(r => r.data)
export const createResearchNote = (body: { title: string; content: string; ticker?: string; note_type?: string } & StagedMutationOptions) =>
  client.post("/research-notes", body).then(r => r.data as StagedMutationResponse)

// Workflow Runs
interface AgentWorkflowResponse {
  name: string
  label: string
  description: string
  requires_ticker: boolean
}

export interface AgentWorkflow {
  name: string
  label: string
  description: string
  requiresTicker: boolean
}

export const fetchAgentWorkflows = () =>
  client.get("/agent/workflows").then(r =>
    (r.data as AgentWorkflowResponse[]).map(wf => ({
      name: wf.name,
      label: wf.label,
      description: wf.description,
      requiresTicker: wf.requires_ticker,
    })),
  )
export const fetchWorkflowRuns = (params?: { workflow_name?: string; ticker?: string; limit?: number }) =>
  client.get("/workflow-runs", { params }).then(r => r.data)
export const fetchWorkflowRun = (runId: string) =>
  client.get(`/workflow-runs/${runId}`).then(r => r.data)

export interface ProvenanceSelector {
  workflow_run_id?: string
  ontology_run_id?: string
  approval_id?: number
  action_run_id?: number
  agent_session_id?: string
  event_id?: string
  ref_type?: string
  ref_id?: string
  max_depth?: number
}

export interface ProvenanceEvent {
  id: string
  event_type: string
  event_name?: string | null
  status?: string | null
  started_at?: string | null
  finished_at?: string | null
  actor_type?: string | null
  actor_id?: string | null
  summary?: Record<string, unknown> | null
  metadata?: Record<string, unknown> | null
}

export interface ProvenanceLink {
  id?: string
  event_id?: string | null
  source_ref_type: string
  source_ref_id: string
  target_ref_type: string
  target_ref_id: string
  link_type: string
  created_at?: string | null
}

export interface ProvenanceTrace {
  seed: Record<string, unknown>
  events: ProvenanceEvent[]
  links: ProvenanceLink[]
  source_records: Record<string, unknown>[]
  workflow_artifacts: Record<string, unknown>[]
  timeline?: Record<string, unknown>[]
}

export const fetchProvenanceTrace = (params: ProvenanceSelector) =>
  client.get("/provenance/trace", { params }).then(r => r.data as ProvenanceTrace)

export const fetchEntityProvenance = (refType: string, refId: string, maxDepth = 3) =>
  client
    .get(`/provenance/entity/${encodeURIComponent(refType)}/${encodeURIComponent(refId)}`, {
      params: { max_depth: maxDepth },
    })
    .then(r => r.data as ProvenanceTrace)
