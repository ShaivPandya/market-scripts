import { type FormEvent, useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  ArrowRight,
  Bot,
  CheckCircle2,
  ClipboardCheck,
  Cloud,
  Cpu,
  Database,
  Eye,
  FileText,
  GitBranch,
  ListChecks,
  Lock,
  MessageSquare,
  Monitor,
  Route,
  Shield,
  ShieldCheck,
  Sparkles,
  Target,
  User,
  Wrench,
  Workflow,
  X,
  Zap,
  type LucideIcon,
} from "lucide-react"
import { useAuth } from "@/contexts/AuthContext"
import { SegmentedControl } from "@/components/shared/FormControls"
import { Notice } from "@/components/shared/Notice"

import centralBankImg from "@/assets/central-bank-monitor.jpg"
import commoditiesCurveImg from "@/assets/commodities-curve.png"
import economicGrowthImg from "@/assets/economic-growth.jpg"
import industryImg from "@/assets/industry.jpg"
import laborMarketImg from "@/assets/labor-market.jpg"
import ontologyWorkbenchImg from "@/assets/ontology-workbench.png"
import portfolioImg from "@/assets/portfolio.jpg"
import sectorMetricsImg from "@/assets/sector-metrics.png"
import sentimentImg from "@/assets/sentiment.jpg"

import "./LoginPage.css"

type PublicTab = "product" | "architecture" | "agent"

const publicTabs: { value: PublicTab; label: string }[] = [
  { value: "product", label: "Product" },
  { value: "architecture", label: "Architecture" },
  { value: "agent", label: "Agent" },
]

/* ──────────────────────────── shared building blocks ─────────────────────── */

function Eyebrow({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return <p className={`landing-eyebrow${accent ? " accent" : ""}`}>{children}</p>
}

function IconChip({
  icon: Icon,
  variant,
  size,
}: {
  icon: LucideIcon
  variant?: "accent"
  size?: "sm" | "md"
}) {
  return (
    <span
      className={`landing-icon-chip${size === "sm" ? " sm" : ""}${variant === "accent" ? " accent" : ""}`}
    >
      <Icon className={size === "sm" ? "h-[1.05rem] w-[1.05rem]" : "h-5 w-5"} aria-hidden="true" />
    </span>
  )
}

function SectionHead({
  eyebrow,
  title,
  children,
  maxWidth = "44rem",
}: {
  eyebrow: string
  title: string
  children?: React.ReactNode
  maxWidth?: string
}) {
  return (
    <div className="mx-auto text-center" style={{ maxWidth }}>
      <Eyebrow accent>{eyebrow}</Eyebrow>
      <h2 className="landing-section-title text-pretty mt-2.5">{title}</h2>
      {children && <p className="landing-lead text-pretty mt-3.5">{children}</p>}
    </div>
  )
}

/* ──────────────────────────── architecture data ──────────────────────────── */

interface SystemNode {
  icon: LucideIcon
  step: string
  title: string
  short: string
  detail: string
  stack: string[]
  inflow: string
  outflow: string
}

const SYSTEM_NODES: SystemNode[] = [
  {
    icon: User,
    step: "Entry",
    title: "You",
    short: "A question, a click, or a workflow launch.",
    detail:
      "Everything starts with intent — you open a dashboard, ask Stan a question, or launch a playbook. The visitor sees a clean surface; the request that follows is fully governed underneath.",
    stack: ["Browser", "Authenticated session"],
    inflow: "Your intent",
    outflow: "A routed request",
  },
  {
    icon: Monitor,
    step: "Surface",
    title: "Control room",
    short: "The React app on Firebase.",
    detail:
      "The React app on Firebase gives every workflow a fast, polished operating surface. It registers what you're looking at — page, ticker, visible metrics — so the rest of the system has context.",
    stack: ["React", "Firebase Hosting", "Screen context"],
    inflow: "Your intent",
    outflow: "Request + context",
  },
  {
    icon: Route,
    step: "Router",
    title: "Request brain",
    short: "FastAPI on Cloud Run.",
    detail:
      "FastAPI on Cloud Run receives each question and routes it to the right research capability. Like a triage desk, it decides what kind of request this is before any heavy work begins.",
    stack: ["FastAPI", "Cloud Run", "Intent router"],
    inflow: "Request + context",
    outflow: "Dispatched job",
  },
  {
    icon: Workflow,
    step: "Compute",
    title: "Research engines",
    short: "Python analysis modules.",
    detail:
      "Python modules combine market data, macro signals, portfolio context, filings, and documents. Heavy analysis runs on warm worker pools so the interface never blocks.",
    stack: ["Python", "Async jobs", "Worker pools"],
    inflow: "Dispatched job",
    outflow: "Structured result",
  },
  {
    icon: Database,
    step: "Memory",
    title: "Durable memory",
    short: "Cloud SQL + Cloud Storage.",
    detail:
      "Cloud SQL and Cloud Storage keep state, versions, artifacts, and generated research recoverable. Every important output is traceable back to the evidence that formed it.",
    stack: ["Cloud SQL", "Cloud Storage", "Ontology + provenance"],
    inflow: "Structured result",
    outflow: "Versioned record",
  },
]

interface LoopStep {
  num: string
  icon: LucideIcon
  title: string
  detail: string
  gate?: boolean
}

const LOOP_STEPS: LoopStep[] = [
  {
    num: "01",
    icon: Cloud,
    title: "Ingest",
    detail:
      "Fresh prices, macro releases, transcripts, PDFs, and portfolio changes enter as source records — each one tagged so it can be cited later.",
  },
  {
    num: "02",
    icon: Activity,
    title: "Analyze",
    detail:
      "Dashboards, models, and LLM workflows turn raw inputs into comparisons, risks, and hypotheses. The slow work runs off the main path.",
  },
  {
    num: "03",
    icon: FileText,
    title: "Decide",
    detail:
      "Recommendations become structured decision objects instead of loose notes or hidden chat output — clear, reviewable, and linked to evidence.",
  },
  {
    num: "04",
    icon: ShieldCheck,
    title: "Approve",
    detail:
      "Important portfolio and research changes pause for human review before they become official. This is the gate where you stay in control.",
    gate: true,
  },
  {
    num: "05",
    icon: GitBranch,
    title: "Record",
    detail:
      "The system writes the final version with evidence, timing, actor, and provenance attached — then feeds the loop so the next decision compounds.",
  },
]

const TRUST_PANELS: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: FileText,
    title: "Evidence stays attached",
    body: "Research outputs point back to the market data, documents, workflow runs, and snapshots that supported them.",
  },
  {
    icon: GitBranch,
    title: "History is versioned",
    body: "The ontology records what Talisman believed at the time, then preserves later corrections as new versions.",
  },
  {
    icon: Lock,
    title: "Automation has guardrails",
    body: "Agents can draft and propose, but governed actions require approval before changing user-visible state.",
  },
]

const BACKBONE_FACTS = [
  "Cloud Run scales the API and job runners.",
  "Cloud SQL stores structured portfolio and workflow state.",
  "Cloud Storage preserves generated documents and artifacts.",
  "Cloud Scheduler refreshes market snapshots and maintenance jobs.",
]

/* ──────────────────────────── agent data ─────────────────────────────────── */

interface PipelineStep {
  key: PipelineKey
  icon: LucideIcon
  label: string
  sub: string
}

type PipelineKey = "domain" | "intent" | "analysis" | "tools" | "record"
type PipelineState = "idle" | "active" | "done" | "blocked"

const PIPELINE: PipelineStep[] = [
  { key: "domain", icon: Shield, label: "Domain check", sub: "Is this a finance question?" },
  { key: "intent", icon: Route, label: "Intent routing", sub: "Classify the request" },
  { key: "analysis", icon: Cpu, label: "Hidden analysis", sub: "Decision-quality pass" },
  { key: "tools", icon: Wrench, label: "Tool loop", sub: "Fetch live data" },
  { key: "record", icon: GitBranch, label: "Record", sub: "Persist + stage proposal" },
]

interface Scenario {
  id: string
  prompt: string
  intent?: string
  tools?: string[]
  reply: string
  proposal?: { title: string; body: string }
  blocked?: boolean
}

const SCENARIOS: Scenario[] = [
  {
    id: "thesis",
    prompt: "What do you think of my long NVDA thesis?",
    intent: "Thesis review",
    tools: ["Portfolio", "Dossier · NVDA", "Market data"],
    reply:
      "Your NVDA long still has an edge, but it's getting crowded. The core thesis — datacenter capex stays strong through 2026 — is intact, and your AVGO and MU reads corroborate it. What's changed is the asymmetry: consensus now agrees with you, so the variant view is thinner. The kill condition to watch is a hyperscaler capex guide-down. My call: hold, don't add.",
    proposal: {
      title: "Stage thesis update → “Crowded — hold”",
      body: "Flags the NVDA thesis as crowded and attaches the kill condition. Awaiting your approval.",
    },
  },
  {
    id: "scan",
    prompt: "Scan for ideas in energy",
    intent: "Opportunity discovery",
    tools: ["Screeners", "Sector metrics", "Commodities curve"],
    reply:
      "Energy screens cheap, but the tape is weak — the sector is down ~5% on the week and the WTI curve is in backwardation. Two names stand out as research candidates, not buys: a refiner with widening crack spreads, and an integrated major trading below replacement value. Both need a catalyst before they're actionable.",
    proposal: {
      title: "Add 2 candidates to Idea Watchlist",
      body: "Stages two ideas with the screen that surfaced them. Nothing enters your book.",
    },
  },
  {
    id: "trim",
    prompt: "Should I trim my NVDA position?",
    intent: "Portfolio query",
    tools: ["Portfolio", "Position sizing", "Risk snapshot"],
    reply:
      "NVDA is now ~14% of your book — your largest position and above your usual 10% cap. On conviction it earns a seat, but that concentration adds real drawdown risk into the next earnings print. Trimming roughly a third brings you back to target without abandoning the thesis.",
    proposal: {
      title: "Draft action item: trim NVDA → 10%",
      body: "Creates a sizing action item for your review. Stan never resizes the position himself.",
    },
  },
  {
    id: "offtopic",
    prompt: "Any dinner spots tonight?",
    blocked: true,
    reply:
      "That's outside what I do. I stay inside markets, your portfolio, and Talisman's research workflows — ask me about a thesis, a position, or what's moving today and I'm all in.",
  },
]

const IDENTITY: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: Target,
    title: "Sharp co-PM, not an assistant",
    body: "Stan has a view, leads with conclusions, and pushes back when a thesis has a hole. He tells you what would change his mind instead of hedging forever.",
  },
  {
    icon: Shield,
    title: "Finance-only domain",
    body: "Stan stays within markets, portfolio, macro, thesis, and Talisman workflow questions. Off-topic requests are declined politely.",
  },
  {
    icon: Lock,
    title: "Decision support, not execution",
    body: "Stan recommends and records. He does not place trades, size positions autonomously, or mutate your book without approval.",
  },
]

const INTENTS = [
  { t: "Thesis review", e: "What do you think of my idea on X?" },
  { t: "Opportunity discovery", e: "Scan for ideas in Y" },
  { t: "Catalyst status", e: "Did this catalyst play out?" },
  { t: "Portfolio query", e: "Holdings, exposure, and P&L" },
  { t: "Workflow handoff", e: "Run my Morning Brief" },
  { t: "General research", e: "Informational market questions" },
  { t: "Casual", e: "Greetings and light chat" },
]

const PASSES: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: ListChecks,
    title: "Decision Quality pass",
    body: "A structured second opinion before Stan speaks: thesis clarity, mispricing, catalyst, evidence for and against, invalidation, and sizing. Gates can downgrade or block a recommendation when inputs are thin. You see prose — never the raw checklist.",
  },
  {
    icon: Eye,
    title: "Opportunity Candidate pass",
    body: "Lighter triage for discovery scans: ranks ideas and may graduate promising ones to full decision quality. Never issues buy or sell calls on its own.",
  },
]

const TOOLS: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: Database,
    title: "Read tools",
    body: "Portfolio, dossiers, market data, charts, and ontology queries. Stan must fetch before asserting any price, position, or catalyst status.",
  },
  {
    icon: FileText,
    title: "Proposal tools",
    body: "Thesis changes, catalysts, action items, and watch triggers are staged as pending approvals. Stan drafts; he never writes to your book directly.",
  },
  {
    icon: ShieldCheck,
    title: "Quality gates",
    body: "Data-quality warnings and policy checks — concentration, liquidity, stale data — surface before anything actionable is staged for review.",
  },
]

const CAN_DO = [
  "Read live portfolio, market, and macro data",
  "Pressure-test a thesis and name the kill conditions",
  "Run governed workflows and discovery scans",
  "Draft proposals: thesis changes, action items, watch triggers",
]
const CANT_DO = [
  "Place or execute trades",
  "Resize a position autonomously",
  "Mutate your book without approval",
  "Answer outside the finance domain",
]

const WORKFLOWS = [
  { t: "Morning Brief", d: "Overnight moves, portfolio snapshot, and today's focus." },
  { t: "Thesis Review", d: "Structured pressure-test of an active thesis." },
  { t: "Pre-Earnings Prep", d: "Positioning and expectations ahead of a print." },
  { t: "Post-Earnings Review", d: "What changed, what held, what to do next." },
  { t: "Weekly Portfolio Review", d: "Cross-portfolio risk, P&L context, action items." },
  { t: "Thesis Invalidation Check", d: "Kill conditions, catalyst status, thesis drift." },
]

const SAFETY: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: Lock,
    title: "Approval-first writeback",
    body: "Stan drafts; you approve in the UI. Applied changes get versioned in the ontology — see Architecture for the full trust model.",
  },
  {
    icon: ShieldCheck,
    title: "Model egress with audit",
    body: "Context sent to external models is governed with warnings and an audit trail, so you always know what left the boundary.",
  },
  {
    icon: ClipboardCheck,
    title: "Durable async turns",
    body: "Long agent runs execute on warm worker pools, so chat stays responsive while deeper analysis completes.",
  },
]

const PROOF_POINTS: { icon: LucideIcon; t: string; d: string }[] = [
  {
    icon: Workflow,
    t: "A research system",
    d: "Not a chat box — dashboards, models, and workflows on shared memory.",
  },
  {
    icon: ShieldCheck,
    t: "Governed by design",
    d: "Evidence, versioning, and approvals behind every output.",
  },
  {
    icon: Bot,
    t: "Stan, your co-pilot",
    d: "An agent that proposes — and never trades without you.",
  },
]

/* ──────────────────────────── product data ───────────────────────────────── */

const FEATURED_PRODUCT = {
  img: portfolioImg,
  cat: "Core",
  title: "Your whole book, at a glance",
  body:
    "Track the entire portfolio in one adaptive grid — real-time P&L, per-position charts, and comparable performance across daily, weekly, and monthly timeframes. Group by conviction or theme and drill into any name.",
  chips: ["Real-time P&L", "Per-position charts", "Theme grouping"],
}

const PRODUCT_FEATURES: { img: string; cat: string; title: string; body: string }[] = [
  {
    img: sectorMetricsImg,
    cat: "Markets",
    title: "Sector Metrics",
    body: "Relative performance across all 11 sectors — ETF returns, breadth, and rotation, normalized so leaders and laggards are obvious.",
  },
  {
    img: economicGrowthImg,
    cat: "Macro",
    title: "Economic Growth",
    body: "The macro cycle in real time: commodities, equities vs. benchmark, and leading indicators that gauge where growth is heading.",
  },
  {
    img: centralBankImg,
    cat: "Macro",
    title: "Central Bank Monitor",
    body: "AI-summarized policy across 10 central banks — rate decisions, minutes, and forward guidance, with the key lines pulled out.",
  },
  {
    img: sentimentImg,
    cat: "Markets",
    title: "Sentiment",
    body: "Investor positioning and mood — put/call, AAII surveys, volatility, and the NAAIM exposure index — to spot crowded trades.",
  },
  {
    img: industryImg,
    cat: "Research",
    title: "Industry Monitor",
    body: "Macro signals pulled straight from earnings-call transcripts across housing, trucking, banks, retail, and capital goods.",
  },
  {
    img: laborMarketImg,
    cat: "Macro",
    title: "Labor Market",
    body: "Jobless claims, JOLTS, wage growth, and hours worked — the health of the consumer in a single dashboard.",
  },
  {
    img: commoditiesCurveImg,
    cat: "Markets",
    title: "Commodities Curve",
    body: "Forward term structure vs. 30 days ago — front month, spreads, and contango or backwardation at a glance.",
  },
  {
    img: ontologyWorkbenchImg,
    cat: "Research",
    title: "Ontology Workbench",
    body: "Query portfolio-linked risk snapshots in natural language. Read-only evidence, fully versioned — the memory behind every decision.",
  },
]

/* ──────────────────────────── architecture: system diagram ───────────────── */

function SystemDiagram() {
  const [active, setActive] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const stop = useCallback(() => {
    setPlaying(false)
    if (timer.current) clearTimeout(timer.current)
  }, [])

  useEffect(() => {
    if (!playing) return
    if (active >= SYSTEM_NODES.length - 1) {
      timer.current = setTimeout(() => setPlaying(false), 1400)
      return () => {
        if (timer.current) clearTimeout(timer.current)
      }
    }
    timer.current = setTimeout(() => setActive(a => a + 1), 1200)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [playing, active])

  const play = () => {
    setActive(0)
    setPlaying(true)
  }

  const node = SYSTEM_NODES[active]
  const NodeIcon = node.icon

  return (
    <div>
      <div className="landing-diagram-row">
        {SYSTEM_NODES.map((n, i) => {
          const Icon = n.icon
          return (
            <div key={n.title} className="flex flex-1 items-stretch gap-1 min-w-0">
              <button
                type="button"
                className="landing-diagram-node"
                data-active={i <= active}
                onClick={() => {
                  stop()
                  setActive(i)
                }}
              >
                <div className="node-step">{n.step}</div>
                <IconChip icon={Icon} />
                <div>
                  <div className="landing-card-title">{n.title}</div>
                  <p className="landing-body-sm mt-1">{n.short}</p>
                </div>
              </button>
              {i < SYSTEM_NODES.length - 1 && (
                <div className="landing-diagram-link" data-active={i < active}>
                  <ArrowRight aria-hidden="true" />
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="landing-detail-panel" key={active}>
        <div className="landing-fade-in">
          <div className="flex items-center gap-2.5 mb-2">
            <span className="landing-badge info">
              Stage {active + 1} of {SYSTEM_NODES.length}
            </span>
            <span className="landing-card-title flex items-center gap-2">
              <NodeIcon className="h-4 w-4 text-[hsl(var(--accent))]" aria-hidden="true" />
              {node.title}
            </span>
          </div>
          <p className="landing-body text-pretty">{node.detail}</p>
          <div className="landing-chip-row mt-3.5">
            {node.stack.map(s => (
              <span className="landing-tech-chip" key={s}>
                <span className="dot" />
                {s}
              </span>
            ))}
          </div>
        </div>
        <div className="landing-flow-meta landing-fade-in">
          <div className="landing-flow-meta-row">
            <span className="k">In</span>
            <span className="v">{node.inflow}</span>
          </div>
          <div className="landing-flow-meta-row">
            <span className="k">Out</span>
            <span className="v">{node.outflow}</span>
          </div>
          <button
            type="button"
            className="landing-trace-btn mt-1"
            onClick={playing ? stop : play}
          >
            {playing ? <X aria-hidden="true" /> : <Zap aria-hidden="true" />}
            {playing ? "Stop" : "Trace a request"}
          </button>
        </div>
      </div>
    </div>
  )
}

/* ──────────────────────────── architecture: research loop ────────────────── */

function ResearchLoop() {
  const [active, setActive] = useState(0)
  const [auto, setAuto] = useState(true)

  useEffect(() => {
    if (!auto) return
    const t = setTimeout(() => setActive(a => (a + 1) % LOOP_STEPS.length), 2600)
    return () => clearTimeout(t)
  }, [auto, active])

  const step = LOOP_STEPS[active]
  const StepIcon = step.icon

  return (
    <div>
      <div className="landing-stepper">
        {LOOP_STEPS.map((s, i) => {
          const Icon = s.icon
          return (
            <button
              type="button"
              key={s.title}
              className="landing-step"
              data-active={i === active}
              data-gate={!!s.gate}
              onClick={() => {
                setAuto(false)
                setActive(i)
              }}
            >
              <div className="landing-step-top">
                <span className="landing-step-num">{s.num}</span>
                <span className="landing-step-icon">
                  <Icon aria-hidden="true" />
                </span>
              </div>
              <span className="landing-step-title">{s.title}</span>
              {s.gate && (
                <span className="landing-gate-flag">
                  <User aria-hidden="true" />
                  Human gate
                </span>
              )}
            </button>
          )
        })}
      </div>
      <div className="landing-step-detail" key={active}>
        <IconChip icon={StepIcon} variant={step.gate ? "accent" : undefined} />
        <div className="landing-fade-in flex-1">
          <div className="flex items-center gap-2.5">
            <span className="landing-card-title">{step.title}</span>
            {step.gate && (
              <span className="landing-badge info">
                <Lock aria-hidden="true" />
                Approval required
              </span>
            )}
          </div>
          <p className="landing-body text-pretty mt-1.5">{step.detail}</p>
        </div>
        <span className="landing-badge neutral font-mono ml-auto shrink-0">loops &#8635;</span>
      </div>
    </div>
  )
}

/* ──────────────────────────── architecture showcase ──────────────────────── */

function ArchitectureShowcase() {
  return (
    <div className="flex flex-col gap-[clamp(3rem,2rem+4vw,5rem)]">
      <SectionHead
        eyebrow="Backend architecture"
        title="A trust engine for investment research"
        maxWidth="46rem"
      >
        Talisman is built like a research desk with memory. The front end feels simple, but behind
        every dashboard is a governed system that gathers evidence, runs deeper analysis in the
        background, and keeps a clear record of how decisions were formed.
      </SectionHead>

      <div>
        <div className="mb-5 max-w-[44rem]">
          <Eyebrow accent>System map</Eyebrow>
          <h3 className="landing-subsection-title mt-2">From one click to a research system</h3>
          <p className="landing-body text-pretty mt-2.5">
            Follow a single request across the stack. Tap any stage to see what it does — or trace
            the whole path end to end.
          </p>
        </div>
        <SystemDiagram />
      </div>

      <div>
        <div className="mb-5 max-w-[44rem]">
          <Eyebrow accent>Research loop</Eyebrow>
          <h3 className="landing-subsection-title mt-2">
            Every decision moves through five stages
          </h3>
          <p className="landing-body text-pretty mt-2.5">
            Heavy analysis never blocks the interface. Work flows from raw inputs to a recorded
            decision — pausing at a human gate before anything becomes official.
          </p>
        </div>
        <ResearchLoop />
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <div>
          <Eyebrow accent>Why it matters</Eyebrow>
          <h3 className="landing-subsection-title mt-2">
            AI is useful only when it's accountable
          </h3>
          <p className="landing-body text-pretty mt-2.5">
            Talisman treats generated research as a proposal, not a command. The ontology and
            provenance layers turn analysis into structured objects with evidence, timestamps,
            approvals, and version history.
          </p>
          <div
            className="mt-4 rounded-[var(--radius-lg)] border p-4"
            style={{
              backgroundColor: "hsl(var(--background-card-muted))",
              borderColor: "hsl(var(--separator))",
            }}
          >
            <Eyebrow>Production backbone</Eyebrow>
            <div className="mt-2.5 flex flex-col gap-2">
              {BACKBONE_FACTS.map(b => (
                <div key={b} className="flex items-start gap-2">
                  <CheckCircle2
                    className="h-4 w-4 flex-none text-positive"
                    style={{ marginTop: "0.18rem" }}
                    aria-hidden="true"
                  />
                  <span className="landing-body-sm">{b}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="landing-cards-grid">
          {TRUST_PANELS.map(p => {
            const Icon = p.icon
            return (
              <div className="landing-info-card" key={p.title}>
                <IconChip icon={Icon} />
                <div>
                  <div className="landing-card-title">{p.title}</div>
                  <p className="landing-body-sm text-pretty mt-1.5">{p.body}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ──────────────────────────── agent: scripted Stan demo ──────────────────── */

interface ChatMessage {
  id: number
  role: "user" | "stan"
  content: string
  meta?: boolean
  declined?: boolean
  proposal?: { title: string; body: string }
}

function renderBold(text: string) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g)
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? (
      <strong key={i}>{p.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{p}</span>
    ),
  )
}

function PipeRail({ states }: { states: Partial<Record<PipelineKey, PipelineState>> }) {
  return (
    <div className="landing-pipe-rail">
      {PIPELINE.map(p => {
        const st: PipelineState = states[p.key] ?? "idle"
        const status =
          st === "active" ? "running" : st === "done" ? "done" : st === "blocked" ? "declined" : ""
        const Icon = p.icon
        return (
          <div className="landing-pipe-item" data-state={st} key={p.key}>
            <span className="landing-pipe-dot">
              {st === "done" ? (
                <CheckCircle2 aria-hidden="true" />
              ) : st === "blocked" ? (
                <X aria-hidden="true" />
              ) : st === "active" ? (
                <Icon className="landing-pipe-spin" aria-hidden="true" />
              ) : (
                <Icon aria-hidden="true" />
              )}
            </span>
            <div>
              <div className="landing-pipe-label">{p.label}</div>
              <div className="landing-pipe-sub">{p.sub}</div>
            </div>
            {status && <span className="landing-pipe-status">{status}</span>}
          </div>
        )
      })}
    </div>
  )
}

function StanDemo() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 0,
      role: "stan",
      content: "I'm Stan — your research co-pilot. Pick a question below and watch how I handle it.",
    },
  ])
  const [pipe, setPipe] = useState<Partial<Record<PipelineKey, PipelineState>>>({})
  const [running, setRunning] = useState(false)
  const [typing, setTyping] = useState(false)
  const [usedIds, setUsedIds] = useState<string[]>([])
  const runId = useRef(0)
  const msgId = useRef(1)
  const bodyRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [messages, typing, pipe])

  const wait = (ms: number, id: number) =>
    new Promise<void>(res => {
      setTimeout(() => {
        if (runId.current === id) res()
      }, ms)
    })

  const pushMessage = (m: Omit<ChatMessage, "id">) =>
    setMessages(prev => [...prev, { ...m, id: msgId.current++ }])

  async function run(sc: Scenario) {
    const id = ++runId.current
    setRunning(true)
    setPipe({})
    pushMessage({ role: "user", content: sc.prompt })
    setUsedIds(u => (u.includes(sc.id) ? u : [...u, sc.id]))
    await wait(450, id)
    if (runId.current !== id) return

    setPipe({ domain: "active" })
    await wait(700, id)
    if (runId.current !== id) return

    if (sc.blocked) {
      setPipe({ domain: "blocked" })
      await wait(550, id)
      if (runId.current !== id) return
      setTyping(true)
      await wait(700, id)
      if (runId.current !== id) return
      setTyping(false)
      pushMessage({ role: "stan", content: sc.reply, declined: true })
      setRunning(false)
      return
    }

    setPipe({ domain: "done" })
    setPipe(p => ({ ...p, intent: "active" }))
    await wait(750, id)
    if (runId.current !== id) return
    setPipe(p => ({ ...p, intent: "done" }))
    pushMessage({ role: "stan", content: `Routed as **${sc.intent}**.`, meta: true })

    setPipe(p => ({ ...p, analysis: "active" }))
    await wait(900, id)
    if (runId.current !== id) return
    setPipe(p => ({ ...p, analysis: "done" }))

    setPipe(p => ({ ...p, tools: "active" }))
    setTyping(true)
    pushMessage({
      role: "stan",
      content: `Fetching ${(sc.tools ?? []).join(" · ")}`,
      meta: true,
    })
    await wait(1100, id)
    if (runId.current !== id) return
    setPipe(p => ({ ...p, tools: "done" }))

    setPipe(p => ({ ...p, record: "active" }))
    await wait(650, id)
    if (runId.current !== id) return
    setPipe(p => ({ ...p, record: "done" }))
    setTyping(false)
    pushMessage({ role: "stan", content: sc.reply, proposal: sc.proposal })
    setRunning(false)
  }

  function reset() {
    runId.current++
    setRunning(false)
    setTyping(false)
    setPipe({})
    setUsedIds([])
    setMessages([
      {
        id: msgId.current++,
        role: "stan",
        content:
          "I'm Stan — your research co-pilot. Pick a question below and watch how I handle it.",
      },
    ])
  }

  return (
    <div className="landing-chat-shell">
      <div className="landing-chat-window">
        <div className="landing-chat-head">
          <span className="landing-stan-avatar">
            <Bot aria-hidden="true" />
          </span>
          <div>
            <div className="landing-card-title text-base">Stan</div>
            <div className="landing-chat-head-meta">research co-pilot · guarded</div>
          </div>
          <div className="landing-chat-dotrow">
            <span />
            <span />
            <span />
          </div>
        </div>
        <div className="landing-chat-body" ref={bodyRef}>
          {messages.map(m => {
            if (m.role === "user") {
              return (
                <div className="landing-bubble user" key={m.id}>
                  {m.content}
                </div>
              )
            }
            if (m.meta) {
              return (
                <div className="landing-stan-row items-center" key={m.id}>
                  <span className="landing-badge neutral font-mono" style={{ fontWeight: 600 }}>
                    <Sparkles aria-hidden="true" />
                    {renderBold(m.content)}
                  </span>
                </div>
              )
            }
            return (
              <div className="landing-stan-row" key={m.id}>
                <span className="landing-stan-avatar">
                  <Bot aria-hidden="true" />
                </span>
                <div className="flex flex-col">
                  <div className="landing-bubble stan">{renderBold(m.content)}</div>
                  {m.proposal && (
                    <div className="landing-proposal-card">
                      <div className="landing-proposal-top">
                        <span className="landing-badge info">
                          <Lock aria-hidden="true" />
                          Proposal · awaiting approval
                        </span>
                      </div>
                      <div className="landing-card-title text-[0.92rem]">{m.proposal.title}</div>
                      <p className="landing-body-sm mt-1">{m.proposal.body}</p>
                      <div className="landing-proposal-actions">
                        <span className="landing-mini-btn approve">
                          <CheckCircle2 aria-hidden="true" />
                          Approve
                        </span>
                        <span className="landing-mini-btn dismiss">Dismiss</span>
                      </div>
                    </div>
                  )}
                  {m.declined && (
                    <span
                      className="landing-badge warning mt-2 self-start"
                    >
                      <Shield aria-hidden="true" />
                      Finance-only guardrail
                    </span>
                  )}
                </div>
              </div>
            )
          })}
          {typing && (
            <div className="landing-stan-row">
              <span className="landing-stan-avatar">
                <Bot aria-hidden="true" />
              </span>
              <div className="landing-bubble stan">
                <span className="landing-typing">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            </div>
          )}
        </div>
        <div className="landing-chat-foot">
          <div className="landing-prompt-chips">
            {SCENARIOS.map(sc => (
              <button
                type="button"
                key={sc.id}
                className="landing-prompt-chip"
                disabled={running || usedIds.includes(sc.id)}
                onClick={() => run(sc)}
              >
                {sc.prompt}
              </button>
            ))}
            {usedIds.length > 0 && !running && (
              <button
                type="button"
                className="landing-prompt-chip"
                onClick={reset}
                style={{ color: "hsl(var(--accent))" }}
              >
                Reset ↺
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div
          className="rounded-[var(--radius-lg)] border px-4 py-3"
          style={{
            backgroundColor: "hsl(var(--background-card-muted))",
            borderColor: "hsl(var(--separator))",
          }}
        >
          <Eyebrow accent>Live request pipeline</Eyebrow>
          <p className="landing-body-sm mt-1.5">
            Every message runs the same governed path. Watch it light up as Stan answers.
          </p>
        </div>
        <PipeRail states={pipe} />
      </div>
    </div>
  )
}

/* ──────────────────────────── agent showcase ─────────────────────────────── */

function AgentShowcase() {
  return (
    <div className="flex flex-col gap-[clamp(3rem,2rem+4vw,5rem)]">
      <SectionHead
        eyebrow="AI agent"
        title="Stan — a research co-pilot with guardrails"
        maxWidth="48rem"
      >
        Stan isn't a generic chatbot or a trading bot. He's an investment research agent embedded
        in Talisman: he reads live portfolio and market data, pressure-tests ideas, runs governed
        workflows, and proposes changes — but never executes trades or mutates your book without
        approval.
      </SectionHead>

      <div className="grid gap-3 md:grid-cols-3">
        {IDENTITY.map(p => {
          const Icon = p.icon
          return (
            <div
              className="landing-info-card flex-col gap-3.5"
              key={p.title}
            >
              <IconChip icon={Icon} />
              <div>
                <div className="landing-card-title">{p.title}</div>
                <p className="landing-body-sm text-pretty mt-1.5">{p.body}</p>
              </div>
            </div>
          )
        })}
      </div>

      <div>
        <div className="mb-5 max-w-[46rem]">
          <Eyebrow accent>See it work</Eyebrow>
          <h3 className="landing-subsection-title mt-2">Watch a question move through Stan</h3>
          <p className="landing-body text-pretty mt-2.5">
            Send one of the sample questions. Stan classifies it, runs a hidden quality pass,
            fetches live data, and answers — ending in a proposal that waits for your approval.
            Ask something off-topic and the guardrail kicks in.
          </p>
        </div>
        <StanDemo />
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <div>
          <Eyebrow accent>Intent routing</Eyebrow>
          <h3 className="landing-subsection-title mt-2">The right analysis for the question</h3>
          <p className="landing-body text-pretty mt-2.5 mb-4">
            The router sorts every message into one of seven classes. When confidence is low, a
            deterministic fallback keeps routing reliable.
          </p>
          <div className="flex flex-col gap-2">
            {INTENTS.map(it => (
              <div className="landing-intent-row" key={it.t}>
                <span className="it">{it.t}</span>
                <span className="ie">{it.e}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <Eyebrow accent>Hidden passes</Eyebrow>
          <h3 className="landing-subsection-title mt-2">Rigor happens before Stan speaks</h3>
          <p className="landing-body text-pretty mt-2.5 mb-4">
            For serious decisions, structured internal passes run first. You see natural language;
            the scoring stays behind the scenes.
          </p>
          <div className="landing-cards-grid">
            {PASSES.map(p => {
              const Icon = p.icon
              return (
                <div className="landing-info-card" key={p.title}>
                  <IconChip icon={Icon} />
                  <div>
                    <div className="landing-card-title">{p.title}</div>
                    <p className="landing-body-sm text-pretty mt-1.5">{p.body}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div>
          <Eyebrow accent>Tools and context</Eyebrow>
          <h3 className="landing-subsection-title mt-2">Live data, not stale memory</h3>
          <p className="landing-body text-pretty mt-2.5 mb-4">
            Stan reaches Talisman's analysis modules through a governed tool layer. Read tools
            fetch facts; proposal tools draft memos awaiting your signature.
          </p>
          <div className="landing-cards-grid">
            {TOOLS.map(t => {
              const Icon = t.icon
              return (
                <div className="landing-info-card" key={t.title}>
                  <IconChip icon={Icon} />
                  <div>
                    <div className="landing-card-title">{t.title}</div>
                    <p className="landing-body-sm text-pretty mt-1.5">{t.body}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <div
          className="rounded-[var(--radius-xl)] border p-5"
          style={{
            backgroundColor: "hsl(var(--background-card) / 0.9)",
            borderColor: "hsl(var(--border))",
            boxShadow: "var(--shadow-soft)",
          }}
        >
          <IconChip icon={MessageSquare} variant="accent" />
          <div className="landing-card-title mt-3.5">Screen context</div>
          <p className="landing-body-sm text-pretty mt-2">
            Dashboards register what you're looking at — page, ticker, visible metrics — and pass
            it with your message. Ask "explain this chart" while viewing Market Technicals and
            Stan already knows which chart you mean, without redundant fetches.
          </p>
          <div className="landing-chip-row mt-4">
            <span className="landing-tech-chip">
              <span className="dot" />
              page
            </span>
            <span className="landing-tech-chip">
              <span className="dot" />
              ticker
            </span>
            <span className="landing-tech-chip">
              <span className="dot" />
              visible metrics
            </span>
          </div>
        </div>
      </div>

      <div>
        <div className="mb-5 max-w-[46rem]">
          <Eyebrow accent>The execution boundary</Eyebrow>
          <h3 className="landing-subsection-title mt-2">A clear line Stan will not cross</h3>
          <p className="landing-body text-pretty mt-2.5">
            Everything on the left, Stan does on his own. Everything on the right needs you. The
            boundary is structural, not a setting.
          </p>
        </div>
        <div className="landing-boundary">
          <div className="landing-boundary-col can">
            <span className="landing-badge success">
              <CheckCircle2 aria-hidden="true" />
              Stan can
            </span>
            <div className="mt-3">
              {CAN_DO.map(c => (
                <div className="landing-boundary-item" key={c}>
                  <CheckCircle2 className="tick" aria-hidden="true" />
                  {c}
                </div>
              ))}
            </div>
          </div>
          <div className="landing-boundary-line" />
          <div className="landing-boundary-col cant">
            <span className="landing-badge warning">
              <Lock aria-hidden="true" />
              Needs your approval
            </span>
            <div className="mt-3">
              {CANT_DO.map(c => (
                <div className="landing-boundary-item" key={c}>
                  <X className="cross" aria-hidden="true" />
                  {c}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div>
        <div className="mb-5 max-w-[46rem]">
          <Eyebrow accent>Governed workflows</Eyebrow>
          <h3 className="landing-subsection-title mt-2">
            Deterministic playbooks, not ad-hoc prompts
          </h3>
          <p className="landing-body text-pretty mt-2.5">
            Workflows are fixed tool sequences plus a single synthesis step — launched from the UI.
            Each run becomes an ontology object with artifacts you can revisit.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {WORKFLOWS.map(w => (
            <div className="landing-wf-card" key={w.t}>
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-[hsl(var(--accent))]" aria-hidden="true" />
                <span className="landing-card-title text-[0.98rem]">{w.t}</span>
              </div>
              <p className="landing-body-sm text-pretty mt-2">{w.d}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <div>
          <Eyebrow accent>Safety and memory</Eyebrow>
          <h3 className="landing-subsection-title mt-2">Useful AI stays accountable</h3>
          <p className="landing-body text-pretty mt-2.5">
            Stan remembers session history and can reference prior workflow runs, but every
            important belief change leaves a trace.
          </p>
        </div>
        <div className="landing-cards-grid">
          {SAFETY.map(s => {
            const Icon = s.icon
            return (
              <div className="landing-info-card" key={s.title}>
                <IconChip icon={Icon} />
                <div>
                  <div className="landing-card-title">{s.title}</div>
                  <p className="landing-body-sm text-pretty mt-1.5">{s.body}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ──────────────────────────── product showcase ───────────────────────────── */

function ProductShowcase() {
  return (
    <div className="flex flex-col gap-[clamp(2.5rem,2rem+3vw,4rem)]">
      <SectionHead
        eyebrow="The product"
        title="Everything you need to stay on top of the markets"
        maxWidth="46rem"
      >
        Professional-grade analytics in one workspace — portfolio, macro, markets, and research,
        each a focused dashboard that feeds the same shared memory.
      </SectionHead>

      <div className="landing-featured">
        <div className="landing-shot-frame">
          <img src={FEATURED_PRODUCT.img} alt={FEATURED_PRODUCT.title} />
        </div>
        <div className="feat-body">
          <span className="landing-feature-cat">{FEATURED_PRODUCT.cat}</span>
          <h3 className="landing-subsection-title mt-2">{FEATURED_PRODUCT.title}</h3>
          <p className="landing-body text-pretty mt-2.5">{FEATURED_PRODUCT.body}</p>
          <div className="landing-chip-row mt-5">
            {FEATURED_PRODUCT.chips.map(chip => (
              <span className="landing-tech-chip" key={chip}>
                <span className="dot" />
                {chip}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="landing-product-grid">
        {PRODUCT_FEATURES.map(f => (
          <div className="landing-feature-card" key={f.title}>
            <div className="landing-shot-frame">
              <img src={f.img} alt={f.title} loading="lazy" />
            </div>
            <div className="fc-body">
              <span className="landing-feature-cat">{f.cat}</span>
              <div className="landing-card-title mt-1.5">{f.title}</div>
              <p className="landing-body-sm text-pretty mt-2">{f.body}</p>
            </div>
          </div>
        ))}
      </div>

      <p
        className="mx-auto max-w-[40rem] text-center"
        style={{ color: "hsl(var(--foreground-tertiary))", fontSize: "0.9rem", lineHeight: 1.58 }}
      >
        Plus FX, bonds, yield curve, housing, liquidity, screeners, DCF and FX models, and more —
        all wired to Stan and the shared ontology.
      </p>
    </div>
  )
}

/* ──────────────────────────── public showcase tabs ───────────────────────── */

function PublicShowcase() {
  const [activeTab, setActiveTab] = useState<PublicTab>("product")

  return (
    <div className="w-full">
      <div className="mb-10 flex justify-center px-4 sm:px-6">
        <SegmentedControl options={publicTabs} value={activeTab} onChange={setActiveTab} />
      </div>
      <div className="mx-auto w-full max-w-6xl px-4 pb-24 sm:px-6 landing-fade-in" key={activeTab}>
        {activeTab === "product" && <ProductShowcase />}
        {activeTab === "architecture" && <ArchitectureShowcase />}
        {activeTab === "agent" && <AgentShowcase />}
      </div>
    </div>
  )
}

/* ──────────────────────────── login card + hero ──────────────────────────── */

function LoginCard() {
  const { login, mode } = useAuth()
  const navigate = useNavigate()
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await login(password)
      navigate("/", { replace: true })
    } catch {
      setError("Incorrect password. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      id="signin"
      className="w-full max-w-md p-6 sm:p-8 theme-surface"
      style={{ alignSelf: "center" }}
    >
      <p className="theme-eyebrow mb-3">Welcome back</p>
      <h2 className="text-[2rem] font-semibold tracking-[-0.04em] text-app">Talisman</h2>

      {mode === "cloudflare" ? (
        <>
          <p className="mb-6 body-copy mt-1">Sign-in is handled by Cloudflare Access.</p>
          <button
            type="button"
            onClick={() => login("")}
            className="theme-button-base theme-button-primary w-full"
          >
            Continue
          </button>
        </>
      ) : (
        <>
          <p className="mb-6 body-copy mt-1">Enter your password to continue.</p>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="password" className="theme-field-label">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="theme-input"
              />
            </div>

            {error && <Notice tone="error">{error}</Notice>}

            <button
              type="submit"
              disabled={loading}
              className="theme-button-base theme-button-primary w-full disabled:opacity-50"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>
        </>
      )}
      <p
        className="mt-4"
        style={{ color: "hsl(var(--foreground-tertiary))", fontSize: "0.78rem" }}
      >
        Scroll down to see how Talisman works — the product, the architecture, and Stan, the AI
        agent.
      </p>
    </div>
  )
}

function Hero() {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 pt-6 md:pt-12 pb-[clamp(2.5rem,2rem+3vw,4rem)]">
      <div className="landing-hero">
        <div>
          <p className="landing-eyebrow accent">Investment research platform</p>
          <h1 className="page-title text-pretty mt-3">Investment research with a memory.</h1>
          <p className="landing-lead text-pretty mt-4 max-w-[34rem]">
            Talisman gathers the evidence, runs the deep analysis in the background, and keeps a
            clear record of how every decision was formed — with an AI co-pilot that has a view
            but stays on a leash.
          </p>
          <div className="flex flex-col gap-3.5 mt-7 max-w-[32rem]">
            {PROOF_POINTS.map(p => {
              const Icon = p.icon
              return (
                <div key={p.t} className="flex items-start gap-3">
                  <IconChip icon={Icon} size="sm" />
                  <div>
                    <div className="landing-card-title text-base">{p.t}</div>
                    <p className="landing-body-sm mt-0.5">{p.d}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <LoginCard />
      </div>

      <div className="landing-shot mt-[clamp(2.5rem,2rem+3vw,4rem)]">
        <div className="landing-shot-bar">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
          <span
            className="ml-1.5"
            style={{
              fontFamily:
                'ui-monospace, "SFMono-Regular", "SF Mono", Menlo, Monaco, Consolas, monospace',
              fontSize: "0.74rem",
              color: "hsl(var(--foreground-tertiary))",
            }}
          >
            app.talisman — Portfolio Dashboard
          </span>
        </div>
        <img src={portfolioImg} alt="Talisman portfolio dashboard" />
      </div>
    </div>
  )
}

/* ──────────────────────────── page ───────────────────────────────────────── */

export function LoginPage() {
  return (
    <div className="min-h-screen bg-app text-app">
      <div className="theme-page">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-stretch gap-12 pb-16">
          <Hero />
          <PublicShowcase />
        </div>
      </div>
    </div>
  )
}
