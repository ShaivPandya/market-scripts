import { type FormEvent, useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Cloud,
  Database,
  FileText,
  GitBranch,
  Lock,
  Server,
  ShieldCheck,
  Workflow,
  type LucideIcon,
} from "lucide-react"
import { useAuth } from "@/contexts/AuthContext"
import { SegmentedControl } from "@/components/shared/FormControls"
import { Notice } from "@/components/shared/Notice"
import { SurfaceCard } from "@/components/shared/SurfaceCard"

import centralBankImg from "@/assets/central-bank-monitor.jpg"
import countryDashboardImg from "@/assets/country-dashboard.jpg"
import economicGrowthImg from "@/assets/economic-growth.jpg"
import industryImg from "@/assets/industry.jpg"
import laborMarketImg from "@/assets/labor-market.jpg"
import marketTechnicalsImg from "@/assets/market-technicals.jpg"
import portfolioImg from "@/assets/portfolio.jpg"
import sentimentImg from "@/assets/sentiment.jpg"

type PublicTab = "product" | "architecture"

const publicTabs: { value: PublicTab; label: string }[] = [
  { value: "product", label: "Product" },
  { value: "architecture", label: "Architecture" },
]

const features = [
  {
    title: "Portfolio Dashboard",
    description:
      "Track your entire portfolio at a glance — real-time P&L, individual position charts, and performance across daily, weekly, and monthly timeframes.",
    image: portfolioImg,
  },
  {
    title: "Market Technicals",
    description:
      "Analyze market breadth, price-volume signals, and technical indicators to identify emerging trends, divergences, and inflection points across major indices.",
    image: marketTechnicalsImg,
  },
  {
    title: "Economic Growth Dashboard",
    description:
      "Monitor the macro cycle in real time — commodity trends, regional growth signals, leading indicators, and cross-asset momentum to gauge where the economy is heading.",
    image: economicGrowthImg,
  },
  {
    title: "Central Bank Monitor",
    description:
      "Stay ahead of monetary policy — AI-generated summaries of central bank decisions, rate changes, and GDP forecasts from the Fed, ECB, and more.",
    image: centralBankImg,
  },
  {
    title: "Industry Monitor",
    description:
      "Drill into sector-level fundamentals with earnings call transcripts, company-level financial snapshots, and AI-generated summaries across housing, trucking, retail, and more.",
    image: industryImg,
  },
  {
    title: "Country Dashboard",
    description:
      "Compare economies side by side — GDP growth, inflation, rates, and trade balances across developed and emerging markets with historical context.",
    image: countryDashboardImg,
  },
  {
    title: "Labor Market",
    description:
      "Track employment trends, jobless claims, wage growth, and labor force participation to assess the health of the consumer and the broader economy.",
    image: laborMarketImg,
  },
  {
    title: "Sentiment",
    description:
      "Gauge investor positioning and mood through put/call ratios, volatility term structure, and survey data to spot crowded trades and contrarian opportunities.",
    image: sentimentImg,
  },
]

const systemMapNodes: {
  title: string
  description: string
  icon: LucideIcon
}[] = [
  {
    title: "Control room",
    description: "The React app on Firebase gives every workflow a fast, polished operating surface.",
    icon: Activity,
  },
  {
    title: "Request brain",
    description: "FastAPI on Cloud Run receives each question and routes it to the right research capability.",
    icon: Server,
  },
  {
    title: "Research engines",
    description: "Python modules combine market data, macro signals, portfolio context, filings, and documents.",
    icon: Workflow,
  },
  {
    title: "Durable memory",
    description: "Cloud SQL and Cloud Storage keep state, versions, artifacts, and generated research recoverable.",
    icon: Database,
  },
]

const researchLoopSteps: {
  label: string
  title: string
  description: string
  icon: LucideIcon
}[] = [
  {
    label: "01",
    title: "Ingest",
    description: "Fresh prices, macro releases, transcripts, PDFs, and portfolio changes enter as source records.",
    icon: Cloud,
  },
  {
    label: "02",
    title: "Analyze",
    description: "Dashboards, models, and LLM workflows turn raw inputs into comparisons, risks, and hypotheses.",
    icon: Activity,
  },
  {
    label: "03",
    title: "Decide",
    description: "Recommendations become structured decision objects instead of loose notes or hidden chat output.",
    icon: FileText,
  },
  {
    label: "04",
    title: "Approve",
    description: "Important portfolio and research changes pause for human review before they become official.",
    icon: ShieldCheck,
  },
  {
    label: "05",
    title: "Record",
    description: "The system writes the final version with evidence, timing, actor, and provenance attached.",
    icon: GitBranch,
  },
]

const trustPanels: {
  title: string
  description: string
  icon: LucideIcon
}[] = [
  {
    title: "Evidence stays attached",
    description:
      "Research outputs can point back to the market data, documents, workflow runs, and snapshots that supported them.",
    icon: FileText,
  },
  {
    title: "History is versioned",
    description:
      "The ontology records what Talisman believed at the time, then preserves later corrections as new versions.",
    icon: GitBranch,
  },
  {
    title: "Automation has guardrails",
    description:
      "Agents can draft and propose, but governed actions require approval before changing user-visible state.",
    icon: Lock,
  },
]

function FeatureShowcase() {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 pb-24 sm:px-6">
      <div className="mb-12 text-center">
        <h2 className="page-title text-center !text-[clamp(1.7rem,1.4rem+1vw,2.3rem)]">
          Everything you need to stay on top of the markets
        </h2>
        <p className="mx-auto mt-3 max-w-2xl body-copy">
          Professional-grade analytics, all in one place.
        </p>
      </div>

      <div className="flex flex-col gap-16">
        {features.map((feature, i) => (
          <div
            key={feature.title}
            className={`theme-surface flex flex-col items-center gap-8 overflow-hidden p-4 sm:p-6 md:flex-row ${i % 2 === 1 ? "md:flex-row-reverse" : ""}`}
          >
            <img
              src={feature.image}
              alt={feature.title}
              className="w-full rounded-[1.2rem] border border-app object-cover md:w-3/5"
            />
            <div className="flex flex-col gap-2 md:w-2/5">
              <h3 className="text-xl font-semibold tracking-[-0.02em] text-app">{feature.title}</h3>
              <p className="body-copy">{feature.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function DiagramNode({
  title,
  description,
  icon: Icon,
}: {
  title: string
  description: string
  icon: LucideIcon
}) {
  return (
    <div className="theme-surface flex min-h-[11rem] flex-col gap-3 p-4">
      <div className="flex h-11 w-11 items-center justify-center rounded-full border border-app bg-card text-[hsl(var(--accent))]">
        <Icon className="h-5 w-5" aria-hidden="true" />
      </div>
      <div>
        <h4 className="text-base font-semibold text-app">{title}</h4>
        <p className="mt-1 text-sm leading-6 text-muted">{description}</p>
      </div>
    </div>
  )
}

function ArchitectureShowcase() {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 pb-24 sm:px-6">
      <div className="mb-10 text-center">
        <p className="theme-eyebrow">Backend architecture</p>
        <h2 className="page-title text-center !text-[clamp(1.7rem,1.4rem+1vw,2.3rem)]">
          A trust engine for investment research
        </h2>
        <p className="mx-auto mt-3 max-w-3xl body-copy">
          Talisman is designed like a research desk with memory. The front end feels simple, but behind every
          dashboard is a governed system that gathers evidence, runs deeper analysis in the background, and keeps
          a clear record of how decisions were formed.
        </p>
      </div>

      <div className="flex flex-col gap-10">
        <section className="grid gap-5 lg:grid-cols-[0.95fr_1.35fr] lg:items-center">
          <div>
            <p className="theme-eyebrow">System map</p>
            <h3 className="text-2xl font-semibold tracking-[-0.03em] text-app">From one click to a research system</h3>
            <p className="mt-3 body-copy">
              A visitor sees a clean dashboard. The signed-in app sees a production pipeline: requests are routed,
              jobs are dispatched, state is stored durably, and every important output can be traced.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {systemMapNodes.map((node, index) => (
              <div key={node.title} className="relative">
                <DiagramNode {...node} />
                {index < systemMapNodes.length - 1 && (
                  <ArrowRight
                    className="absolute -right-2 top-1/2 z-10 hidden h-4 w-4 -translate-y-1/2 rounded-full bg-app text-subtle xl:block"
                    aria-hidden="true"
                  />
                )}
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="mb-6 max-w-3xl">
            <p className="theme-eyebrow">Research loop</p>
            <h3 className="text-2xl font-semibold tracking-[-0.03em] text-app">Slow work moves off the main path</h3>
            <p className="mt-3 body-copy">
              Heavy analysis does not block the interface. Durable async jobs and warm worker pools handle portfolio
              sizing, agent turns, ontology queries, and deeper model runs while the app keeps showing progress.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-5">
            {researchLoopSteps.map((step, index) => {
              const Icon = step.icon
              return (
                <div key={step.title} className="theme-surface relative p-4">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <span className="rounded-full border border-app bg-card px-2.5 py-1 text-xs font-semibold text-subtle">
                      {step.label}
                    </span>
                    <Icon className="h-5 w-5 text-[hsl(var(--accent))]" aria-hidden="true" />
                  </div>
                  <h4 className="text-base font-semibold text-app">{step.title}</h4>
                  <p className="mt-2 text-sm leading-6 text-muted">{step.description}</p>
                  {index < researchLoopSteps.length - 1 && (
                    <ArrowRight
                      className="absolute -right-2 top-1/2 z-10 hidden h-4 w-4 -translate-y-1/2 rounded-full bg-app text-subtle xl:block"
                      aria-hidden="true"
                    />
                  )}
                </div>
              )
            })}
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
          <div>
            <p className="theme-eyebrow">Why it matters</p>
            <h3 className="text-2xl font-semibold tracking-[-0.03em] text-app">AI is useful only when it is accountable</h3>
            <p className="mt-3 body-copy">
              Talisman treats generated research as a proposal, not a command. The ontology and provenance layers
              turn analysis into structured objects with evidence, timestamps, approvals, and version history.
            </p>
          </div>

          <div className="grid gap-3">
            {trustPanels.map(panel => {
              const Icon = panel.icon
              return (
                <div key={panel.title} className="theme-surface flex gap-3 p-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-app bg-card text-[hsl(var(--accent))]">
                    <Icon className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div>
                    <h4 className="text-base font-semibold text-app">{panel.title}</h4>
                    <p className="mt-1 text-sm leading-6 text-muted">{panel.description}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[0.95fr_1.35fr] lg:items-center">
          <div>
            <p className="theme-eyebrow">Production backbone</p>
            <h3 className="text-2xl font-semibold tracking-[-0.03em] text-app">Built for research that compounds</h3>
            <p className="mt-3 body-copy">
              The system is not a collection of one-off scripts. It is a durable operating layer where dashboards,
              documents, workflows, jobs, and approvals share the same memory.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            {[
              "Cloud Run scales the API and job runners.",
              "Cloud SQL stores structured portfolio and workflow state.",
              "Cloud Storage preserves generated documents and artifacts.",
              "Cloud Scheduler refreshes market snapshots and maintenance jobs.",
            ].map(item => (
              <div key={item} className="theme-surface flex min-h-[5rem] gap-2 p-3 text-sm leading-6 text-muted">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-positive" aria-hidden="true" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}

function PublicShowcase() {
  const [activeTab, setActiveTab] = useState<PublicTab>("product")

  return (
    <div className="w-full">
      <div className="mb-10 flex justify-center px-4 sm:px-6">
        <SegmentedControl options={publicTabs} value={activeTab} onChange={setActiveTab} />
      </div>
      {activeTab === "product" ? <FeatureShowcase /> : <ArchitectureShowcase />}
    </div>
  )
}

export function LoginPage() {
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
    <div className="min-h-screen bg-app text-app">
      <div className="theme-page">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-16 pb-16 pt-10 md:pt-16">
          <SurfaceCard className="w-full max-w-md p-6 sm:p-8">
            <p className="theme-eyebrow mb-3">Welcome back</p>
            <h1 className="mb-1 text-[2rem] font-semibold tracking-[-0.04em] text-app">Market Dashboard</h1>

            {mode === "cloudflare" ? (
              <>
                <p className="mb-6 body-copy">
                  Sign-in is handled by Cloudflare Access.
                </p>
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
                <p className="mb-6 body-copy">Enter your password to continue.</p>
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

                  {error && (
                    <Notice tone="error">{error}</Notice>
                  )}

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
          </SurfaceCard>

          <PublicShowcase />
        </div>
      </div>
    </div>
  )
}
