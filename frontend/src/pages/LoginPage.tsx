import { type FormEvent, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAuth } from "@/contexts/AuthContext"
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

          <FeatureShowcase />
        </div>
      </div>
    </div>
  )
}
