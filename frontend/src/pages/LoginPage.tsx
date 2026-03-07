import { type FormEvent, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAuth } from "@/contexts/AuthContext"

import portfolioImg from "@/assets/portfolio.png"
import centralBankImg from "@/assets/central-bank-monitor.png"
import economicGrowthImg from "@/assets/economic-growth.jpg"
import industryImg from "@/assets/industry.png"

const features = [
  {
    title: "Portfolio Dashboard",
    description:
      "Track your entire portfolio at a glance — real-time P&L, individual position charts, and performance across daily, weekly, and monthly timeframes.",
    image: portfolioImg,
  },
  {
    title: "Economic Growth Dashboard",
    description:
      "Monitor the macro cycle in real time — commodity trends, regional growth signals, leading indicators, and cross-asset momentum to gauge where the economy is heading.",
    image: economicGrowthImg,
  },
  {
    title: "Industry Monitor",
    description:
      "Drill into sector-level fundamentals with earnings call transcripts, company-level financial snapshots, and AI-generated summaries across housing, trucking, retail, and more.",
    image: industryImg,
  },
  {
    title: "Central Bank Monitor",
    description:
      "Stay ahead of monetary policy — AI-generated summaries of central bank decisions, rate changes, and GDP forecasts from the Fed, ECB, and more.",
    image: centralBankImg,
  },
]

function FeatureShowcase() {
  return (
    <div className="mx-auto w-full max-w-5xl px-6 pb-24">
      <div className="mb-12 text-center">
        <h2 className="text-2xl font-semibold text-app">
          Everything you need to stay on top of the markets
        </h2>
        <p className="mt-2 text-sm text-muted">
          Professional-grade analytics, all in one place.
        </p>
      </div>

      <div className="flex flex-col gap-20">
        {features.map((feature, i) => (
          <div
            key={feature.title}
            className={`flex flex-col items-center gap-8 md:flex-row ${i % 2 === 1 ? "md:flex-row-reverse" : ""}`}
          >
            <img
              src={feature.image}
              alt={feature.title}
              className="w-full rounded-xl border border-app shadow-sm md:w-3/5"
            />
            <div className="flex flex-col gap-2 md:w-2/5">
              <h3 className="text-lg font-semibold text-app">{feature.title}</h3>
              <p className="text-sm leading-relaxed text-muted">{feature.description}</p>
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
      <div className="flex flex-col items-center pt-24 pb-16">
        <div className="theme-surface w-full max-w-sm rounded-xl p-8">
          <h1 className="mb-1 text-xl font-semibold text-app">Market Dashboard</h1>

          {mode === "cloudflare" ? (
            <>
              <p className="mb-6 text-sm text-muted">
                Sign-in is handled by Cloudflare Access.
              </p>
              <button
                type="button"
                onClick={() => login("")}
                className="theme-button-primary w-full rounded-lg px-4 py-2 text-sm font-semibold"
              >
                Continue
              </button>
            </>
          ) : (
            <>
              <p className="mb-6 text-sm text-muted">Enter your password to continue.</p>
              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="password" className="text-sm font-medium text-app">
                    Password
                  </label>
                  <input
                    id="password"
                    type="password"
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="theme-input rounded-lg px-3 py-2 text-sm"
                  />
                </div>

                {error && (
                  <p className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="theme-button-primary rounded-lg px-4 py-2 text-sm font-semibold disabled:opacity-50"
                >
                  {loading ? "Signing in..." : "Sign in"}
                </button>
              </form>
            </>
          )}
        </div>
      </div>

      <FeatureShowcase />
    </div>
  )
}
