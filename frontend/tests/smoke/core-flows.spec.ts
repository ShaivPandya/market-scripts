import { authenticate, expect, test } from "./fixtures"
import { expectDecisionTraceDrawer, expectPrimaryControlsSeparated } from "./operatingWorkflow"

test.describe("password auth entry", () => {
  test.use({ authenticated: false })

  test("redirects protected routes to login and signs in with password auth", async ({ page }) => {
    await page.goto("/workspace")

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole("heading", { name: "Talisman" })).toBeVisible()
    await expect(page.getByText("Sign in with your username and password.")).toBeVisible()

    await page.getByLabel("Password").fill("smoke-password")
    await page.getByRole("button", { name: "Sign in" }).click()

    await expect(page).toHaveURL(/\/$/)
    await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()
  })
})

test("renders the portfolio dashboard happy path with deterministic holdings", async ({ page }) => {
  await authenticate(page)
  await page.goto("/")

  await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()
  await expect(page.getByRole("button", { name: "Past Week" })).toBeVisible()
  await expect(page.getByRole("button", { name: "Unified" })).toBeVisible()
  await expect(page.getByText("AI Infrastructure", { exact: true })).toBeVisible()
  await expect(page.getByRole("link", { name: "MSFT" })).toBeVisible()
  await expect(page.getByRole("link", { name: "NVDA" })).toBeVisible()
})

test("renders workspace common operating picture and enforces approval note gating", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByText("Here's where your book stands and the few things worth your attention.")).toBeVisible()
  await expect(page.getByText("Today's read")).toBeVisible()
  await expect(page.getByText(/Bullish regime/)).toBeVisible()
  await expect(page.getByRole("heading", { name: "Action queue" })).toBeVisible()
  await expect(page.getByText("Recommended action shifted from hold to trim after risk review.")).toBeVisible()
  await expect(page.getByRole("heading", { name: "Portfolio risk" })).toBeVisible()
  const recommendationReview = page.locator("section").filter({
    has: page.getByRole("heading", { name: /Recommendation Review/ }),
  })
  await expect(recommendationReview).toBeVisible()
  await expect(recommendationReview.getByText("Policy Missing")).toHaveCount(0)
  await expect(page.getByRole("heading", { name: /Thesis Claim Issues/ })).toBeVisible()
  await expect(page.getByText("AI infrastructure demand remains durable through 2026.")).toBeVisible()
  await expect(page.getByRole("heading", { name: "Timeline" })).toBeVisible()
  await expect(page.getByText("1/2 approvals recorded")).toBeVisible()
  await expect(page.getByText("Portfolio manager (portfolio:default)")).toBeVisible()

  await expect(page.getByText("Source health", { exact: true })).toBeVisible()
  await expect(page.getByText(/SLA breach/).first()).toBeVisible()

  await page.getByRole("button", { name: "Review", exact: true }).click()

  const reviewDialog = page.getByRole("dialog", { name: "Review Approval" })
  await expect(reviewDialog).toBeVisible()
  await expect(reviewDialog.getByText("Create internal action item")).toBeVisible()
  await expect(reviewDialog.getByText("standard source needs review")).toBeVisible()

  const approveButton = reviewDialog.getByRole("button", { name: "Approve & Apply" })
  const rejectButton = reviewDialog.getByRole("button", { name: "Reject Proposal" })
  await expect(approveButton).toBeDisabled()
  await expectPrimaryControlsSeparated(approveButton, rejectButton)

  await page.getByLabel("Decision note").fill("Reviewed staged research follow-up for smoke coverage.")
  await expect(approveButton).toBeEnabled()
})

test("opens approval review from workspace approval_id deep link", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace?approval_id=smoke-approval")

  await expect(page).toHaveURL(/\/workspace(?:\?approval_id=smoke-approval)?$/)
  await expect(page.getByRole("dialog", { name: "Review Approval" })).toBeVisible()
  await expect(page.getByText("Create internal action item")).toBeVisible()
})

test("clears workspace pressure rows and bulk dismisses approvals", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByRole("heading", { name: "Action queue" })).toBeVisible()
  await expect(page.getByText("AI capex concentration")).toBeVisible()

  await page.getByRole("button", { name: "Clear MSFT pressure row" }).click()
  await expect(page.getByText("AI capex concentration")).toBeHidden()

  await page.getByRole("button", { name: "Dismiss all", exact: true }).click()
  const dialog = page.getByRole("dialog", { name: "Dismiss All Pending Approvals" })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByText("rejects every currently pending approval")).toBeVisible()

  await dialog.getByRole("button", { name: "Dismiss All", exact: true }).click()
  await expect(dialog).toBeHidden()
  await expect(page.getByRole("button", { name: "Dismiss all", exact: true })).toHaveCount(0)
})

test("shows decision learning review queue on workspace", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByRole("heading", { name: "Decision Learning" })).toBeVisible()
  await expect(page.getByText("Smoke draft post-mortem for review.")).toBeVisible()

  await page.getByRole("button", { name: "Review post-mortem" }).click()
  const dialog = page.getByRole("dialog", { name: "Review Post-Mortem" })
  await expect(dialog).toBeVisible()
  await dialog.getByRole("button", { name: "Finalize" }).click()
  await expect(dialog).toBeHidden()
})

test("renders OpportunityScout queue and stages dismiss feedback", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByRole("heading", { name: "OpportunityScout" })).toBeVisible()
  await expect(page.getByText("Kill condition monitor hit: Margin compression threshold")).toBeVisible()
  await expect(page.getByText("Why now:")).toBeVisible()
  await expect(page.getByText("Missing:")).toBeVisible()
  await expect(page.getByRole("link", { name: "NVDA" }).first()).toBeVisible()

  await page.getByRole("button", { name: "Dismiss NVDA" }).click()
  await expect(page.getByText("Dismiss staged. Approval is required before app state changes.")).toBeVisible()
})

test("runs an ontology workbench query with mocked async results", async ({ page }) => {
  await authenticate(page)
  await page.goto("/ontology")

  await expect(page.getByRole("heading", { name: "Ontology Workbench" })).toBeVisible()
  await expect(page.getByRole("heading", { name: "Monitor And Mission Builder" })).toBeVisible()

  await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks")
  await page.getByRole("button", { name: "Run Query" }).click()

  await expect(page.getByText("Risk Analysis Results")).toBeVisible()
  await expect(page.getByText("ontology-smoke-run")).toBeVisible()
  await expect(page.getByRole("cell", { name: "MSFT" })).toBeVisible()
  await expect(page.getByText("Liquidity impulse")).toBeVisible()
  await expect(page.getByText("Source Health And Staleness")).toBeVisible()
})

test("stages a low-code monitor from ontology workbench", async ({ page }) => {
  await authenticate(page)
  await page.goto("/ontology")

  await page.getByRole("textbox", { name: "Name" }).fill("Smoke Custom Monitor")
  await page.getByRole("textbox", { name: "Ticker Scope" }).fill("MSFT")
  await page.getByRole("textbox", { name: "Condition", exact: true }).fill("Watch MSFT evidence quality")
  await page.getByRole("button", { name: "Preview Monitor" }).click()
  await expect(page.getByText("Smoke preview")).toBeVisible()
  await page.getByRole("button", { name: "Stage Definition" }).click()
  await expect(page.getByText("Definition staged for approval")).toBeVisible()
})

test("runs a historical ontology workbench query with temporal context", async ({ page, apiMocks }) => {
  await authenticate(page)
  await page.goto("/ontology")

  await expect(page.getByRole("heading", { name: "Ontology Workbench" })).toBeVisible()

  await page.getByRole("button", { name: "Historical" }).click()
  await page.getByRole("textbox", { name: "As of", exact: true }).fill("2026-05-10T09:30")
  await page.getByRole("switch", { name: /Include history/ }).click()
  await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks as of last review")
  await page.getByRole("button", { name: "Run Query" }).click()

  const temporalContext = page.getByRole("region", { name: "Temporal query context" })
  await expect(temporalContext).toBeVisible()
  await expect(temporalContext.getByText("Temporal Context")).toBeVisible()
  await expect(temporalContext.getByText("temporal_read_model")).toBeVisible()
  await expect(temporalContext.getByText("History Included")).toBeVisible()
  await expect(temporalContext.getByText("Yes")).toBeVisible()
  await expect(page.getByRole("cell", { name: "MSFT" })).toBeVisible()

  expect(typeof apiMocks.ontologyQueryRequest?.as_of).toBe("string")
  expect(apiMocks.ontologyQueryRequest?.include_history).toBe(true)
})

test("renders liquidity data-quality warning and formatted screen context", async ({ page }) => {
  await authenticate(page)
  await page.goto("/liquidity")

  await expect(page.getByRole("heading", { name: "Liquidity Dashboard" })).toBeVisible()
  await expect(page.getByText("Liquidity data quality degraded")).toBeVisible()
  await expect(page.getByText("Suppressed partial weekly bucket ending 2026-05-20")).toBeVisible()
  await expect(page.getByText("2026-05-13", { exact: true })).toBeVisible()

  await page.getByRole("button", { name: "Open Stan" }).click()

  await expect(page.getByRole("dialog", { name: "Stan" })).toBeVisible()
  await expect(page.getByText("Regional Scores")).toBeVisible()
  await expect(page.getByText("us: +0.05, europe: -0.37, japan: +0.47")).toBeVisible()
})

test("renders first-turn live agent chat response without switching sessions", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByText("Here's where your book stands and the few things worth your attention.")).toBeVisible()

  await page.getByRole("button", { name: "Open Stan" }).click()
  await expect(page.getByRole("dialog")).toBeVisible()
  await expect(page.getByText("Stan is ready")).toBeVisible()

  await page.getByRole("textbox", { name: "Message Stan" }).fill("Summarize my portfolio's performance")
  await page.getByRole("button", { name: "Send message" }).click()

  await expect(page.getByText("Portfolio summary smoke response.")).toBeVisible({ timeout: 10_000 })
  await expect(page.getByText("Stan is ready")).toBeHidden()
})

test("opens the agent chat shell with workflow and preference fixtures", async ({ page }) => {
  await authenticate(page)
  await page.goto("/")

  await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()

  await page.getByRole("button", { name: "Open Stan" }).click()

  await expect(page.getByRole("dialog", { name: "Stan" })).toBeVisible()
  await expect(page.getByText("Stan is ready")).toBeVisible()
  await expect(page.getByPlaceholder("Ask about markets, portfolio, macro...")).toBeVisible()
  await expect(page.getByRole("heading", { name: "Workflows" })).toBeVisible()
  await expect(page.getByRole("button", { name: "Weekly Portfolio Review" })).toBeVisible()

  await page.getByRole("button", { name: "Open conversation history" }).click()
  await expect(page.getByRole("heading", { name: "Today" })).toBeVisible()
  await expect(page.getByText("NVDA Earnings Prep").first()).toBeVisible()

  await page.getByRole("button", { name: "Rename NVDA Earnings Prep" }).click()
  await page.getByRole("textbox", { name: "Conversation title" }).fill("Renamed NVDA Chat")
  await page.keyboard.press("Enter")
  await expect(page.getByText("Renamed NVDA Chat").first()).toBeVisible()
})

test("renders position dossier scenario comparison disclosure", async ({ page }) => {
  await authenticate(page)
  await page.goto("/dossier/MSFT")

  await expect(page.getByRole("heading", { name: "MSFT" })).toBeVisible()
  await page.getByRole("button", { name: "Scenarios", exact: true }).click()

  await expect(page.getByRole("heading", { name: "Action Scenario Comparison" })).toBeVisible()
  await expect(page.getByText("Scenario simulation is decision support only")).toBeVisible()
  await page.getByRole("button", { name: "Run comparison" }).click()
  await expect(page.getByText("medium uncertainty")).toBeVisible()
  await expect(page.getByText("Liquidity missing")).toBeVisible()
  await expect(page.getByText("No automatic trade recommendation or execution is produced by the simulator.")).toBeVisible()
})

test("renders position dossier evidence ledger tab", async ({ page }) => {
  await authenticate(page)
  await page.goto("/dossier/MSFT")

  await expect(page.getByRole("heading", { name: "MSFT" })).toBeVisible()
  await page.getByRole("button", { name: "Evidence", exact: true }).click()

  await expect(page.getByRole("heading", { name: "Evidence Ledger" })).toBeVisible()
  await expect(page.getByText("AI capex remains durable")).toBeVisible()
  await expect(page.getByText("Azure growth re-accelerated in latest quarter")).toBeVisible()
  await expect(page.getByRole("button", { name: "Trace", exact: true }).first()).toBeVisible()
  await expect(page.getByRole("link", { name: "Weekly report" })).toBeVisible()
})

test("opens dossier pressure test workflow and records prior run trace", async ({ page }) => {
  await authenticate(page)
  await page.goto("/dossier/MSFT")

  await expect(page.getByRole("heading", { name: "MSFT" })).toBeVisible()
  await page.getByRole("button", { name: "Workflows", exact: true }).click()

  await expect(page.getByText("Position Dossier Pressure Test", { exact: true })).toBeVisible()
  await expect(page.getByText("position dossier pressure test", { exact: true })).toBeVisible()
  await page.getByRole("button", { name: "Run pressure test" }).click()
  await expect(page.getByRole("dialog", { name: "Stan" })).toBeVisible()

  await page.getByRole("button", { name: "Close Stan" }).click()
  await page.getByRole("button", { name: "Trace workflow dossier-workflow-smoke" }).click()
  await expectDecisionTraceDrawer(page.getByRole("dialog", { name: "Decision Trace" }))
})

test("opens workflow trace from workspace timeline", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByRole("heading", { name: "Timeline" })).toBeVisible()
  await page.getByRole("button", { name: "View workflow workflow-smoke trace" }).click()
  await expectDecisionTraceDrawer(page.getByRole("dialog", { name: "Decision Trace" }))
})

test("opens ontology run lineage trace after temporal query", async ({ page }) => {
  await authenticate(page)
  await page.goto("/ontology")

  await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks")
  await page.getByRole("button", { name: "Run Query" }).click()
  await expect(page.getByText("Risk Analysis Results")).toBeVisible()

  await page.getByRole("button", { name: "Trace", exact: true }).click()
  const drawer = page.getByRole("dialog", { name: "Decision Trace" })
  await expect(drawer).toBeVisible()
  await expect(drawer.getByRole("heading", { name: "Provenance" })).toBeVisible()
})

test("opens unified decision trace drawer from workspace approval", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await page.getByRole("button", { name: "View approval smoke-approval trace" }).click()
  await expectDecisionTraceDrawer(page.getByRole("dialog", { name: "Decision Trace" }))
})

test("opens decision trace drawer from OpportunityScout and dossier evidence", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await page.getByRole("heading", { name: "OpportunityScout" }).locator("xpath=ancestor::section[1]").getByRole("button", { name: "Trace", exact: true }).click()
  const scoutDrawer = page.getByRole("dialog", { name: "Decision Trace" })
  await expect(scoutDrawer).toBeVisible()
  await expect(scoutDrawer.getByRole("heading", { name: "NVDA" })).toBeVisible()
  await page.keyboard.press("Escape")
  await expect(scoutDrawer).toBeHidden()

  await page.goto("/dossier/MSFT")
  await page.getByRole("button", { name: "Evidence", exact: true }).click()
  await page.getByRole("button", { name: "Trace", exact: true }).first().click()
  const evidenceDrawer = page.getByRole("dialog", { name: "Decision Trace" })
  await expect(evidenceDrawer).toBeVisible()
  await expect(evidenceDrawer.getByRole("heading", { name: "Provenance" })).toBeVisible()
})
