import { authenticate, expect, test } from "./fixtures"

test("redirects protected routes to login and signs in with password auth", async ({ page }) => {
  await page.goto("/workspace")

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole("heading", { name: "Talisman" })).toBeVisible()
  await expect(page.getByText("Enter your password to continue.")).toBeVisible()

  await page.getByLabel("Password").fill("smoke-password")
  await page.getByRole("button", { name: "Sign in" }).click()

  await expect(page).toHaveURL(/\/$/)
  await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()
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

  await expect(page.getByRole("heading", { name: "Workspace" })).toBeVisible()
  await expect(page.getByText("Market Regime", { exact: true })).toBeVisible()
  await expect(page.getByText("Risk-on")).toBeVisible()
  await expect(page.getByText("Portfolio Risk")).toBeVisible()
  await expect(page.getByRole("heading", { name: /Pending Approvals/ })).toBeVisible()
  await expect(page.getByText("1/2 approvals recorded")).toBeVisible()
  await expect(page.getByText("Portfolio manager (portfolio:default)")).toBeVisible()

  await expect(page.getByRole("heading", { name: "Source Health" })).toBeVisible()
  await expect(page.getByText("1 critical stale")).toBeVisible()
  await expect(page.getByText("SLA breach").first()).toBeVisible()
  await expect(page.getByText("Critical").first()).toBeVisible()

  await page.getByRole("button", { name: "Review" }).click()

  const reviewDialog = page.getByRole("dialog", { name: "Review Approval" })
  await expect(reviewDialog).toBeVisible()
  await expect(reviewDialog.getByText("Create internal action item")).toBeVisible()
  await expect(reviewDialog.getByText("standard source needs review")).toBeVisible()

  const approveButton = reviewDialog.getByRole("button", { name: "Approve & Apply" })
  await expect(approveButton).toBeDisabled()

  await page.getByLabel("Decision note").fill("Reviewed staged research follow-up for smoke coverage.")
  await expect(approveButton).toBeEnabled()
})

test("clears workspace pressure rows and bulk dismisses approvals", async ({ page }) => {
  await authenticate(page)
  await page.goto("/workspace")

  await expect(page.getByRole("heading", { name: "Positions Under Pressure" })).toBeVisible()
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

test("runs an ontology workbench query with mocked async results", async ({ page }) => {
  await authenticate(page)
  await page.goto("/ontology")

  await expect(page.getByRole("heading", { name: "Ontology Workbench" })).toBeVisible()

  await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks")
  await page.getByRole("button", { name: "Run Query" }).click()

  await expect(page.getByText("Risk Analysis Results")).toBeVisible()
  await expect(page.getByText("ontology-smoke-run")).toBeVisible()
  await expect(page.getByRole("cell", { name: "MSFT" })).toBeVisible()
  await expect(page.getByText("Liquidity impulse")).toBeVisible()
  await expect(page.getByText("Source Health And Staleness")).toBeVisible()
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

test("renders position dossier evidence ledger tab", async ({ page }) => {
  await authenticate(page)
  await page.goto("/dossier/MSFT")

  await expect(page.getByRole("heading", { name: "MSFT" })).toBeVisible()
  await page.getByRole("button", { name: "Evidence", exact: true }).click()

  await expect(page.getByRole("heading", { name: "Evidence Ledger" })).toBeVisible()
  await expect(page.getByText("AI capex remains durable")).toBeVisible()
  await expect(page.getByText("Azure growth re-accelerated in latest quarter")).toBeVisible()
  await expect(page.getByRole("button", { name: "Lineage" })).toBeVisible()
  await expect(page.getByRole("link", { name: "Weekly report" })).toBeVisible()
})
