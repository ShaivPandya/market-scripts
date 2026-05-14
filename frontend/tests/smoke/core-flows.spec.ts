import { authenticate, expect, test } from "./fixtures"

test("redirects protected routes to login and signs in with password auth", async ({ page }) => {
  await page.goto("/workspace")

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole("heading", { name: "Market Dashboard" })).toBeVisible()
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
  await expect(page.getByText("Market Regime")).toBeVisible()
  await expect(page.getByText("Risk-on")).toBeVisible()
  await expect(page.getByText("Portfolio Risk")).toBeVisible()
  await expect(page.getByRole("heading", { name: /Pending Approvals/ })).toBeVisible()

  await page.getByRole("button", { name: "Review" }).click()

  await expect(page.getByRole("dialog", { name: "Review Approval" })).toBeVisible()
  await expect(page.getByText("Create internal action item")).toBeVisible()

  const approveButton = page.getByRole("button", { name: "Approve & Apply" })
  await expect(approveButton).toBeDisabled()

  await page.getByLabel("Decision note").fill("Reviewed staged research follow-up for smoke coverage.")
  await expect(approveButton).toBeEnabled()
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
})
