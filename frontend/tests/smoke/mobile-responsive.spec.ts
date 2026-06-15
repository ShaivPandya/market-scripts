import { authenticate, expect, test } from "./fixtures"
import { dismissFloatingAlerts, expectDecisionTraceDrawer, expectPrimaryControlsSeparated } from "./operatingWorkflow"

test.describe("mobile responsive surfaces", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  })

  test("shows mobile navigation chrome and opens Stan in compact mode", async ({ page }) => {
    await authenticate(page)
    await page.goto("/")

    await expect(page.getByRole("button", { name: "Open navigation" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Open Stan" }).first()).toBeVisible()

    await page.getByRole("button", { name: "Open Stan" }).first().click()
    await expect(page.getByRole("dialog", { name: "Stan" })).toBeVisible()
    await expect(page.getByPlaceholder("Ask about markets, portfolio, macro...")).toBeVisible()
    await page.getByRole("button", { name: "Workflows" }).click()
    await expect(page.getByRole("button", { name: "Weekly Portfolio Review" })).toBeVisible()
  })

  test("opens mobile sidebar and navigates via workflow link", async ({ page }) => {
    await authenticate(page)
    await page.goto("/")

    await dismissFloatingAlerts(page)

    await page.getByRole("button", { name: "Open navigation" }).click()

    const nav = page.getByRole("navigation", { name: "Primary navigation" })
    await expect(nav.getByText("Core", { exact: true })).toBeVisible()
    await dismissFloatingAlerts(page)
    await nav.getByRole("link", { name: "Workspace" }).click()

    await expect(page).toHaveURL(/\/workspace$/)
    await expect(page.getByText("Here's where your book stands and the few things worth your attention.")).toBeVisible()
    await expect(page.locator(".theme-floating").getByText("Workspace", { exact: true })).toBeVisible()
  })

  test("reviews workspace approvals with stacked dialog actions", async ({ page }) => {
    await authenticate(page)
    await page.goto("/workspace")

    await expect(page.getByRole("heading", { name: "Action queue" })).toBeVisible()
    await page.getByRole("button", { name: "Review", exact: true }).click()

    const reviewDialog = page.getByRole("dialog", { name: "Review Approval" })
    await expect(reviewDialog).toBeVisible()
    await expect(reviewDialog.getByRole("button", { name: "Approve & Apply" })).toBeDisabled()

    await page.getByLabel("Decision note").fill("Reviewed staged research follow-up for mobile smoke coverage.")
    const approveButton = reviewDialog.getByRole("button", { name: "Approve & Apply" })
    const rejectButton = reviewDialog.getByRole("button", { name: "Reject Proposal" })
    await expect(approveButton).toBeEnabled()
    await expect(rejectButton).toBeVisible()
    await expectPrimaryControlsSeparated(approveButton, rejectButton)
  })

  test("renders OpportunityScout queue and opens decision trace on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/workspace")

    await expect(page.getByRole("heading", { name: "OpportunityScout" })).toBeVisible()
    await expect(page.getByText("Kill condition monitor hit: Margin compression threshold")).toBeVisible()

    await page
      .getByRole("heading", { name: "OpportunityScout" })
      .locator("xpath=ancestor::section[1]")
      .getByRole("button", { name: "Trace", exact: true })
      .click()
    await expectDecisionTraceDrawer(page.getByRole("dialog", { name: "Decision Trace" }))
  })

  test("opens decision trace from workspace approval on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/workspace")

    await page.getByRole("button", { name: "View approval smoke-approval trace" }).click()
    await expectDecisionTraceDrawer(page.getByRole("dialog", { name: "Decision Trace" }))
  })

  test("runs dossier pressure test and scenario comparison on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/dossier/MSFT")

    await page.getByRole("button", { name: "Workflows", exact: true }).click()
    await expect(page.getByRole("button", { name: "Run pressure test" })).toBeVisible()
    await page.getByRole("button", { name: "Run pressure test" }).click()
    await expect(page.getByRole("dialog", { name: "Stan" })).toBeVisible()
    await page.getByRole("button", { name: "Close Stan" }).click()

    await page.getByRole("button", { name: "Scenarios", exact: true }).click()
    await expect(page.getByRole("heading", { name: "Action Scenario Comparison" })).toBeVisible()
    await page.getByRole("button", { name: "Run comparison" }).click()
    await expect(page.getByText("medium uncertainty")).toBeVisible()
  })

  test("stages monitor definition from ontology builder on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/ontology")

    await expect(page.getByRole("heading", { name: "Monitor And Mission Builder" })).toBeVisible()
    await page.getByRole("textbox", { name: "Name" }).fill("Mobile Smoke Monitor")
    await page.getByRole("textbox", { name: "Ticker Scope" }).fill("MSFT")
    await page.getByRole("textbox", { name: "Condition", exact: true }).fill("Watch MSFT evidence quality")
    await page.getByRole("button", { name: "Preview Monitor" }).click()
    await expect(page.getByText("Smoke preview")).toBeVisible()
    await page.getByRole("button", { name: "Stage Definition" }).click()
    await expect(page.getByText("Definition staged for approval")).toBeVisible()
  })

  test("runs historical ontology query with temporal context on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/ontology")

    await page.getByRole("button", { name: "Historical" }).click()
    await page.getByRole("textbox", { name: "As of", exact: true }).fill("2026-05-10T09:30")
    await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks as of last review")
    await page.getByRole("button", { name: "Run Query" }).click()

    const temporalContext = page.getByRole("region", { name: "Temporal query context" })
    await expect(temporalContext).toBeVisible()
    await expect(temporalContext.getByText("History Included")).toBeVisible()
    await expect(page.getByText("Risk Analysis Results")).toBeVisible()
  })

  test("runs ontology workbench query with card-friendly results", async ({ page }) => {
    await authenticate(page)
    await page.goto("/ontology")

    await expect(page.getByRole("heading", { name: "Ontology Workbench" })).toBeVisible()
    await page.getByPlaceholder("Which positions are in deteriorating macro conditions?").fill("Show elevated portfolio risks")
    await page.getByRole("button", { name: "Run Query" }).click()

    await expect(page.getByText("Risk Analysis Results")).toBeVisible()
    await expect(page.getByText("Liquidity impulse")).toBeVisible()
    await expect(page.getByText("Source Health And Staleness")).toBeVisible()
  })

  test("renders dossier tabs and evidence ledger on mobile", async ({ page }) => {
    await authenticate(page)
    await page.goto("/dossier/MSFT")

    await expect(page.getByRole("heading", { name: "MSFT" })).toBeVisible()
    await page.getByRole("button", { name: "Evidence", exact: true }).click()

    await expect(page.getByRole("heading", { name: "Evidence Ledger" })).toBeVisible()
    await expect(page.getByText("AI capex remains durable")).toBeVisible()
    await expect(page.getByRole("button", { name: "Trace", exact: true }).first()).toBeVisible()
  })
})
