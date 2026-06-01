import { authenticate, expect, test } from "./fixtures"

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

    const approvalAlert = page.getByRole("status").filter({ hasText: "Action item staged" })
    if (await approvalAlert.isVisible()) {
      await approvalAlert.getByRole("button", { name: "Dismiss action item alert" }).click()
    }

    await page.getByRole("button", { name: "Open navigation" }).click()

    const nav = page.getByRole("navigation", { name: "Primary navigation" })
    await expect(nav.getByText("Command Portfolio")).toBeVisible()
    await nav.getByRole("link", { name: "Portfolio Commander" }).click()

    await expect(page).toHaveURL(/\/workspace$/)
    await expect(page.getByRole("heading", { name: "Portfolio Commander" })).toBeVisible()
    await expect(page.locator(".theme-floating").getByText("Portfolio Commander", { exact: true })).toBeVisible()
  })

  test("reviews workspace approvals with stacked dialog actions", async ({ page }) => {
    await authenticate(page)
    await page.goto("/workspace")

    await expect(page.getByRole("heading", { name: "Portfolio Commander" })).toBeVisible()
    await page.getByRole("button", { name: "Review", exact: true }).click()

    const reviewDialog = page.getByRole("dialog", { name: "Review Approval" })
    await expect(reviewDialog).toBeVisible()
    await expect(reviewDialog.getByRole("button", { name: "Approve & Apply" })).toBeDisabled()

    await page.getByLabel("Decision note").fill("Reviewed staged research follow-up for mobile smoke coverage.")
    await expect(reviewDialog.getByRole("button", { name: "Approve & Apply" })).toBeEnabled()
    await expect(reviewDialog.getByRole("button", { name: "Reject Proposal" })).toBeVisible()
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
    await expect(page.getByRole("button", { name: "Lineage" })).toBeVisible()
  })
})
