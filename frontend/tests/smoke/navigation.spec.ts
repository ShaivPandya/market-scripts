import { authenticate, expect, test } from "./fixtures"

function primaryNav(page: import("@playwright/test").Page) {
  return page.getByRole("navigation", { name: "Primary navigation" })
}

test("shows grouped sidebar sections on desktop", async ({ page }) => {
  await authenticate(page)
  await page.goto("/")

  const nav = primaryNav(page)
  await expect(nav.getByText("Core", { exact: true })).toBeVisible()
  await expect(nav.getByText("Labs", { exact: true })).toBeVisible()
  await expect(nav.getByText("Monitors", { exact: true })).toBeVisible()
  await expect(nav.getByText("Macro", { exact: true })).toBeVisible()
  await expect(nav.getByText("Assets", { exact: true })).toBeVisible()
  await expect(nav.getByText("Settings", { exact: true })).toBeVisible()
})

test("navigates via workflow sidebar links on desktop", async ({ page }) => {
  await authenticate(page)
  await page.goto("/")

  const nav = primaryNav(page)

  await nav.getByRole("link", { name: "Workspace" }).click()
  await expect(page).toHaveURL(/\/workspace$/)
  await expect(page.getByRole("heading", { name: "Action queue" })).toBeVisible()

  await nav.getByRole("link", { name: "Portfolio Dashboard" }).click()
  await expect(page).toHaveURL(/\/$/)
  await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()

  await nav.getByRole("link", { name: "Ontology Workbench" }).click()
  await expect(page).toHaveURL(/\/ontology$/)
  await expect(page.getByRole("heading", { name: "Ontology Workbench" })).toBeVisible()

  await nav.getByRole("link", { name: "Liquidity" }).click()
  await expect(page).toHaveURL(/\/liquidity$/)
  await expect(page.getByRole("heading", { name: "Liquidity Dashboard" })).toBeVisible()
})

test("navigates via command palette page search", async ({ page }) => {
  await authenticate(page)
  await page.goto("/")

  await page.getByRole("button", { name: "Search pages" }).click()
  await expect(page.getByPlaceholder("Search pages")).toBeVisible()

  await page.getByPlaceholder("Search pages").fill("Workspace")
  await page.getByRole("option", { name: /Workspace/ }).click()

  await expect(page).toHaveURL(/\/workspace$/)
  await expect(page.getByRole("heading", { name: "Action queue" })).toBeVisible()
})

test("legacy portfolio path redirects to dashboard", async ({ page }) => {
  await authenticate(page)
  await page.goto("/portfolio")

  await expect(page).toHaveURL(/\/$/)
  await expect(page.getByRole("heading", { name: "Portfolio Dashboard" })).toBeVisible()
})

test("legacy optimizer path redirects to analyzer", async ({ page }) => {
  await authenticate(page)
  await page.goto("/optimizer")

  await expect(page).toHaveURL(/\/analyzer$/)
})

test("legacy screener paths redirect to screeners", async ({ page }) => {
  await authenticate(page)
  await page.goto("/quality")

  await expect(page).toHaveURL(/\/screeners$/)
  await expect(page.getByRole("heading", { name: "Screeners" })).toBeVisible()
})
