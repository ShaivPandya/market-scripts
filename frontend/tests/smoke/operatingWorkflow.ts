import { expect, type Locator, type Page } from "@playwright/test"

interface Box {
  x: number
  y: number
  width: number
  height: number
}

function intersectionArea(a: Box, b: Box): number {
  const overlapWidth = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
  const overlapHeight = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y))
  return overlapWidth * overlapHeight
}

/** Fails when two visible controls overlap by more than half of the smaller control's area. */
export async function expectPrimaryControlsSeparated(...locators: Locator[]) {
  const boxes: Box[] = []
  for (const locator of locators) {
    await expect(locator).toBeVisible()
    const box = await locator.boundingBox()
    expect(box).not.toBeNull()
    boxes.push(box!)
  }

  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i]
      const b = boxes[j]
      const overlap = intersectionArea(a, b)
      const smallerArea = Math.min(a.width * a.height, b.width * b.height)
      expect(overlap).toBeLessThan(smallerArea * 0.5)
    }
  }
}

export async function expectDecisionTraceDrawer(drawer: Locator) {
  await expect(drawer).toBeVisible()
  await expect(drawer.getByRole("heading", { name: "Blockers" })).toBeVisible()
  await expect(drawer.getByRole("heading", { name: "Gates" })).toBeVisible()
  await expect(drawer.getByRole("heading", { name: "Provenance" })).toBeVisible()
}

export async function dismissFloatingAlerts(page: Page) {
  const dismiss = page.getByRole("button", { name: "Dismiss action item alert" })
  if (await dismiss.isVisible().catch(() => false)) {
    await dismiss.click()
    await expect(page.getByRole("status").filter({ hasText: "Action item staged" })).toBeHidden()
  }
}
