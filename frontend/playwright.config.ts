import { defineConfig, devices } from "@playwright/test"

const ci = Boolean(process.env.CI)

export default defineConfig({
  testDir: "./tests/smoke",
  fullyParallel: true,
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  retries: ci ? 2 : 0,
  reporter: ci
    ? [
        ["list"],
        ["html", { open: "never" }],
      ]
    : "list",
  outputDir: "test-results",
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1",
    url: "http://127.0.0.1:5173/login",
    reuseExistingServer: !ci,
    timeout: 120_000,
  },
})
