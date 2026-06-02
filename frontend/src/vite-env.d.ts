/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AUTH_MODE?: "cloudflare" | "password"
  readonly VITE_SENTRY_DSN?: string
  readonly VITE_SENTRY_ENVIRONMENT?: string
  readonly VITE_SENTRY_RELEASE?: string
  readonly VITE_SENTRY_ENABLED?: string
  readonly VITE_SENTRY_FORCE_DEV?: string
  readonly VITE_SENTRY_TRACES_SAMPLE_RATE?: string
  readonly VITE_TALISMAN_RELEASE_GIT_SHA_SHORT?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

