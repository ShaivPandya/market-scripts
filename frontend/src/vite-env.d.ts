/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AUTH_MODE?: "cloudflare" | "password"
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

