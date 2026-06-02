let currentCsrfToken: string | null = null

export function setCsrfToken(token: string | null): void {
  currentCsrfToken = token
}

export function getCsrfToken(): string | null {
  return currentCsrfToken
}

export function csrfHeaders(): Record<string, string> {
  const token = getCsrfToken()
  return token ? { "X-CSRF-Token": token } : {}
}
