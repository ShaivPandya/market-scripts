type Env = {
  API_ORIGIN?: string
  API_PROXY_SECRET?: string
}

export async function onRequest(context: { request: Request; env: Env }) {
  const { request, env } = context

  const apiOrigin = (env.API_ORIGIN ?? "").trim()
  if (!apiOrigin) {
    return new Response("Missing API_ORIGIN", { status: 500 })
  }

  const proxySecret = (env.API_PROXY_SECRET ?? "").trim()
  if (!proxySecret) {
    return new Response("Missing API_PROXY_SECRET", { status: 500 })
  }

  const incomingUrl = new URL(request.url)
  const targetUrl = new URL(apiOrigin)
  targetUrl.pathname = incomingUrl.pathname
  targetUrl.search = incomingUrl.search

  const headers = new Headers(request.headers)
  headers.set("X-Api-Proxy-Secret", proxySecret)

  // Let fetch calculate these for the forwarded request.
  headers.delete("host")
  headers.delete("content-length")

  const method = request.method.toUpperCase()
  const body = method === "GET" || method === "HEAD" ? undefined : request.body

  let upstream: Response
  try {
    upstream = await fetch(targetUrl.toString(), {
      method,
      headers,
      body,
      redirect: "manual",
    })
  } catch {
    return new Response("Upstream API request failed", { status: 502 })
  }

  const outHeaders = new Headers(upstream.headers)

  // Preserve Set-Cookie headers (multiple cookies need special handling in Workers).
  const setCookies = (
    upstream.headers as unknown as {
      getSetCookie?: () => string[]
    }
  ).getSetCookie?.()
  if (setCookies?.length) {
    outHeaders.delete("set-cookie")
    for (const cookie of setCookies) {
      outHeaders.append("set-cookie", cookie)
    }
  }

  outHeaders.set("Cache-Control", "no-store")

  return new Response(upstream.body, {
    status: upstream.status,
    headers: outHeaders,
  })
}
