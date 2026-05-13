## 2026-05-13 - [Middleware Secret Verification Timing Attack]
**Vulnerability:** A timing attack was possible in the `_require_proxy_secret` middleware where `API_PROXY_SECRET` was compared using the standard `!=` operator.
**Learning:** Standard string comparison in Python (and many other languages) returns as soon as it finds a mismatching character, leaking information about the prefix of the secret through the response time.
**Prevention:** Always use `hmac.compare_digest` (or equivalent constant-time comparison functions) when validating secrets, API keys, or tokens to prevent timing-based side-channel attacks.
