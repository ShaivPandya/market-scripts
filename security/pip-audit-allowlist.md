# pip-audit Allowlist

Each ignored vulnerability must have a bounded exception. Remove the ignore when
the dependency graph no longer reports the CVE.

| Vulnerability | Package context | Reason | Owner | Expires |
| --- | --- | --- | --- | --- |
| CVE-2024-23342 | `ecdsa`, transitive through JWT tooling | The API verifies password-session JWTs with HS256 and does not use `ecdsa` for ECDSA private-key signing. Keep ignored while evaluating replacement of `python-jose` with a smaller JWT verifier. | Platform | 2026-08-01 |
| CVE-2026-4539 | `Pygments`, transitive developer/runtime utility dependency | The service has no endpoint that syntax-highlights attacker-supplied code. The lock pins `Pygments==2.20.0`; keep the ignore only while scanner metadata catches up. | Platform | 2026-08-01 |
| CVE-2026-33752 | `curl_cffi`, transitive through `yfinance` | The lock pins `curl_cffi==0.15.0` through `yfinance==1.3.0`; keep the ignore only while scanner metadata catches up. App code must not pass arbitrary user URLs to `yfinance` or `curl_cffi`. | Platform | 2026-08-01 |
