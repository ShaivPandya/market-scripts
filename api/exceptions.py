"""
Custom exception hierarchy for the Market Analysis API.

These exceptions are caught by global handlers in api/main.py and converted
to structured JSON error responses.  Internal details are logged server-side
but never leaked to the client.
"""


class AppError(Exception):
    """Base application error — all custom exceptions inherit from this."""

    def __init__(self, message: str, *, status_code: int = 500, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail


class DataFetchError(AppError):
    """An external dependency (yfinance, FRED, SEC, LLM APIs, etc.) failed."""

    def __init__(self, source: str, detail: str = ""):
        # Avoid returning HTTP 502 from the app itself. When the API is behind
        # Cloudflare, origin 502 responses are presented as gateway failures,
        # which hides the real dependency error from the UI.
        super().__init__(f"Data fetch failed: {source}", status_code=424, detail=detail)
        self.source = source


class SnapshotUnavailableError(AppError):
    """A production endpoint needs a precomputed snapshot that is not available."""

    def __init__(self, snapshot_key: str, detail: str = ""):
        super().__init__(f"Snapshot unavailable: {snapshot_key}", status_code=503, detail=detail)
        self.snapshot_key = snapshot_key


class ConfigurationError(AppError):
    """A required environment variable or configuration value is missing."""

    def __init__(self, key: str, detail: str = ""):
        super().__init__(f"Required configuration missing: {key}", status_code=503, detail=detail)
        self.key = key


class AsyncJobDispatchError(AppError):
    """The API could not enqueue a background job for execution."""

    def __init__(self, detail: str = ""):
        super().__init__("Async job dispatch failed", status_code=503, detail=detail)


class AnalysisError(AppError):
    """An analysis computation failed (e.g. optimiser infeasible)."""

    def __init__(self, message: str = "Analysis computation failed", detail: str = ""):
        super().__init__(message, status_code=500, detail=detail)


class NotFoundError(AppError):
    """A requested resource does not exist."""

    def __init__(self, resource: str, identifier: str = "", detail: str = ""):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        super().__init__(message, status_code=404, detail=detail)
        self.resource = resource
        self.identifier = identifier


class ValidationError(AppError):
    """Request data failed validation."""

    def __init__(self, message: str = "Validation failed", detail: str = ""):
        super().__init__(message, status_code=422, detail=detail)


class ConflictError(AppError):
    """A request conflicts with the current durable resource state."""

    def __init__(self, message: str = "Conflict", detail: str = ""):
        super().__init__(message, status_code=409, detail=detail)
