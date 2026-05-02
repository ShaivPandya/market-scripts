from __future__ import annotations

import os

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
MULTIPART_FORM_DATA_OVERHEAD_BYTES = 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class _BodyLimitExceeded(Exception):
    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        super().__init__(_limit_detail(limit_bytes))


def format_bytes(limit_bytes: int) -> str:
    mib = limit_bytes / (1024 * 1024)
    if mib.is_integer():
        return f"{int(mib)} MiB"
    return f"{mib:.1f} MiB"


def max_request_body_bytes() -> int:
    raw = (os.environ.get("MAX_REQUEST_BODY_BYTES") or "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_REQUEST_BODY_BYTES


def _limit_detail(limit_bytes: int) -> str:
    return f"Request body exceeds the {format_bytes(limit_bytes)} limit."


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before routing or endpoint parsing."""

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int | None = None,
        path_limits: dict[str, int] | None = None,
    ):
        self.app = app
        self.max_body_size = max_body_size
        self.path_limits = path_limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        global_limit = self.max_body_size if self.max_body_size is not None else max_request_body_bytes()
        path_limit = self.path_limits.get(str(scope.get("path") or ""))
        if path_limit is None:
            limit_bytes = global_limit
        elif global_limit > 0:
            limit_bytes = min(global_limit, path_limit)
        else:
            limit_bytes = path_limit
        if limit_bytes <= 0:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit_bytes:
                    response = JSONResponse({"detail": _limit_detail(limit_bytes)}, status_code=413)
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > limit_bytes:
                    raise _BodyLimitExceeded(limit_bytes)
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _BodyLimitExceeded as exc:
            if response_started:
                raise
            response = JSONResponse({"detail": _limit_detail(exc.limit_bytes)}, status_code=413)
            await response(scope, receive, send)


async def read_upload_file_bytes(
    file: UploadFile,
    *,
    limit_bytes: int,
    limit_label: str | None = None,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            label = limit_label or format_bytes(limit_bytes)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {label} limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)
