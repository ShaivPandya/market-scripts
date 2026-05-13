from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from api.request_limits import (
    MULTIPART_FORM_DATA_OVERHEAD_BYTES,
    BodySizeLimitMiddleware,
    read_upload_file_bytes,
)


def test_multipart_path_limit_allows_exact_file_limit_and_route_rejects_larger_file():
    file_limit = 64
    app = FastAPI()
    app.add_middleware(
        BodySizeLimitMiddleware,
        path_limits={"/upload": file_limit + MULTIPART_FORM_DATA_OVERHEAD_BYTES},
    )

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):  # noqa: B008 - FastAPI parameter declaration
        payload = await read_upload_file_bytes(file, limit_bytes=file_limit, limit_label="64 bytes")
        return {"size": len(payload)}

    client = TestClient(app)

    exact = client.post("/upload", files={"file": ("exact.md", b"x" * file_limit, "text/markdown")})
    oversized = client.post("/upload", files={"file": ("oversized.md", b"x" * (file_limit + 1), "text/markdown")})

    assert exact.status_code == 200
    assert exact.json() == {"size": file_limit}
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "Uploaded file exceeds the 64 bytes limit."


def test_main_multipart_endpoint_body_limits_include_form_overhead():
    import api.main as main
    import api.routers.management_quality as management_quality_router
    import api.routers.overview as overview_router
    import api.routers.thesis as thesis_router
    from api.routers import economic_growth as economic_growth_router
    from api.routers import portfolio_news as portfolio_news_router

    assert main._ENDPOINT_BODY_LIMITS["/api/thesis/generate"] == (
        thesis_router.MAX_UPLOAD_SIZE_BYTES + MULTIPART_FORM_DATA_OVERHEAD_BYTES
    )
    assert main._ENDPOINT_BODY_LIMITS["/api/overview/generate"] == (
        overview_router.MAX_UPLOAD_SIZE_BYTES + MULTIPART_FORM_DATA_OVERHEAD_BYTES
    )
    assert main._ENDPOINT_BODY_LIMITS["/api/management-quality/generate"] == (
        management_quality_router.MAX_UPLOAD_SIZE_BYTES + MULTIPART_FORM_DATA_OVERHEAD_BYTES
    )
    assert main._ENDPOINT_BODY_LIMITS["/api/economic-growth/crb-file"] == (
        economic_growth_router.MAX_CRB_UPLOAD_SIZE_BYTES + MULTIPART_FORM_DATA_OVERHEAD_BYTES
    )
    assert main._ENDPOINT_BODY_LIMITS["/api/portfolio-news"] == (
        portfolio_news_router.MAX_UPLOAD_SIZE_BYTES + MULTIPART_FORM_DATA_OVERHEAD_BYTES
    )
