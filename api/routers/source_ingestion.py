from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.exceptions import NotFoundError, ValidationError
from api.request_limits import read_upload_file_bytes
from api.routers.auth import require_actor
from api.serializers import serialize_response
from ontology.policy import Actor
from ontology.source_ingestion import (
    MAX_SOURCE_UPLOAD_SIZE_BYTES,
    SourceIngestionService,
    UploadInput,
    require_multimodal_ingestion_enabled,
)

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]


class SourceManifestRequest(BaseModel):
    manifest_id: str = Field(..., min_length=1)
    name: str | None = None
    source_kind: str = "document"
    allowed_mime_types: list[str] = Field(default_factory=list)
    dataset: str | None = None
    sensitivity: str = "private"
    extractor_ids: list[str] = Field(default_factory=list)
    materialization_policy: str = "manual_review"
    retention_class: str = "user_state"
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionRequest(BaseModel):
    artifact_uid: str = Field(..., min_length=1)
    extractor_ids: list[str] | None = None


@router.post("/source-ingestion/manifests")
def create_source_manifest(req: SourceManifestRequest, actor: ActorDep):
    require_multimodal_ingestion_enabled()
    return serialize_response(
        SourceIngestionService().create_manifest(req.model_dump(exclude_none=True), actor=_actor_dict(actor))
    )


@router.post("/source-ingestion/uploads")
async def upload_source_artifact(
    actor: ActorDep,
    manifest_id: str = Form(...),  # noqa: B008
    file: UploadFile = File(...),  # noqa: B008
    title: str | None = Form(None),  # noqa: B008
    ticker: str | None = Form(None),  # noqa: B008
    dataset: str | None = Form(None),  # noqa: B008
    record_kind: str | None = Form(None),  # noqa: B008
    record_key: str | None = Form(None),  # noqa: B008
    run_extractors: bool = Form(False),  # noqa: B008
):
    require_multimodal_ingestion_enabled()
    content = await read_upload_file_bytes(file, limit_bytes=MAX_SOURCE_UPLOAD_SIZE_BYTES, limit_label="30 MiB")
    result = SourceIngestionService().upload_artifact(
        UploadInput(
            manifest_id=manifest_id,
            filename=file.filename or "",
            content_type=file.content_type or "",
            content=content,
            title=title,
            ticker=ticker,
            dataset=dataset,
            record_kind=record_kind,
            record_key=record_key,
            run_extractors=run_extractors,
            actor=_actor_dict(actor),
        )
    )
    return serialize_response(result)


@router.get("/source-ingestion/artifacts")
def list_source_artifacts(
    artifact_type: str = "all",
    manifest_id: str | None = None,
    ticker: str | None = None,
    limit: int = 25,
):
    require_multimodal_ingestion_enabled()
    if artifact_type not in {"all", "document", "media"}:
        raise ValidationError("artifact_type must be all, document, or media.")
    normalized_artifact_type = cast(Literal["all", "document", "media"], artifact_type)
    return serialize_response(
        SourceIngestionService().list_artifacts(
            artifact_type=normalized_artifact_type, manifest_id=manifest_id, ticker=ticker, limit=limit
        )
    )


@router.get("/source-ingestion/artifacts/{artifact_uid:path}")
def get_source_artifact(artifact_uid: str):
    require_multimodal_ingestion_enabled()
    return serialize_response(SourceIngestionService().get_artifact_detail(artifact_uid))


@router.post("/source-ingestion/extractions")
def enqueue_source_extraction(req: ExtractionRequest, actor: ActorDep):
    require_multimodal_ingestion_enabled()
    payload = {**req.model_dump(exclude_none=True), "actor": _actor_dict(actor)}
    row, _disposition = enqueue_registered_job("source_extraction", payload, cache_key=None, reuse_completed=False)
    return enqueue_response(row, "/api/source-ingestion/extractions/{job_id}")


@router.get("/source-ingestion/extractions/{run_id}")
def get_source_extraction(run_id: str):
    require_multimodal_ingestion_enabled()
    try:
        return poll_registered_job(run_id)
    except KeyError:
        raise NotFoundError("source extraction job", run_id) from None


@router.get("/source-ingestion/observations")
def summarize_source_observations(artifact_uid: str | None = None, limit: int = 20):
    require_multimodal_ingestion_enabled()
    return serialize_response(SourceIngestionService().summarize_observations(artifact_uid=artifact_uid, limit=limit))


def _actor_dict(actor: Actor) -> dict[str, Any]:
    return {
        "actor_type": actor.actor_type,
        "actor_id": actor.actor_id,
        "roles": list(actor.roles),
        "source": actor.source,
        "parent_actor_id": actor.parent_actor_id,
    }
