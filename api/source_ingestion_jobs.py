from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ontology.source_ingestion import SourceIngestionService, require_multimodal_ingestion_enabled


class SourceExtractionJobRequest(BaseModel):
    artifact_uid: str = Field(..., min_length=1)
    extractor_ids: list[str] | None = None
    actor: dict[str, Any] | None = None


def run_source_extraction_job(req: SourceExtractionJobRequest) -> dict[str, Any]:
    require_multimodal_ingestion_enabled()
    return SourceIngestionService().run_extractions(
        req.artifact_uid,
        extractor_ids=req.extractor_ids,
        actor=req.actor,
    )
