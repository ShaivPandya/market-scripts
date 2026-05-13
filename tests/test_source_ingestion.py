from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

import ontology.source_ingestion as source_ingestion
from api.exceptions import ValidationError
from ontology.action_registry import get_action, get_tool_exposure
from ontology.object_service import normalize_object_payload
from ontology.schemas.objects import AnalystFeedbackV1, MediaArtifactV1, SourceManifestV1
from ontology.schemas.registry import NODE_SCHEMAS
from ontology.schemas.relations import RELATION_REGISTRY
from ontology.source_ingestion import SourceIngestionService, UploadInput, sniff_mime_type


class FakeObjects:
    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []

    def get_object(self, object_uid: str, **_kwargs):
        return self.objects.get(object_uid)

    def query_objects(self, object_type: str | None = None, filters: dict[str, Any] | None = None, **_kwargs):
        rows = [row for row in self.objects.values() if object_type is None or row["object_type"] == object_type]
        if filters:
            rows = [
                row
                for row in rows
                if all((row.get("properties") or {}).get(key) == value for key, value in filters.items())
            ]
        return rows

    def write_object(self, object_type, business_key, properties, valid_from, **_kwargs):
        del valid_from
        object_uid = source_ingestion.object_uid_for(object_type, business_key, properties)
        normalized = normalize_object_payload(object_uid, object_type, business_key, properties)
        row = {
            "object_uid": normalized["object_uid"],
            "object_type": normalized["object_type"],
            "business_key": business_key,
            "properties": normalized["properties"],
            "_meta": {"temporal": {"version_id": f"version:{normalized['object_uid']}"}},
        }
        self.objects[row["object_uid"]] = row
        return row

    def write_relation(self, relation_type, source_uid, target_uid, properties, valid_from, **_kwargs):
        del valid_from
        row = {
            "relation_type": relation_type,
            "source_object_uid": source_uid,
            "target_object_uid": target_uid,
            "properties": dict(properties or {}),
        }
        self.relations.append(row)
        return row

    def query_relations(
        self,
        relation_type: str | None = None,
        *,
        source_object_uid: str | None = None,
        target_object_uid: str | None = None,
        **_kwargs,
    ):
        rows = self.relations
        if relation_type:
            rows = [row for row in rows if row["relation_type"] == relation_type]
        if source_object_uid:
            rows = [row for row in rows if row["source_object_uid"] == source_object_uid]
        if target_object_uid:
            rows = [row for row in rows if row["target_object_uid"] == target_object_uid]
        return rows


class FakeTemporalRepo:
    def __init__(self):
        self.records: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def write_source_record_version(self, write):
        key = (write.vendor, write.source_name, write.dataset, write.record_kind, write.record_key_hash)
        if key not in self.records:
            record = asdict(write)
            record["source_record_id"] = f"00000000-0000-0000-0000-{len(self.records) + 1:012d}"
            record["load_time"] = write.load_time
            self.records[key] = record
        return self.records[key]


@pytest.fixture()
def fake_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SourceIngestionService:
    monkeypatch.setattr(source_ingestion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        source_ingestion.provenance, "deterministic_id", lambda *parts: "pv:" + ":".join(map(str, parts))
    )
    monkeypatch.setattr(source_ingestion.provenance, "start_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(source_ingestion.provenance, "finish_event", lambda *args, **kwargs: None)
    return SourceIngestionService(objects=FakeObjects(), temporal_repo=FakeTemporalRepo())


def test_multimodal_schema_registration():
    for object_type in [
        "SourceManifest",
        "MediaArtifact",
        "ExtractionRun",
        "Observation",
        "Classification",
        "PatternDetection",
        "AnalystFeedback",
    ]:
        assert object_type in NODE_SCHEMAS
    for relation_type in [
        "source_manifest_governs_source_record",
        "source_record_produces_document_artifact",
        "source_record_produces_media_artifact",
        "artifact_has_extraction_run",
        "extraction_run_produces_observation",
        "extraction_run_produces_classification",
        "extraction_run_produces_pattern_detection",
        "analyst_feedback_targets_object",
    ]:
        assert relation_type in RELATION_REGISTRY

    manifest = SourceManifestV1(manifest_id="my upload", name="My Upload", dataset="research")
    media = MediaArtifactV1(
        media_id="abc",
        mime_type="image/png",
        content_hash="abc",
        artifact_uri="/tmp/abc.png",
    )
    assert manifest.manifest_id == "my_upload"
    assert media.media_id == "abc"
    with pytest.raises(PydanticValidationError):
        AnalystFeedbackV1(
            feedback_id="bad",
            target_object_uid="observation:x",
            target_object_type="Observation",
            decision="correct",
        )


def test_sniff_mime_type_validates_signatures():
    assert sniff_mime_type(b"%PDF-1.7\n", "application/pdf", "a.pdf") == "application/pdf"
    assert sniff_mime_type(b"# hello", "text/markdown", "a.md") == "text/markdown"
    assert sniff_mime_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, "image/png", "a.png") == "image/png"
    with pytest.raises(ValidationError):
        sniff_mime_type(b"not a pdf", "application/pdf", "a.pdf")


def test_upload_markdown_writes_redacted_source_and_dedupes_bytes(fake_service: SourceIngestionService):
    manifest = fake_service.create_manifest(
        {
            "manifest_id": "research_uploads",
            "name": "Research Uploads",
            "dataset": "research_uploads",
            "allowed_mime_types": ["text/markdown"],
            "extractor_ids": ["deterministic.artifact_metadata", "deterministic.document_text"],
        }
    )
    assert manifest["object_uid"] == "source_manifest:research_uploads"

    upload = UploadInput(
        manifest_id="research_uploads",
        filename="note.md",
        content_type="text/markdown",
        content=b"# Thesis\n\nThis is source text.",
        title="Note",
    )
    first = fake_service.upload_artifact(upload)
    second = fake_service.upload_artifact(upload)

    assert first["artifact"]["object_type"] == "DocumentArtifact"
    assert first["duplicate_artifact_bytes"] is False
    assert second["duplicate_artifact_bytes"] is True
    source_records = fake_service.temporal_repo.records.values()  # type: ignore[attr-defined]
    assert len(list(source_records)) == 1
    source_record = next(iter(fake_service.temporal_repo.records.values()))  # type: ignore[attr-defined]
    assert "This is source text" not in str(source_record["payload"])
    assert {"source_manifest_governs_source_record", "source_record_produces_document_artifact"} <= {
        relation["relation_type"]
        for relation in fake_service.objects.relations  # type: ignore[attr-defined]
    }


def test_image_upload_with_extractors_writes_media_observation_and_classification(fake_service: SourceIngestionService):
    fake_service.create_manifest(
        {
            "manifest_id": "image_uploads",
            "name": "Images",
            "dataset": "image_uploads",
            "allowed_mime_types": ["image/png"],
            "extractor_ids": ["deterministic.artifact_metadata", "deterministic.image_metadata"],
        }
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (2).to_bytes(4, "big") + (3).to_bytes(4, "big") + b"\x00" * 8
    result = fake_service.upload_artifact(
        UploadInput(
            manifest_id="image_uploads",
            filename="image.png",
            content_type="image/png",
            content=png,
            run_extractors=True,
        )
    )
    assert result["artifact"]["object_type"] == "MediaArtifact"
    assert result["artifact"]["width"] == 2
    assert result["artifact"]["height"] == 3
    object_types = {row["object_type"] for row in fake_service.objects.objects.values()}  # type: ignore[attr-defined]
    assert {"ExtractionRun", "Observation", "Classification"} <= object_types
    relation_types = {relation["relation_type"] for relation in fake_service.objects.relations}  # type: ignore[attr-defined]
    assert "artifact_has_extraction_run" in relation_types
    assert "extraction_run_produces_observation" in relation_types


def test_agent_source_tools_are_read_or_proposal_only():
    assert get_tool_exposure("list_source_artifacts").access_mode == "read"
    assert get_tool_exposure("get_source_artifact").access_mode == "read"
    assert get_tool_exposure("summarize_extracted_observations").access_mode == "read"
    proposal = get_tool_exposure("propose_analyst_feedback")
    assert proposal.access_mode == "proposal"
    assert proposal.action_id == "create_analyst_feedback"
    typed = proposal.input_model.model_validate(
        {
            "target_object_uid": "observation:abc",
            "target_object_type": "Observation",
            "decision": "reject",
            "reason": "The extraction is not supported by the source.",
        }
    )
    action_payload = proposal.to_action_input(typed)  # type: ignore[union-attr]
    assert action_payload["note"] == "The extraction is not supported by the source."
    get_action("create_analyst_feedback").input_model.model_validate(action_payload)
