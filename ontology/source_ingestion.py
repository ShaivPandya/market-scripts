"""Multimodal source ingestion for governed document and image artifacts."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from api import provenance
from api.exceptions import ConfigurationError, NotFoundError, ValidationError
from api.postgres import use_postgres_state
from api.provenance import redacted_summary, stable_hash
from api.state_storage import object_updated, read_bytes, write_bytes
from ontology.extractors import ArtifactContext, enabled_extractors_for_mime, get_extractor
from ontology.extractors.deterministic import IMAGE_MIME_TYPES, image_dimensions
from ontology.object_service import OntologyObjectService, object_uid_for
from ontology.schemas.identity import source_manifest_id
from ontology.sources.base import payload_fingerprint
from ontology.sources.source_registry import source_registry_metadata
from ontology.temporal_repository import SourceRecordWrite, TemporalOntologyRepository
from paths import PROJECT_ROOT

OPERATIONAL_ONTOLOGY_RUN_ID = "operational"
SOURCE_ARTIFACT_PREFIX = "live/source_artifacts"
MAX_SOURCE_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024

SUPPORTED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "text/markdown",
        "text/x-markdown",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/webp",
    }
)

_EXT_BY_MIME = {
    "application/pdf": ".pdf",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def multimodal_ingestion_enabled() -> bool:
    raw = os.getenv("MULTIMODAL_INGESTION_ENABLED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def require_multimodal_ingestion_enabled() -> None:
    if not multimodal_ingestion_enabled():
        raise ConfigurationError("MULTIMODAL_INGESTION_ENABLED")
    if not use_postgres_state():
        raise ConfigurationError("STATE_DB_BACKEND=postgres")


@dataclass(frozen=True, slots=True)
class UploadInput:
    manifest_id: str
    filename: str
    content_type: str
    content: bytes
    title: str | None = None
    ticker: str | None = None
    dataset: str | None = None
    record_kind: str | None = None
    record_key: str | None = None
    run_extractors: bool = False
    actor: dict[str, Any] | None = None


class SourceIngestionService:
    def __init__(
        self,
        *,
        objects: OntologyObjectService | None = None,
        temporal_repo: TemporalOntologyRepository | None = None,
    ):
        self.objects = objects or OntologyObjectService()
        self.temporal_repo = temporal_repo or TemporalOntologyRepository()

    def create_manifest(self, payload: dict[str, Any], *, actor: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest_id = _clean_key(payload.get("manifest_id") or payload.get("id") or payload.get("name"), "manifest_id")
        now = _now()
        event_id = provenance.deterministic_id("pv:source_manifest", manifest_id)
        registry = source_registry_metadata("source_ingestion_document")
        provenance.start_event(
            event_id=event_id,
            event_type="source_manifest_write",
            event_name="source_ingestion.create_manifest",
            summary={"manifest_id": manifest_id, "source_kind": payload.get("source_kind")},
            metadata={"retention_class": payload.get("retention_class") or "user_state", "source_registry": registry},
        )
        manifest_metadata = dict(payload.get("metadata") or {})
        if registry:
            manifest_metadata["source_registry"] = registry
        properties = {
            "manifest_id": manifest_id,
            "name": payload.get("name") or manifest_id,
            "source_kind": payload.get("source_kind") or "document",
            "allowed_mime_types": _list_text(payload.get("allowed_mime_types")) or sorted(SUPPORTED_MIME_TYPES),
            "dataset": payload.get("dataset") or manifest_id,
            "sensitivity": payload.get("sensitivity") or "private",
            "extractor_ids": _list_text(payload.get("extractor_ids")),
            "materialization_policy": payload.get("materialization_policy") or "manual_review",
            "retention_class": payload.get("retention_class") or "user_state",
            "status": payload.get("status") or "active",
            "created_at": payload.get("created_at") or now,
            "updated_at": now,
            "metadata": manifest_metadata,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        row = self.objects.write_object(
            "SourceManifest",
            manifest_id,
            properties,
            now,
            actor=actor or _system_actor(),
            provenance=event_id,
            input_hash=stable_hash(properties),
        )
        provenance.finish_event(event_id, status="succeeded", output_value={"manifest_id": manifest_id})
        return _flatten(row)

    def upload_artifact(self, upload: UploadInput) -> dict[str, Any]:
        if not upload.content:
            raise ValidationError("Uploaded file is empty.")
        if len(upload.content) > MAX_SOURCE_UPLOAD_SIZE_BYTES:
            raise ValidationError("Uploaded file exceeds 30 MiB.")
        filename = _safe_filename(upload.filename)
        mime_type = sniff_mime_type(upload.content, upload.content_type, filename)
        registry = _source_registry_for_mime(mime_type)
        manifest_uid = source_manifest_id(upload.manifest_id)
        manifest = self.objects.get_object(manifest_uid)
        if not manifest:
            raise NotFoundError("source manifest", upload.manifest_id)
        manifest_props = _props(manifest)
        allowed = {str(value).lower() for value in manifest_props.get("allowed_mime_types") or []}
        if allowed and mime_type not in allowed:
            raise ValidationError(f"MIME type {mime_type} is not allowed by manifest {upload.manifest_id}.")

        now = _now()
        content_hash = hashlib.sha256(upload.content).hexdigest()
        extension = _EXT_BY_MIME[mime_type]
        storage_key = f"{SOURCE_ARTIFACT_PREFIX}/{content_hash[:2]}/{content_hash}{extension}"
        local_path = PROJECT_ROOT / storage_key
        artifact_bytes_existed = object_updated(local_path, storage_key) is not None
        if not artifact_bytes_existed:
            write_bytes(
                local_path,
                storage_key,
                upload.content,
                content_type=mime_type,
                metadata={
                    "content_hash": content_hash,
                    "manifest_id": upload.manifest_id,
                    "filename": filename,
                    "mime_type": mime_type,
                },
            )
        artifact_uri = _artifact_uri(storage_key)

        event_id = provenance.deterministic_id("pv:source_upload", upload.manifest_id, content_hash)
        payload = {
            "manifest_id": upload.manifest_id,
            "filename": filename,
            "title": upload.title,
            "ticker": upload.ticker,
            "mime_type": mime_type,
            "content_hash": content_hash,
            "byte_size": len(upload.content),
            "artifact_uri": artifact_uri,
        }
        if registry:
            payload["source_registry"] = registry
        provenance.start_event(
            event_id=event_id,
            event_type="source_upload",
            event_name="source_ingestion.upload",
            summary={key: value for key, value in payload.items() if key != "artifact_uri"},
            metadata={"storage_key": storage_key, "redaction_policy": "hash_only", "source_registry": registry},
        )
        source_record = self.temporal_repo.write_source_record_version(
            SourceRecordWrite(
                vendor=str((registry or {}).get("vendor_name") or "user_upload"),
                source_name=str(manifest_props.get("source_kind") or "multimodal_upload"),
                source_version="1",
                dataset=upload.dataset or str(manifest_props.get("dataset") or upload.manifest_id),
                record_kind=upload.record_kind or _record_kind_for_mime(mime_type),
                record_key=upload.record_key or f"{upload.manifest_id}:{content_hash}",
                record_key_hash=stable_hash(upload.record_key or f"{upload.manifest_id}:{content_hash}", length=32),
                payload_hash=payload_fingerprint(payload),
                payload=redacted_summary(payload),
                artifact_uri=artifact_uri,
                status="ok",
                quality="ok",
                as_of=now,
                load_time=now,
                valid_from=now,
                provenance_event_id=event_id,
            )
        )
        source_record_id = str(source_record["source_record_id"])
        source_uid = self._write_source_record_object(
            source_record_id=source_record_id,
            source_record=source_record,
            payload=payload,
            provenance_event_id=event_id,
            actor=upload.actor,
        )
        artifact = self._write_artifact_object(
            mime_type=mime_type,
            content_hash=content_hash,
            storage_key=storage_key,
            artifact_uri=artifact_uri,
            source_record_id=source_record_id,
            manifest_id=upload.manifest_id,
            title=upload.title or filename,
            ticker=upload.ticker,
            filename=filename,
            now=now,
            actor=upload.actor,
            provenance_event_id=event_id,
            run_extractors=upload.run_extractors,
            content=upload.content,
        )
        self._write_relation(
            "source_manifest_governs_source_record",
            manifest_uid,
            source_uid,
            event_id,
            actor=upload.actor,
        )
        self._write_relation(
            "source_record_produces_document_artifact"
            if artifact["object_type"] == "DocumentArtifact"
            else "source_record_produces_media_artifact",
            source_uid,
            artifact["object_uid"],
            event_id,
            actor=upload.actor,
        )
        extraction_runs: list[dict[str, Any]] = []
        if upload.run_extractors:
            extractor_ids = _list_text(manifest_props.get("extractor_ids"))
            extraction_runs = self.run_extractions(
                artifact["object_uid"],
                extractor_ids=extractor_ids or None,
                actor=upload.actor,
            )["runs"]
        provenance.finish_event(
            event_id,
            status="succeeded",
            output_value={
                "source_record_id": source_record_id,
                "artifact_uid": artifact["object_uid"],
                "content_hash": content_hash,
            },
        )
        return {
            "status": "ok",
            "source_record": {"source_record_id": source_record_id, "object_uid": source_uid},
            "artifact": artifact,
            "storage_key": storage_key,
            "duplicate_artifact_bytes": artifact_bytes_existed,
            "extraction_runs": extraction_runs,
            "_meta": {"source_registry": registry},
        }

    def run_extractions(
        self,
        artifact_uid: str,
        *,
        extractor_ids: list[str] | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact = self.objects.get_object(artifact_uid)
        if not artifact:
            raise NotFoundError("artifact", artifact_uid)
        props = _props(artifact)
        mime_type = _clean_key(props.get("mime_type"), "mime_type").lower()
        storage_key = str(_props(artifact).get("metadata", {}).get("storage_key") or "")
        artifact_uri = str(props.get("artifact_uri") or "")
        content = read_bytes(PROJECT_ROOT / storage_key, storage_key) if storage_key else b""
        context = ArtifactContext(
            artifact_uid=artifact_uid,
            artifact_type=str(artifact["object_type"]),
            properties=props,
            content=content,
            mime_type=mime_type,
            artifact_uri=artifact_uri,
            storage_key=storage_key or None,
            source_record_id=props.get("source_record_id"),
            content_hash=str(props.get("content_hash") or stable_hash(artifact_uid)),
        )
        extractors = enabled_extractors_for_mime(mime_type, extractor_ids)
        runs = [self._run_one_extractor(context, extractor.extractor_id, actor=actor) for extractor in extractors]
        if extractor_ids:
            disabled_or_unsupported = [
                extractor_id for extractor_id in extractor_ids if extractor_id not in {r["extractor_id"] for r in runs}
            ]
            for extractor_id in disabled_or_unsupported:
                get_extractor(extractor_id)
                runs.append(self._write_disabled_extraction_run(context, extractor_id, actor=actor))
        return {
            "status": "ok",
            "artifact_uid": artifact_uid,
            "runs": runs,
            "_meta": {"source_registry": source_registry_metadata("source_extraction")},
        }

    def list_artifacts(
        self,
        *,
        artifact_type: Literal["document", "media", "all"] = "all",
        manifest_id: str | None = None,
        ticker: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        filters = {key: value for key, value in {"manifest_id": manifest_id, "ticker": ticker}.items() if value}
        types = (
            ["DocumentArtifact", "MediaArtifact"]
            if artifact_type == "all"
            else ["DocumentArtifact" if artifact_type == "document" else "MediaArtifact"]
        )
        rows: list[dict[str, Any]] = []
        for object_type in types:
            rows.extend(self.objects.query_objects(object_type, filters=filters, limit=limit))
        artifacts = [_flatten(row) for row in rows[: max(1, min(limit, 100))]]
        return {"artifacts": artifacts, "count": len(artifacts)}

    def get_artifact_detail(self, artifact_uid: str) -> dict[str, Any]:
        artifact = self.objects.get_object(artifact_uid)
        if not artifact:
            raise NotFoundError("artifact", artifact_uid)
        extraction_relations = self.objects.query_relations(
            "artifact_has_extraction_run", source_object_uid=artifact_uid
        )
        runs = [
            _flatten(row)
            for relation in extraction_relations
            if (row := self.objects.get_object(str(relation.get("target_object_uid") or "")))
        ]
        produced: list[dict[str, Any]] = []
        for run in runs:
            run_uid = str(run.get("object_uid") or "")
            for relation_type in (
                "extraction_run_produces_observation",
                "extraction_run_produces_classification",
                "extraction_run_produces_pattern_detection",
            ):
                for relation in self.objects.query_relations(relation_type, source_object_uid=run_uid):
                    obj = self.objects.get_object(str(relation.get("target_object_uid") or ""))
                    if obj:
                        produced.append(_flatten(obj))
        return {"artifact": _flatten(artifact), "extraction_runs": runs, "extracted_objects": produced}

    def summarize_observations(self, *, artifact_uid: str | None = None, limit: int = 20) -> dict[str, Any]:
        filters = {"artifact_uid": artifact_uid} if artifact_uid else None
        observations = [
            _flatten(row) for row in self.objects.query_objects("Observation", filters=filters, limit=limit)
        ]
        classifications = [
            _flatten(row) for row in self.objects.query_objects("Classification", filters=filters, limit=limit)
        ]
        patterns = [
            _flatten(row) for row in self.objects.query_objects("PatternDetection", filters=filters, limit=limit)
        ]
        return {
            "observations": observations,
            "classifications": classifications,
            "pattern_detections": patterns,
            "count": len(observations) + len(classifications) + len(patterns),
        }

    def _run_one_extractor(
        self, context: ArtifactContext, extractor_id: str, *, actor: dict[str, Any] | None
    ) -> dict[str, Any]:
        extractor = get_extractor(extractor_id)
        started = _now()
        event_id = provenance.deterministic_id(
            "pv:artifact_extraction", context.artifact_uid, extractor_id, context.content_hash
        )
        provenance.start_event(
            event_id=event_id,
            event_type="artifact_extraction",
            event_name=extractor_id,
            summary={"artifact_uid": context.artifact_uid, "mime_type": context.mime_type},
            metadata={
                "extractor_version": extractor.version,
                "source_registry": source_registry_metadata("source_extraction"),
            },
        )
        began = time.perf_counter()
        produced_uids: list[str] = []
        status = "failed"
        error: str | None = None
        output: dict[str, Any] = {}
        try:
            result = extractor.extract(context)
            status = result.status
            error = result.error
            output = dict(result.output or {})
            for extracted in result.objects:
                props = dict(extracted.properties)
                if "source_record_id" in props and not props.get("source_record_id"):
                    props["source_record_id"] = context.source_record_id
                if "artifact_uid" in props and not props.get("artifact_uid"):
                    props["artifact_uid"] = context.artifact_uid
                if "extraction_run_id" in props and not props.get("extraction_run_id"):
                    props["extraction_run_id"] = _run_id(context, extractor_id)
                row = self.objects.write_object(
                    extracted.object_type,
                    extracted.business_key,
                    props,
                    started,
                    actor=actor or _system_actor(),
                    provenance=event_id,
                    source_record_id=context.source_record_id,
                    input_hash=stable_hash(props),
                )
                produced_uids.append(str(row.get("object_uid") or ""))
            self._link_evidence_citations(produced_uids, event_id, actor=actor)
        except Exception as exc:  # noqa: BLE001 - failed extraction runs are persisted
            status = "failed"
            error = str(exc) or exc.__class__.__name__
        completed = _now()
        duration_ms = round((time.perf_counter() - began) * 1000.0, 1)
        run_row = self.objects.write_object(
            "ExtractionRun",
            _run_id(context, extractor_id),
            {
                "extraction_run_id": _run_id(context, extractor_id),
                "extractor_id": extractor_id,
                "extractor_version": extractor.version,
                "artifact_uid": context.artifact_uid,
                "artifact_type": context.artifact_type,
                "source_record_id": context.source_record_id,
                "status": status,
                "started_at": started,
                "completed_at": completed,
                "duration_ms": duration_ms,
                "output_hash": stable_hash(output) if output else None,
                "error": error,
                "provenance_event_id": event_id,
                "produced_object_uids": [uid for uid in produced_uids if uid],
                "metadata": {"output": output, "source_registry": source_registry_metadata("source_extraction")},
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            started,
            actor=actor or _system_actor(),
            provenance=event_id,
            source_record_id=context.source_record_id,
            input_hash=stable_hash({"status": status, "output": output, "error": error}),
        )
        run_uid = str(run_row.get("object_uid") or "")
        self._write_relation("artifact_has_extraction_run", context.artifact_uid, run_uid, event_id, actor=actor)
        for uid in produced_uids:
            relation_type = _produced_relation_for_uid(uid)
            if relation_type:
                self._write_relation(relation_type, run_uid, uid, event_id, actor=actor)
        provenance.finish_event(
            event_id,
            status="succeeded" if status in {"succeeded", "partial"} else "failed",
            output_value={"run_uid": run_uid, "status": status, "produced_object_uids": produced_uids},
            error=error,
        )
        return {**_flatten(run_row), "extractor_id": extractor_id}

    def _write_disabled_extraction_run(
        self, context: ArtifactContext, extractor_id: str, *, actor: dict[str, Any] | None
    ) -> dict[str, Any]:
        event_id = provenance.deterministic_id("pv:artifact_extraction_disabled", context.artifact_uid, extractor_id)
        now = _now()
        row = self.objects.write_object(
            "ExtractionRun",
            _run_id(context, extractor_id),
            {
                "extraction_run_id": _run_id(context, extractor_id),
                "extractor_id": extractor_id,
                "extractor_version": "1",
                "artifact_uid": context.artifact_uid,
                "artifact_type": context.artifact_type,
                "source_record_id": context.source_record_id,
                "status": "disabled",
                "started_at": now,
                "completed_at": now,
                "error": "Extractor is disabled or unsupported for this artifact MIME type.",
                "provenance_event_id": event_id,
                "metadata": {"source_registry": source_registry_metadata("source_extraction")},
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor or _system_actor(),
            provenance=event_id,
            source_record_id=context.source_record_id,
        )
        return {**_flatten(row), "extractor_id": extractor_id}

    def _write_source_record_object(
        self,
        *,
        source_record_id: str,
        source_record: dict[str, Any],
        payload: dict[str, Any],
        provenance_event_id: str,
        actor: dict[str, Any] | None,
    ) -> str:
        props = {
            "source_record_id": source_record_id,
            "vendor": source_record["vendor"],
            "source_name": source_record["source_name"],
            "source_version": source_record["source_version"],
            "dataset": source_record["dataset"],
            "record_kind": source_record["record_kind"],
            "record_key_hash": source_record["record_key_hash"],
            "payload_hash": source_record["payload_hash"],
            "status": source_record["status"],
            "quality": source_record["quality"],
            "as_of": _iso(source_record.get("as_of")),
            "load_time": _iso(source_record.get("load_time")),
            "artifact_uri": source_record.get("artifact_uri"),
            "provenance_event_id": provenance_event_id,
            "metadata": payload,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        row = self.objects.write_object(
            "SourceRecord",
            source_record_id,
            props,
            _now(),
            actor=actor or _system_actor(),
            provenance=provenance_event_id,
            source_record_id=source_record_id,
            input_hash=stable_hash(props),
        )
        return str(row.get("object_uid") or "")

    def _write_artifact_object(
        self,
        *,
        mime_type: str,
        content_hash: str,
        storage_key: str,
        artifact_uri: str,
        source_record_id: str,
        manifest_id: str,
        title: str,
        ticker: str | None,
        filename: str,
        now: str,
        actor: dict[str, Any] | None,
        provenance_event_id: str,
        run_extractors: bool,
        content: bytes,
    ) -> dict[str, Any]:
        common = {
            "title": title,
            "ticker": ticker,
            "mime_type": mime_type,
            "byte_size": len(content),
            "content_hash": content_hash,
            "artifact_uri": artifact_uri,
            "source_record_id": source_record_id,
            "manifest_id": manifest_id,
            "extraction_status": "pending" if run_extractors else "not_requested",
            "status": "active",
            "source_type": "source_ingestion",
            "source_id": provenance_event_id,
            "created_at": now,
            "updated_at": now,
            "metadata": {
                "filename": filename,
                "storage_key": storage_key,
                "source_registry": _source_registry_for_mime(mime_type),
            },
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        if mime_type in IMAGE_MIME_TYPES:
            width, height = image_dimensions(content, mime_type)
            media_id = content_hash
            row = self.objects.write_object(
                "MediaArtifact",
                media_id,
                {
                    **common,
                    "media_id": media_id,
                    "media_type": "image",
                    "width": width,
                    "height": height,
                },
                now,
                actor=actor or _system_actor(),
                provenance=provenance_event_id,
                source_record_id=source_record_id,
                input_hash=stable_hash(common),
            )
        else:
            document_type = _document_type_for_mime(mime_type)
            document_id = content_hash
            row = self.objects.write_object(
                "DocumentArtifact",
                document_id,
                {
                    **common,
                    "document_type": document_type,
                    "document_id": document_id,
                },
                now,
                actor=actor or _system_actor(),
                provenance=provenance_event_id,
                source_record_id=source_record_id,
                input_hash=stable_hash(common),
            )
        return _flatten(row)

    def _write_relation(
        self,
        relation_type: str,
        source_uid: str,
        target_uid: str,
        provenance_event_id: str,
        *,
        actor: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.objects.write_relation(
            relation_type,
            source_uid,
            target_uid,
            {**(extra or {}), "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_uid": target_uid},
            _now(),
            actor=actor or _system_actor(),
            provenance=provenance_event_id,
        )

    def _link_evidence_citations(
        self, produced_uids: list[str], provenance_event_id: str, *, actor: dict[str, Any] | None
    ) -> None:
        evidence_uids = [uid for uid in produced_uids if uid.startswith("evidence:")]
        citation_uids = [uid for uid in produced_uids if uid.startswith("citation:")]
        for evidence_uid in evidence_uids:
            for citation_uid in citation_uids:
                self._write_relation(
                    "evidence_cites_citation", evidence_uid, citation_uid, provenance_event_id, actor=actor
                )


def sniff_mime_type(content: bytes, content_type: str, filename: str) -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower()
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    if suffix in {".md", ".markdown"} and content_type in {"", "text/plain", "text/markdown", "text/x-markdown"}:
        _decode_text(content)
        return "text/markdown"
    if suffix == ".txt" and content_type in {"", "text/plain"}:
        _decode_text(content)
        return "text/plain"
    if content_type in SUPPORTED_MIME_TYPES and content_type.startswith("text/"):
        _decode_text(content)
        return content_type
    raise ValidationError("File must be PDF, Markdown/text, PNG, JPEG, or WebP.")


def _source_registry_for_mime(mime_type: str) -> dict[str, Any] | None:
    source_id = "source_ingestion_media" if mime_type in IMAGE_MIME_TYPES else "source_ingestion_document"
    return source_registry_metadata(source_id)


def _safe_filename(filename: str) -> str:
    text = str(filename or "").strip()
    if not text:
        raise ValidationError("Uploaded file requires a filename.")
    name = Path(text).name
    if name != text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValidationError("Uploaded filename is invalid.")
    return name


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("Uploaded text must be UTF-8.") from exc


def _artifact_uri(storage_key: str) -> str:
    if os.getenv("STATE_STORAGE_BACKEND", "").strip().lower() == "gcs":
        bucket = os.getenv("GCS_STATE_BUCKET", "").strip()
        return f"gs://{bucket}/{storage_key}" if bucket else storage_key
    return str(PROJECT_ROOT / storage_key)


def _run_id(context: ArtifactContext, extractor_id: str) -> str:
    return f"{extractor_id}:{context.content_hash}"


def _produced_relation_for_uid(uid: str) -> str | None:
    if uid.startswith("observation:"):
        return "extraction_run_produces_observation"
    if uid.startswith("classification:"):
        return "extraction_run_produces_classification"
    if uid.startswith("pattern_detection:"):
        return "extraction_run_produces_pattern_detection"
    return None


def _record_kind_for_mime(mime_type: str) -> str:
    if mime_type in IMAGE_MIME_TYPES:
        return "image_artifact"
    if mime_type == "application/pdf":
        return "pdf_artifact"
    return "text_artifact"


def _document_type_for_mime(mime_type: str) -> str:
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    return "text"


def _props(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("properties") or row.get("properties_json") or {})


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    props = _props(row)
    return {
        **props,
        "object_uid": str(row.get("object_uid") or object_uid_for(str(row.get("object_type") or ""), "", props)),
        "object_type": row.get("object_type"),
        "_meta": row.get("_meta") or {},
    }


def _clean_key(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field} is required.")
    return text


def _list_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _system_actor() -> dict[str, Any]:
    return {"actor_type": "system", "actor_id": "source_ingestion"}
