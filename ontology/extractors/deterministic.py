from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ontology.extractors.base import ArtifactContext, ExtractedObject, ExtractorResult

TEXT_MIME_TYPES = frozenset({"text/plain", "text/markdown", "text/x-markdown"})
DOCUMENT_MIME_TYPES = TEXT_MIME_TYPES | frozenset({"application/pdf"})
IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


@dataclass(frozen=True, slots=True)
class ArtifactMetadataExtractor:
    extractor_id: str = "deterministic.artifact_metadata"
    version: str = "1"
    supported_mime_types: frozenset[str] = DOCUMENT_MIME_TYPES | IMAGE_MIME_TYPES
    enabled: bool = True

    def extract(self, context: ArtifactContext) -> ExtractorResult:
        observation_id = f"{self.extractor_id}:{context.content_hash}"
        value: dict[str, Any] = {
            "artifact_uid": context.artifact_uid,
            "artifact_type": context.artifact_type,
            "mime_type": context.mime_type,
            "byte_size": len(context.content),
            "content_hash": context.content_hash,
            "artifact_uri": context.artifact_uri,
        }
        width = context.properties.get("width")
        height = context.properties.get("height")
        if width is not None and height is not None:
            value["width"] = width
            value["height"] = height
        return ExtractorResult(
            status="succeeded",
            output=value,
            objects=[
                ExtractedObject(
                    object_type="Observation",
                    business_key=observation_id,
                    properties={
                        "observation_id": observation_id,
                        "observation_type": "artifact_metadata",
                        "value": value,
                        "confidence": 1.0,
                        "source_record_id": context.source_record_id,
                        "artifact_uid": context.artifact_uid,
                        "status": "active",
                        "ontology_run_id": "operational",
                    },
                )
            ],
        )


@dataclass(frozen=True, slots=True)
class DocumentTextExtractor:
    extractor_id: str = "deterministic.document_text"
    version: str = "1"
    supported_mime_types: frozenset[str] = DOCUMENT_MIME_TYPES
    enabled: bool = True

    def extract(self, context: ArtifactContext) -> ExtractorResult:
        text, error = _extract_text(context.content, context.mime_type)
        if not text:
            return ExtractorResult(status="partial", output={"char_count": 0}, error=error or "No text extracted.")

        preview = text[:2000]
        observation_id = f"{self.extractor_id}:{context.content_hash}:text"
        evidence_id = f"{self.extractor_id}:{context.content_hash}:evidence"
        citation_id = f"{self.extractor_id}:{context.content_hash}:citation"
        return ExtractorResult(
            status="succeeded" if not error else "partial",
            output={"char_count": len(text), "text_preview": preview},
            error=error,
            objects=[
                ExtractedObject(
                    object_type="Observation",
                    business_key=observation_id,
                    properties={
                        "observation_id": observation_id,
                        "observation_type": "extracted_text",
                        "value": {"char_count": len(text), "text_preview": preview},
                        "confidence": 1.0 if not error else 0.6,
                        "source_record_id": context.source_record_id,
                        "artifact_uid": context.artifact_uid,
                        "span": {"start": 0, "end": min(len(text), len(preview))},
                        "status": "active",
                        "ontology_run_id": "operational",
                    },
                ),
                ExtractedObject(
                    object_type="Evidence",
                    business_key=evidence_id,
                    properties={
                        "evidence_id": evidence_id,
                        "evidence_type": "source_excerpt",
                        "title": context.properties.get("title") or context.content_hash,
                        "summary": preview[:500],
                        "source_record_id": context.source_record_id,
                        "document_artifact_id": context.properties.get("document_id"),
                        "confidence": 1.0 if not error else 0.6,
                        "ontology_run_id": "operational",
                    },
                ),
                ExtractedObject(
                    object_type="Citation",
                    business_key=citation_id,
                    properties={
                        "citation_id": citation_id,
                        "source_record_id": context.source_record_id,
                        "document_artifact_id": context.properties.get("document_id"),
                        "title": context.properties.get("title"),
                        "source_path": context.artifact_uri,
                        "span_start": 0,
                        "span_end": min(len(text), len(preview)),
                        "quote_hash": context.content_hash,
                        "ontology_run_id": "operational",
                    },
                ),
            ],
        )


@dataclass(frozen=True, slots=True)
class ImageMetadataExtractor:
    extractor_id: str = "deterministic.image_metadata"
    version: str = "1"
    supported_mime_types: frozenset[str] = IMAGE_MIME_TYPES
    enabled: bool = True

    def extract(self, context: ArtifactContext) -> ExtractorResult:
        width = context.properties.get("width")
        height = context.properties.get("height")
        if width is None or height is None:
            width, height = image_dimensions(context.content, context.mime_type)
        observation_id = f"{self.extractor_id}:{context.content_hash}:dimensions"
        classification_id = f"{self.extractor_id}:{context.content_hash}:classification"
        value = {
            "media_type": "image",
            "mime_type": context.mime_type,
            "width": width,
            "height": height,
            "byte_size": len(context.content),
        }
        return ExtractorResult(
            status="succeeded" if width is not None and height is not None else "partial",
            output=value,
            error=None if width is not None and height is not None else "Image dimensions unavailable.",
            objects=[
                ExtractedObject(
                    object_type="Observation",
                    business_key=observation_id,
                    properties={
                        "observation_id": observation_id,
                        "observation_type": "image_metadata",
                        "value": value,
                        "confidence": 1.0 if width is not None and height is not None else 0.5,
                        "source_record_id": context.source_record_id,
                        "artifact_uid": context.artifact_uid,
                        "status": "active",
                        "ontology_run_id": "operational",
                    },
                ),
                ExtractedObject(
                    object_type="Classification",
                    business_key=classification_id,
                    properties={
                        "classification_id": classification_id,
                        "label": "image",
                        "classifier_id": self.extractor_id,
                        "taxonomy": "media_type",
                        "confidence": 1.0,
                        "source_record_id": context.source_record_id,
                        "artifact_uid": context.artifact_uid,
                        "status": "active",
                        "ontology_run_id": "operational",
                    },
                ),
            ],
        )


@dataclass(frozen=True, slots=True)
class DisabledModelExtractor:
    extractor_id: str
    version: str = "1"
    supported_mime_types: frozenset[str] = DOCUMENT_MIME_TYPES | IMAGE_MIME_TYPES
    enabled: bool = False

    def extract(self, context: ArtifactContext) -> ExtractorResult:
        del context
        return ExtractorResult(status="disabled", error="Model-backed extractors are disabled.")


def _extract_text(content: bytes, mime_type: str) -> tuple[str, str | None]:
    if mime_type in TEXT_MIME_TYPES:
        try:
            return content.decode("utf-8"), None
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace"), "Text contained invalid UTF-8 bytes."
    if mime_type == "application/pdf":
        try:
            from io import BytesIO

            from pypdf import PdfReader
        except ImportError:
            return "", "pypdf is not installed; PDF text extraction skipped."
        try:
            reader = PdfReader(BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages[:25]]
            text = "\n\n".join(page for page in pages if page.strip())
            return text, None if text else "PDF contained no extractable text."
        except Exception as exc:  # noqa: BLE001 - extraction errors are persisted as degraded runs
            return "", str(exc) or exc.__class__.__name__
    return "", f"Unsupported document MIME type: {mime_type}"


def image_dimensions(content: bytes, mime_type: str) -> tuple[int | None, int | None]:
    if mime_type == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
        width, height = struct.unpack(">II", content[16:24])
        return int(width), int(height)
    if mime_type == "image/jpeg" and content.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(content)
    if mime_type == "image/webp" and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return _webp_dimensions(content)
    return None, None


def _jpeg_dimensions(content: bytes) -> tuple[int | None, int | None]:
    idx = 2
    while idx + 9 < len(content):
        if content[idx] != 0xFF:
            idx += 1
            continue
        marker = content[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > len(content):
            return None, None
        length = int.from_bytes(content[idx : idx + 2], "big")
        if length < 2 or idx + length > len(content):
            return None, None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 <= len(content):
                height = int.from_bytes(content[idx + 3 : idx + 5], "big")
                width = int.from_bytes(content[idx + 5 : idx + 7], "big")
                return width, height
            return None, None
        idx += length
    return None, None


def _webp_dimensions(content: bytes) -> tuple[int | None, int | None]:
    chunk = content[12:16]
    if chunk == b"VP8X" and len(content) >= 30:
        width = 1 + int.from_bytes(content[24:27], "little")
        height = 1 + int.from_bytes(content[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(content) >= 30:
        start = content.find(b"\x9d\x01\x2a", 20)
        if start != -1 and start + 7 <= len(content):
            width = int.from_bytes(content[start + 3 : start + 5], "little") & 0x3FFF
            height = int.from_bytes(content[start + 5 : start + 7], "little") & 0x3FFF
            return width, height
    if chunk == b"VP8L" and len(content) >= 25 and content[20] == 0x2F:
        bits = int.from_bytes(content[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None, None
