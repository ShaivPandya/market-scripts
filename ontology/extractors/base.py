from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ExtractionStatus = Literal["succeeded", "partial", "failed", "disabled"]


@dataclass(frozen=True, slots=True)
class ArtifactContext:
    artifact_uid: str
    artifact_type: str
    properties: dict[str, Any]
    content: bytes
    mime_type: str
    artifact_uri: str
    storage_key: str | None
    source_record_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class ExtractedObject:
    object_type: Literal["Observation", "Classification", "PatternDetection", "Evidence", "Citation"]
    business_key: str
    properties: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractorResult:
    status: ExtractionStatus
    objects: list[ExtractedObject] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ArtifactExtractor(Protocol):
    extractor_id: str
    version: str
    supported_mime_types: frozenset[str]
    enabled: bool

    def extract(self, context: ArtifactContext) -> ExtractorResult: ...
