from __future__ import annotations

from ontology.extractors.base import ArtifactExtractor
from ontology.extractors.deterministic import (
    ArtifactMetadataExtractor,
    DisabledModelExtractor,
    DocumentTextExtractor,
    ImageMetadataExtractor,
)

_EXTRACTORS: dict[str, ArtifactExtractor] = {
    "deterministic.artifact_metadata": ArtifactMetadataExtractor(),
    "deterministic.document_text": DocumentTextExtractor(),
    "deterministic.image_metadata": ImageMetadataExtractor(),
    "model.document_extraction": DisabledModelExtractor("model.document_extraction"),
    "model.image_classification": DisabledModelExtractor("model.image_classification"),
}


def available_extractors() -> list[dict[str, object]]:
    return [
        {
            "extractor_id": extractor.extractor_id,
            "version": extractor.version,
            "enabled": extractor.enabled,
            "supported_mime_types": sorted(extractor.supported_mime_types),
        }
        for extractor in _EXTRACTORS.values()
    ]


def get_extractor(extractor_id: str) -> ArtifactExtractor:
    try:
        return _EXTRACTORS[extractor_id]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact extractor: {extractor_id}") from exc


def enabled_extractors_for_mime(mime_type: str, extractor_ids: list[str] | None = None) -> list[ArtifactExtractor]:
    selected = extractor_ids or list(_EXTRACTORS)
    out: list[ArtifactExtractor] = []
    for extractor_id in selected:
        extractor = get_extractor(extractor_id)
        if not extractor.enabled:
            continue
        if mime_type in extractor.supported_mime_types:
            out.append(extractor)
    return out
