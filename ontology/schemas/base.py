from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class OntologySchemaBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1


def clean_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("value must not be blank")
    return text


def clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_lower_text(value: object) -> str:
    return clean_text(value).lower()


def expected_risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
