from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api.state_storage import exists_text, read_text, write_text
from paths import PROJECT_ROOT

MANAGEMENT_QUALITY_DIR = PROJECT_ROOT / "investment_management_quality"
MANAGEMENT_QUALITY_GCS_PREFIX = "live/management_quality"


@dataclass(frozen=True)
class ManagementQualityContentSave:
    output: dict
    source_path: str
    index_content: str


def management_quality_path(ticker: str) -> Path:
    return MANAGEMENT_QUALITY_DIR / f"{ticker}.md"


def management_quality_gcs_key(ticker: str) -> str:
    return f"{MANAGEMENT_QUALITY_GCS_PREFIX}/{ticker}.md"


def management_quality_exists(ticker: str) -> bool:
    return exists_text(management_quality_path(ticker), management_quality_gcs_key(ticker))


def read_management_quality(ticker: str) -> str:
    return read_text(management_quality_path(ticker), management_quality_gcs_key(ticker), encoding="utf-8")


def write_management_quality(ticker: str, content: str) -> str:
    return write_text(
        management_quality_path(ticker),
        management_quality_gcs_key(ticker),
        content,
        encoding="utf-8",
        content_type="text/markdown; charset=utf-8",
    )


def save_management_quality_content(
    ticker: str,
    content: str,
    *,
    preserve_exact_content: bool = False,
) -> ManagementQualityContentSave:
    response_content = content if preserve_exact_content else content.strip()
    write_content = content if preserve_exact_content else f"{response_content}\n"
    source_path = write_management_quality(ticker, write_content)
    return ManagementQualityContentSave(
        output={"status": "ok", "ticker": ticker, "content": response_content},
        source_path=source_path,
        index_content=response_content,
    )
