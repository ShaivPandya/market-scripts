from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api.state_storage import exists_text, read_text, write_text
from paths import PROJECT_ROOT

THESES_DIR = PROJECT_ROOT / "investment_theses"
THESES_GCS_PREFIX = "live/theses"


@dataclass(frozen=True)
class ThesisContentSave:
    output: dict
    source_path: str
    index_content: str


def thesis_path(ticker: str) -> Path:
    return THESES_DIR / f"{ticker}.md"


def thesis_gcs_key(ticker: str) -> str:
    return f"{THESES_GCS_PREFIX}/{ticker}.md"


def thesis_exists(ticker: str) -> bool:
    return exists_text(thesis_path(ticker), thesis_gcs_key(ticker))


def read_thesis(ticker: str) -> str:
    return read_text(thesis_path(ticker), thesis_gcs_key(ticker), encoding="utf-8")


def write_thesis(ticker: str, content: str) -> str:
    return write_text(
        thesis_path(ticker),
        thesis_gcs_key(ticker),
        content,
        encoding="utf-8",
        content_type="text/markdown; charset=utf-8",
    )


def save_thesis_content(
    ticker: str,
    content: str,
    *,
    preserve_exact_content: bool = False,
) -> ThesisContentSave:
    response_content = content if preserve_exact_content else content.strip()
    write_content = content if preserve_exact_content else f"{response_content}\n"
    source_path = write_thesis(ticker, write_content)

    from portfolio.thesis_db import upsert_thesis_meta

    upsert_thesis_meta(ticker, status="active")
    return ThesisContentSave(
        output={"status": "ok", "ticker": ticker, "content": response_content},
        source_path=source_path,
        index_content=response_content,
    )
