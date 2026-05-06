from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api.state_storage import exists_text, read_text, write_text
from paths import PROJECT_ROOT

OVERVIEWS_DIR = PROJECT_ROOT / "investment_overviews"
OVERVIEWS_GCS_PREFIX = "live/overviews"


@dataclass(frozen=True)
class OverviewContentSave:
    output: dict
    source_path: str
    index_content: str


def overview_path(ticker: str) -> Path:
    return OVERVIEWS_DIR / f"{ticker}.md"


def overview_gcs_key(ticker: str) -> str:
    return f"{OVERVIEWS_GCS_PREFIX}/{ticker}.md"


def overview_exists(ticker: str) -> bool:
    return exists_text(overview_path(ticker), overview_gcs_key(ticker))


def read_overview(ticker: str) -> str:
    return read_text(overview_path(ticker), overview_gcs_key(ticker), encoding="utf-8")


def write_overview(ticker: str, content: str) -> str:
    return write_text(
        overview_path(ticker),
        overview_gcs_key(ticker),
        content,
        encoding="utf-8",
        content_type="text/markdown; charset=utf-8",
    )


def save_overview_content(
    ticker: str,
    content: str,
    *,
    preserve_exact_content: bool = False,
) -> OverviewContentSave:
    response_content = content if preserve_exact_content else content.strip()
    write_content = content if preserve_exact_content else f"{response_content}\n"
    source_path = write_overview(ticker, write_content)
    return OverviewContentSave(
        output={"status": "ok", "ticker": ticker, "content": response_content},
        source_path=source_path,
        index_content=response_content,
    )
