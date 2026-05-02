"""User-curated market/news digest storage and parsing.

Raw markdown uploads are the source of truth. Parsed sections and stories are
derived metadata used by the UI and report prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from api.state_storage import delete_file, exists_text, read_text, write_text
from paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DIGESTS_DIR = PROJECT_ROOT / "data_cache" / "news_digests"
DIGESTS_GCS_PREFIX = "live/news_digests"
MANIFEST_PATH = DIGESTS_DIR / "manifest.json"
MANIFEST_GCS_KEY = f"{DIGESTS_GCS_PREFIX}/manifest.json"
FILES_DIR = DIGESTS_DIR / "files"
FILES_GCS_PREFIX = f"{DIGESTS_GCS_PREFIX}/files"

_DEFAULT_TITLE = "Untitled Digest"
_MAX_TITLE_LEN = 180
_MAX_SLUG_LEN = 90


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _safe_title(value: str | None, fallback: str = _DEFAULT_TITLE) -> str:
    title = re.sub(r"\s+", " ", (value or "").strip())
    if not title:
        title = fallback
    return title[:_MAX_TITLE_LEN]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "digest")[:_MAX_SLUG_LEN].strip("-") or "digest"


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except Exception:
        return None


def _extract_filename_date(filename: str | None) -> date | None:
    name = Path(filename or "").name

    # Common digest export shape: 05012026_digest.md (MMDDYYYY).
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)", name)
    if match:
        month, day, year = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            pass

    # Also accept YYYYMMDD, YYYY-MM-DD, or YYYY_MM_DD.
    match = re.search(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d)", name)
    if match:
        year, month, day = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            pass

    return None


def _extract_generated_date(markdown: str, filename: str | None, uploaded_at: datetime | None) -> date:
    match = re.search(r"\bGenerated:\s*(\d{4}-\d{2}-\d{2})\b", markdown, flags=re.IGNORECASE)
    if match:
        parsed = _parse_date(match.group(1))
        if parsed:
            return parsed

    filename_date = _extract_filename_date(filename)
    if filename_date:
        return filename_date

    return (uploaded_at or _now_utc()).date()


def _extract_title(markdown: str, filename: str | None) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
        if match:
            return _safe_title(match.group(1))

    stem = Path(filename or "").stem.replace("_", " ").replace("-", " ").strip()
    return _safe_title(stem, fallback=_DEFAULT_TITLE)


def _strip_bullet(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(\s*)[-*]\s+(.+?)\s*$", line)
    if not match:
        return None
    indent = len(match.group(1).replace("\t", "    "))
    return indent, match.group(2).strip()


def parse_digest_markdown(
    markdown: str,
    *,
    filename: str | None = None,
    uploaded_at: datetime | None = None,
) -> dict[str, Any]:
    """Parse digest markdown into stable metadata and story sections."""
    title = _extract_title(markdown, filename)
    generated = _extract_generated_date(markdown, filename, uploaded_at)

    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    current_story: dict[str, Any] | None = None
    story_index = 0

    def ensure_section() -> dict[str, Any]:
        nonlocal current_section
        if current_section is None:
            current_section = {"name": "Uncategorized", "stories": []}
            sections.append(current_section)
        return current_section

    for raw_line in markdown.replace("\r\n", "\n").split("\n"):
        h2 = re.match(r"^\s*##\s+(.+?)\s*#*\s*$", raw_line)
        if h2:
            current_section = {"name": _safe_title(h2.group(1), fallback="Untitled Section"), "stories": []}
            sections.append(current_section)
            current_story = None
            continue

        bullet = _strip_bullet(raw_line)
        if bullet is None:
            continue

        indent, text = bullet
        if indent <= 1:
            section = ensure_section()
            story_index += 1
            current_story = {
                "id": f"story-{story_index}",
                "section": section["name"],
                "headline": text,
                "notes": [],
            }
            section["stories"].append(current_story)
        elif current_story is not None:
            current_story["notes"].append(text)

    sections = [section for section in sections if section.get("stories")]
    stories = [story for section in sections for story in section.get("stories", [])]

    return {
        "title": title,
        "slug": slugify(title),
        "generated_date": generated.isoformat(),
        "sections": sections,
        "stories": stories,
        "story_count": len(stories),
        "section_count": len(sections),
    }


def _digest_id(generated_date: str, slug: str) -> str:
    return f"{generated_date}-{slugify(slug)}"


def _digest_local_path(digest_id: str) -> Path:
    return FILES_DIR / f"{digest_id}.md"


def _digest_gcs_key(digest_id: str) -> str:
    return f"{FILES_GCS_PREFIX}/{digest_id}.md"


def _empty_manifest() -> dict[str, Any]:
    return {"version": 1, "digests": []}


def _load_manifest() -> dict[str, Any]:
    if not exists_text(MANIFEST_PATH, MANIFEST_GCS_KEY):
        return _empty_manifest()
    try:
        data = json.loads(read_text(MANIFEST_PATH, MANIFEST_GCS_KEY, encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read news digest manifest; using empty manifest", exc_info=True)
        return _empty_manifest()
    if not isinstance(data, dict) or not isinstance(data.get("digests"), list):
        return _empty_manifest()
    return data


def _write_manifest(manifest: dict[str, Any]) -> None:
    write_text(
        MANIFEST_PATH,
        MANIFEST_GCS_KEY,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        content_type="application/json; charset=utf-8",
    )


def _summary_from_parsed(
    *,
    digest_id: str,
    filename: str,
    parsed: dict[str, Any],
    uploaded_at: str,
    updated_at: str,
    content_hash: str,
) -> dict[str, Any]:
    return {
        "id": digest_id,
        "title": parsed["title"],
        "slug": parsed["slug"],
        "filename": filename,
        "generated_date": parsed["generated_date"],
        "uploaded_at": uploaded_at,
        "updated_at": updated_at,
        "content_hash": content_hash,
        "story_count": parsed["story_count"],
        "section_count": parsed["section_count"],
        "sections": [
            {"name": section["name"], "story_count": len(section.get("stories", []))}
            for section in parsed.get("sections", [])
        ],
    }


def _sort_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("generated_date") or ""),
            str(item.get("updated_at") or ""),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )


def save_digest(markdown: str, *, filename: str | None = None) -> dict[str, Any]:
    """Store or replace a digest and return its detail payload."""
    now = _now_utc()
    now_iso = now.isoformat().replace("+00:00", "Z")
    parsed = parse_digest_markdown(markdown, filename=filename, uploaded_at=now)
    digest_id = _digest_id(parsed["generated_date"], parsed["slug"])
    safe_filename = Path(filename or f"{digest_id}.md").name
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    manifest = _load_manifest()
    existing = next((item for item in manifest["digests"] if item.get("id") == digest_id), None)
    uploaded_at = str(existing.get("uploaded_at")) if existing else now_iso

    write_text(
        _digest_local_path(digest_id),
        _digest_gcs_key(digest_id),
        markdown,
        encoding="utf-8",
        content_type="text/markdown; charset=utf-8",
        metadata={
            "digest_id": digest_id,
            "generated_date": parsed["generated_date"],
            "title": parsed["title"],
        },
    )

    summary = _summary_from_parsed(
        digest_id=digest_id,
        filename=safe_filename,
        parsed=parsed,
        uploaded_at=uploaded_at,
        updated_at=now_iso,
        content_hash=content_hash,
    )

    manifest["digests"] = [item for item in manifest["digests"] if item.get("id") != digest_id]
    manifest["digests"].append(summary)
    manifest["digests"] = _sort_summaries(manifest["digests"])
    _write_manifest(manifest)

    return {**summary, "content": markdown, "parsed": parsed}


def _read_digest_content(digest_id: str) -> str:
    return read_text(_digest_local_path(digest_id), _digest_gcs_key(digest_id), encoding="utf-8")


def list_digests() -> dict[str, Any]:
    """Return digest summaries plus parsed story metadata for the library UI."""
    manifest = _load_manifest()
    summaries = _sort_summaries([item for item in manifest.get("digests", []) if isinstance(item, dict)])

    flat_stories: list[dict[str, Any]] = []
    for summary in summaries:
        digest_id = str(summary.get("id") or "")
        if not digest_id:
            continue
        try:
            content = _read_digest_content(digest_id)
            parsed = parse_digest_markdown(content, filename=str(summary.get("filename") or ""))
        except Exception:
            logger.debug("Skipping story parse for missing digest %s", digest_id, exc_info=True)
            continue
        for story in parsed.get("stories", []):
            flat_stories.append(
                {
                    **story,
                    "digest_id": digest_id,
                    "digest_title": summary.get("title"),
                    "generated_date": summary.get("generated_date"),
                }
            )

    return {
        "items": summaries,
        "stories": flat_stories,
        "counts": {
            "digests": len(summaries),
            "stories": len(flat_stories),
        },
    }


def get_digest(digest_id: str) -> dict[str, Any]:
    manifest = _load_manifest()
    summary = next((item for item in manifest.get("digests", []) if item.get("id") == digest_id), None)
    if not summary:
        raise FileNotFoundError(digest_id)

    content = _read_digest_content(digest_id)
    parsed = parse_digest_markdown(content, filename=str(summary.get("filename") or ""))
    return {**summary, "content": content, "parsed": parsed}


def delete_digest(digest_id: str) -> bool:
    manifest = _load_manifest()
    before = len(manifest.get("digests", []))
    manifest["digests"] = [item for item in manifest.get("digests", []) if item.get("id") != digest_id]
    removed_manifest = len(manifest["digests"]) != before
    removed_file = delete_file(_digest_local_path(digest_id), _digest_gcs_key(digest_id))
    if removed_manifest:
        _write_manifest(manifest)
    return removed_manifest or removed_file


def _summary_sort_key(summary: dict[str, Any]) -> tuple[str, str]:
    return (str(summary.get("generated_date") or ""), str(summary.get("updated_at") or ""))


def _truncate(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def get_report_context(
    *,
    days: int,
    now: datetime | None = None,
    max_digests: int = 5,
    max_stories: int = 80,
    notes_per_story: int = 2,
    max_text_chars: int = 700,
) -> dict[str, Any]:
    """Build a compact, deterministic digest context for report prompts."""
    manifest = _load_manifest()
    summaries = _sort_summaries([item for item in manifest.get("digests", []) if isinstance(item, dict)])
    if not summaries:
        return {"window_days": days, "digests": [], "counts": {"digests": 0, "stories": 0}, "fallback_used": False}

    ref = now or _now_utc()
    cutoff = (ref.date() - timedelta(days=max(0, days))).isoformat()
    recent = [item for item in summaries if str(item.get("generated_date") or "") >= cutoff]
    fallback_used = False
    if not recent:
        recent = summaries[:1]
        fallback_used = True

    selected = sorted(recent, key=_summary_sort_key, reverse=True)[:max_digests]
    remaining = max(1, max_stories)
    digest_payloads: list[dict[str, Any]] = []
    included_stories = 0

    for summary in selected:
        if remaining <= 0:
            break
        digest_id = str(summary.get("id") or "")
        if not digest_id:
            continue
        try:
            detail = get_digest(digest_id)
        except Exception:
            continue

        parsed = detail.get("parsed", {})
        sections_payload: list[dict[str, Any]] = []
        for section in parsed.get("sections", []):
            if remaining <= 0:
                break
            stories_payload: list[dict[str, Any]] = []
            for story in section.get("stories", []):
                if remaining <= 0:
                    break
                stories_payload.append(
                    {
                        "headline": _truncate(str(story.get("headline") or ""), max_text_chars),
                        "notes": [
                            _truncate(str(note), max_text_chars)
                            for note in (story.get("notes") or [])[:notes_per_story]
                        ],
                    }
                )
                remaining -= 1
                included_stories += 1
            if stories_payload:
                sections_payload.append({"name": section.get("name"), "stories": stories_payload})

        digest_payloads.append(
            {
                "id": summary.get("id"),
                "title": summary.get("title"),
                "generated_date": summary.get("generated_date"),
                "uploaded_at": summary.get("uploaded_at"),
                "updated_at": summary.get("updated_at"),
                "story_count": summary.get("story_count"),
                "sections": sections_payload,
            }
        )

    return {
        "window_days": days,
        "cutoff_date": cutoff,
        "fallback_used": fallback_used,
        "digests": digest_payloads,
        "counts": {
            "digests": len(digest_payloads),
            "stories": included_stories,
        },
    }
