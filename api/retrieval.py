"""
Embedding-based document retrieval for the AI agent.

Indexes thesis files, reports, and conversation summaries into a local
SQLite database with sentence-transformer embeddings for semantic search.
Uses all-MiniLM-L6-v2 (~80 MB, runs on CPU).

Follows the same connection pattern as memory_db.py (WAL mode, thread-safe).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import open_connection, use_postgres_state

logger = logging.getLogger("api.retrieval")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _REPO_ROOT / "data_cache" / "retrieval" / "embeddings.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# Lazy-loaded model
_model: Any = None
_model_lock = threading.Lock()
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# Chunking parameters
_MAX_CHUNK_TOKENS = 500
_OVERLAP_TOKENS = 100
_CHARS_PER_TOKEN = 4  # rough estimate

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    doc_type   TEXT NOT NULL,   -- thesis, weekly_report, daily_report, conversation_summary
    source_path TEXT,
    ticker     TEXT,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    heading     TEXT
)
"""

_CREATE_CHUNKS_IDX = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)
"""

_CREATE_DOCS_TYPE_IDX = """
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type)
"""

_CREATE_DOCS_TICKER_IDX = """
CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents(ticker)
"""

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    global _conn

    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    if _conn is None:
        with _lock:
            if _conn is None:
                _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute("PRAGMA foreign_keys=ON")
                _conn.row_factory = sqlite3.Row
                _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_DOCUMENTS)
    conn.execute(_CREATE_CHUNKS)
    conn.execute(_CREATE_CHUNKS_IDX)
    conn.execute(_CREATE_DOCS_TYPE_IDX)
    conn.execute(_CREATE_DOCS_TICKER_IDX)
    conn.commit()


# ---------------------------------------------------------------------------
# Embedding model (lazy load)
# ---------------------------------------------------------------------------


def _get_model():
    """Load the sentence-transformer model on first use."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence-transformer model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def _embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return _embed([text])[0]


# ---------------------------------------------------------------------------
# Serialization helpers (embedding ↔ blob)
# ---------------------------------------------------------------------------


def _embedding_to_blob(vec: list[float]) -> bytes:
    """Pack float32 vector into bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Unpack bytes into float32 vector."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_by_headings(content: str) -> list[tuple[str | None, str]]:
    """Split markdown content by ## headings. Returns (heading, text) pairs."""
    lines = content.split("\n")
    chunks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            # Save previous chunk
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    chunks.append((current_heading, text))
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last chunk
    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append((current_heading, text))

    return chunks


def _chunk_by_window(text: str, heading: str | None = None) -> list[tuple[str | None, str]]:
    """Split text into overlapping windows based on character count."""
    max_chars = _MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN
    overlap_chars = _OVERLAP_TOKENS * _CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [(heading, text)]

    chunks: list[tuple[str | None, str]] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((heading, chunk_text))
        start += max_chars - overlap_chars

    return chunks


def _chunk_document(content: str, doc_type: str) -> list[tuple[str | None, str]]:
    """Chunk a document based on its type.

    Markdown docs (thesis, reports): split by ## headings, then window large sections.
    Plain text (summaries): window-based chunking.
    """
    if doc_type in ("thesis", "weekly_report", "daily_report"):
        heading_chunks = _chunk_by_headings(content)
        # Sub-chunk large sections
        max_chars = _MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN
        result: list[tuple[str | None, str]] = []
        for heading, text in heading_chunks:
            if len(text) > max_chars:
                result.extend(_chunk_by_window(text, heading))
            else:
                result.append((heading, text))
        return result
    else:
        return _chunk_by_window(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_document(
    doc_type: str,
    content: str,
    ticker: str | None = None,
    source_path: str | None = None,
    doc_id: str | None = None,
) -> str:
    """Index a document: chunk it, embed chunks, store in SQLite.

    If doc_id is provided and already exists, the document is re-indexed
    (old chunks deleted, new ones created).

    Returns the doc_id.
    """
    if not content or not content.strip():
        raise ValueError("Cannot index empty content")

    conn = _get_conn()
    did = doc_id or str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    # Chunk the document
    raw_chunks = _chunk_document(content, doc_type)
    if not raw_chunks:
        raise ValueError("Document produced no chunks")

    # Embed all chunks in one batch
    chunk_texts = [text for _, text in raw_chunks]
    embeddings = _embed(chunk_texts)

    if use_postgres_state():
        _pg_index_document(
            doc_id=did,
            doc_type=doc_type,
            content=content,
            ticker=ticker,
            source_path=source_path,
            raw_chunks=raw_chunks,
            embeddings=embeddings,
            updated_at=now,
        )
        logger.info(
            "Indexed doc_id=%s type=%s ticker=%s chunks=%d",
            did,
            doc_type,
            ticker,
            len(raw_chunks),
        )
        return did

    with _lock:
        # Delete existing chunks if re-indexing
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (did,))

        # Upsert the document record
        conn.execute(
            """
            INSERT INTO documents (doc_id, doc_type, source_path, ticker, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                content = excluded.content,
                updated_at = excluded.updated_at,
                source_path = excluded.source_path,
                ticker = excluded.ticker
            """,
            (did, doc_type, source_path, ticker, content, now, now),
        )

        # Insert chunks
        for i, ((heading, text), emb) in enumerate(zip(raw_chunks, embeddings, strict=True)):
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, embedding, heading)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), did, i, text, _embedding_to_blob(emb), heading),
            )

        conn.commit()

    logger.info(
        "Indexed doc_id=%s type=%s ticker=%s chunks=%d",
        did,
        doc_type,
        ticker,
        len(raw_chunks),
    )
    return did


def search(
    query: str,
    doc_types: list[str] | None = None,
    tickers: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search indexed documents by semantic similarity.

    Returns top-K results with doc metadata, chunk content, heading, and score.
    """
    if not query or not query.strip():
        return []

    query_emb = _embed_single(query)
    if use_postgres_state():
        return _pg_search(query_emb, doc_types=doc_types, tickers=tickers, top_k=top_k)

    conn = _get_conn()

    # Build filter clause
    where_parts: list[str] = []
    params: list[Any] = []

    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        where_parts.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)

    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        placeholders = ",".join("?" for _ in upper_tickers)
        where_parts.append(f"UPPER(d.ticker) IN ({placeholders})")
        params.extend(upper_tickers)

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    with _lock:
        rows = conn.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.embedding,
                   c.heading, d.doc_type, d.ticker, d.source_path, d.created_at
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            {where_clause}
            """,
            params,
        ).fetchall()

    if not rows:
        return []

    # Compute cosine similarity (embeddings are already L2-normalized)
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        chunk_emb = _blob_to_embedding(row["embedding"])
        score = sum(a * b for a, b in zip(query_emb, chunk_emb, strict=True))
        scored.append((score, row))

    # Sort by score descending, take top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    results: list[dict[str, Any]] = []
    for score, row in top:
        results.append(
            {
                "doc_type": row["doc_type"],
                "ticker": row["ticker"],
                "heading": row["heading"],
                "content_snippet": row["content"][:500],
                "relevance_score": round(score, 4),
                "source_path": row["source_path"],
                "created_at": row["created_at"],
                "doc_id": row["doc_id"],
            }
        )

    return results


def delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks."""
    if use_postgres_state():
        with open_connection(register_pgvector=True) as conn:
            cur = conn.execute("DELETE FROM retrieval_documents WHERE doc_id = %s", (doc_id,))
            conn.commit()
            return bool(cur.rowcount > 0)

    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return cur.rowcount > 0


def get_indexed_count() -> dict[str, int]:
    """Return count of indexed documents by type."""
    if use_postgres_state():
        with open_connection(register_pgvector=True) as conn:
            rows = conn.execute(
                "SELECT doc_type, COUNT(*) AS cnt FROM retrieval_documents GROUP BY doc_type"
            ).fetchall()
        return {row["doc_type"]: row["cnt"] for row in rows}

    conn = _get_conn()
    with _lock:
        rows = conn.execute("SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type").fetchall()
    return {row["doc_type"]: row["cnt"] for row in rows}


def reindex_all_theses() -> int:
    """Scan investment_theses/ directory and index all thesis files.

    Useful for initial population or re-indexing after model change.
    Returns number of files indexed.
    """
    theses_dir = _REPO_ROOT / "investment_theses"
    if not theses_dir.is_dir():
        logger.warning("Thesis directory not found: %s", theses_dir)
        return 0

    count = 0
    for path in sorted(theses_dir.glob("*.md")):
        ticker = path.stem.upper()
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_id = f"thesis-{ticker}"
        try:
            index_document(
                doc_type="thesis",
                content=content,
                ticker=ticker,
                source_path=str(path),
                doc_id=doc_id,
            )
            count += 1
        except Exception:
            logger.warning("Failed to index thesis %s", ticker, exc_info=True)

    logger.info("Reindexed %d thesis files", count)
    return count


def _pg_index_document(
    *,
    doc_id: str,
    doc_type: str,
    content: str,
    ticker: str | None,
    source_path: str | None,
    raw_chunks: list[tuple[str | None, str]],
    embeddings: list[list[float]],
    updated_at: str,
) -> None:
    created_at = datetime.fromisoformat(updated_at)
    rows = [
        (str(uuid.uuid4()), doc_id, i, text, heading, emb)
        for i, ((heading, text), emb) in enumerate(zip(raw_chunks, embeddings, strict=True))
    ]
    with open_connection(register_pgvector=True) as conn:
        conn.execute("DELETE FROM retrieval_chunks WHERE doc_id = %s", (doc_id,))
        conn.execute(
            """
            INSERT INTO retrieval_documents (doc_id, doc_type, source_path, ticker, content, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                doc_type = EXCLUDED.doc_type,
                content = EXCLUDED.content,
                updated_at = EXCLUDED.updated_at,
                source_path = EXCLUDED.source_path,
                ticker = EXCLUDED.ticker
            """,
            (doc_id, doc_type, source_path, ticker, content, created_at, created_at),
        )
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO retrieval_chunks (chunk_id, doc_id, chunk_index, content, heading, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def _pg_search(
    query_emb: list[float],
    *,
    doc_types: list[str] | None,
    tickers: list[str] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    where_parts: list[str] = []
    params: list[Any] = [query_emb]
    if doc_types:
        where_parts.append("d.doc_type = ANY(%s)")
        params.append(doc_types)
    if tickers:
        where_parts.append("UPPER(d.ticker) = ANY(%s)")
        params.append([t.upper() for t in tickers])
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    params.extend([query_emb, top_k])

    with open_connection(register_pgvector=True) as conn:
        rows = conn.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content,
                   c.heading, d.doc_type, d.ticker, d.source_path, d.created_at,
                   1 - (c.embedding <=> %s) AS score
            FROM retrieval_chunks c
            JOIN retrieval_documents d ON c.doc_id = d.doc_id
            {where_clause}
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()

    return [
        {
            "doc_type": row["doc_type"],
            "ticker": row["ticker"],
            "heading": row["heading"],
            "content_snippet": row["content"][:500],
            "relevance_score": round(float(row["score"]), 4),
            "source_path": row["source_path"],
            "created_at": row["created_at"].isoformat()
            if hasattr(row["created_at"], "isoformat")
            else row["created_at"],
            "doc_id": row["doc_id"],
        }
        for row in rows
    ]
