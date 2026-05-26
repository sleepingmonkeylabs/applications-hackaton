"""Build a chunks + embeddings reference table for a given text column.

Given ``(source_table, text_column)``, this module:

1. Reads ``(id_column, text_column)`` from ``source_table``.
2. Splits each text value into chunks at paragraph / sentence boundaries.
3. Embeds chunks with a Sentence-Transformers model
   (default: ``intfloat/multilingual-e5-base``, 768-d, Swedish-capable).
4. Drops and recreates ``chunks_table`` and writes
   ``(source_id, source_table, source_column, chunk_index, chunk_text,
   embedding, model_name)``.
5. Adds an HNSW index on ``embedding`` using cosine distance.

The separate-table design (vs a pgvector column on the source table)
keeps multi-chunk-per-row, lets us re-embed without touching the source,
and supports holding embeddings from multiple models side by side.
"""

from __future__ import annotations

import re
from typing import Iterable

import polars as pl
from pgvector.psycopg import register_vector
from psycopg import Connection, connect
from sentence_transformers import SentenceTransformer

from app_proj.ingest.load_postgres import get_connection_uri

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_DIM = 768
DEFAULT_CHUNK_CHARS = 600


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split ``text`` on paragraph then sentence boundaries.

    Foundation purpose texts (ANDAMAL) are typically short, but a handful
    string together several distinct purposes — those want to be split.
    Splitting on blank lines first preserves authorial structure;
    sentence-level fallback handles paragraphs that exceed ``max_chars``.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sent in sentences:
            if not buf:
                buf = sent
            elif len(buf) + 1 + len(sent) <= max_chars:
                buf = f"{buf} {sent}"
            else:
                chunks.append(buf)
                buf = sent
        if buf:
            chunks.append(buf)

    return chunks


def _e5_prefix(model_name: str, texts: Iterable[str], kind: str = "passage") -> list[str]:
    """e5 models expect ``"passage: "`` / ``"query: "`` prefixes."""
    if "e5" in model_name.lower():
        return [f"{kind}: {t}" for t in texts]
    return list(texts)


def _create_chunks_table(
    conn: Connection, chunks_table: str, embedding_dim: int
) -> None:
    """Drop & recreate the chunks table and ensure pgvector is installed."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f'DROP TABLE IF EXISTS "{chunks_table}"')
        cur.execute(
            f"""
            CREATE TABLE "{chunks_table}" (
                chunk_id      BIGSERIAL PRIMARY KEY,
                source_id     BIGINT      NOT NULL,
                source_table  TEXT        NOT NULL,
                source_column TEXT        NOT NULL,
                chunk_index   INT         NOT NULL,
                chunk_text    TEXT        NOT NULL,
                embedding     vector({embedding_dim}) NOT NULL,
                model_name    TEXT        NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def build_chunks_table(
    source_table: str,
    text_column: str,
    id_column: str = "id",
    chunks_table: str = "chunks",
    model_name: str = DEFAULT_MODEL,
    embedding_dim: int = DEFAULT_DIM,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    batch_size: int = 64,
    connection_uri: str | None = None,
) -> int:
    """Build the chunks table for one text column of one source table.

    Parameters
    ----------
    source_table
        Table to read text from (e.g. ``stiftelser``).
    text_column
        Column holding the text to chunk + embed (e.g. ``andamal``).
    id_column
        Primary key column on ``source_table`` (default ``id``).
    chunks_table
        Target chunks table name. Will be dropped & recreated.
    model_name
        Sentence-Transformers model. Default is multilingual-e5-base.
    embedding_dim
        Vector dimensionality. Must match the model.
    max_chars
        Max chars per chunk before sentence-splitting kicks in.
    batch_size
        Embedding batch size.
    connection_uri
        Optional libpq URI. Falls back to ``get_connection_uri()``.

    Returns
    -------
    int
        Number of chunk rows written.
    """
    uri = connection_uri or get_connection_uri()

    # Pull source rows via Polars + ADBC (same driver we use for writes).
    rows_df = pl.read_database_uri(
        query=(
            f'SELECT "{id_column}" AS source_id, '
            f'"{text_column}" AS source_text '
            f'FROM "{source_table}"'
        ),
        uri=uri,
        engine="adbc",
    )

    # Build the (still un-embedded) chunk records.
    chunk_records: list[dict] = []
    for row in rows_df.iter_rows(named=True):
        text_val = row["source_text"]
        if text_val is None:
            continue
        for i, piece in enumerate(chunk_text(str(text_val), max_chars=max_chars)):
            chunk_records.append(
                {
                    "source_id": row["source_id"],
                    "chunk_index": i,
                    "chunk_text": piece,
                }
            )

    if not chunk_records:
        return 0

    # Embed in batches.
    model = SentenceTransformer(model_name)
    prefixed = _e5_prefix(model_name, (r["chunk_text"] for r in chunk_records))
    vectors = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Write everything in one transaction.
    with connect(uri) as conn:
        _create_chunks_table(conn, chunks_table, embedding_dim)
        register_vector(conn)

        with conn.cursor() as cur:
            insert_sql = (
                f'INSERT INTO "{chunks_table}" '
                "(source_id, source_table, source_column, chunk_index, "
                " chunk_text, embedding, model_name) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            )
            cur.executemany(
                insert_sql,
                [
                    (
                        rec["source_id"],
                        source_table,
                        text_column,
                        rec["chunk_index"],
                        rec["chunk_text"],
                        vec,
                        model_name,
                    )
                    for rec, vec in zip(chunk_records, vectors)
                ],
            )

            # HNSW + cosine — pair this with normalized embeddings above.
            cur.execute(
                f'CREATE INDEX ON "{chunks_table}" '
                "USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                f'CREATE INDEX ON "{chunks_table}" (source_table, source_id)'
            )

    return len(chunk_records)


__all__ = ["chunk_text", "build_chunks_table", "DEFAULT_MODEL", "DEFAULT_DIM"]
