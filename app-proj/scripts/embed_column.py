"""Chunk + embed a text column of a Postgres table.

Run from the project root::

    uv run python scripts/embed_column.py \
        --source-table stiftelser \
        --text-column andamal \
        --chunks-table chunks
"""

from __future__ import annotations

import argparse

from app_proj.embed.chunk_embed import (
    DEFAULT_DIM,
    DEFAULT_MODEL,
    build_chunks_table,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-table", required=True)
    p.add_argument("--text-column", required=True)
    p.add_argument("--id-column", default="id")
    p.add_argument("--chunks-table", default="chunks")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    print(
        f"Embedding {args.source_table}.{args.text_column} "
        f"with model {args.model} (dim={args.dim}) …"
    )
    n = build_chunks_table(
        source_table=args.source_table,
        text_column=args.text_column,
        id_column=args.id_column,
        chunks_table=args.chunks_table,
        model_name=args.model,
        embedding_dim=args.dim,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
    )
    print(f"  → {n:,} chunks written to '{args.chunks_table}'")


if __name__ == "__main__":
    main()
