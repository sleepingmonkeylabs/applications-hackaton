"""Ingest the Länsstyrelsen stiftelser JSON dump into Postgres.

Run from the project root::

    uv run python scripts/ingest_stiftelser.py \
        --json ../stiftelser_2026-05-26_1154.json \
        --table stiftelser
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app_proj.ingest.load_postgres import write_df_to_postgres
from app_proj.ingest.parse_json import parse_stiftelser_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        required=True,
        type=Path,
        help="Path to the stiftelser JSON dump.",
    )
    p.add_argument(
        "--table",
        default="stiftelser",
        help="Target Postgres table name (default: stiftelser).",
    )
    args = p.parse_args()

    print(f"Parsing {args.json} …")
    df = parse_stiftelser_json(args.json)
    print(f"  → {df.height:,} rows, {df.width} columns")
    print(f"  columns: {df.columns}")

    print(f"Writing to Postgres table '{args.table}' (drop & recreate) …")
    rows = write_df_to_postgres(df, args.table)
    print(f"  → {rows:,} rows written")


if __name__ == "__main__":
    main()
