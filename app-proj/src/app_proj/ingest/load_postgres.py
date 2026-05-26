"""Write a Polars DataFrame to a Postgres table.

Drop-and-recreate semantics: the target table is replaced and its schema
inferred from the DataFrame. Uses the ADBC writer (fastest Polars path
to Postgres; needs ``adbc-driver-postgresql``).

Connection config is read from environment variables; defaults match the
local Docker setup documented in the repo README.
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus

import polars as pl
from dotenv import load_dotenv

load_dotenv()


def get_connection_uri() -> str:
    """Build a libpq URI from PG_* environment variables.

    Defaults to the local Docker setup: ``adm:pwd@localhost:5432/lans-db``.
    The database name is URL-quoted so values containing ``-`` (like
    ``lans-db``) work without manual escaping.
    """
    user = os.getenv("PG_USER", "adm")
    pwd = os.getenv("PG_PASSWORD", "pwd")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DATABASE", "lans-db")
    return f"postgresql://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{quote_plus(db)}"


def write_df_to_postgres(
    df: pl.DataFrame,
    table_name: str,
    connection_uri: str | None = None,
) -> int:
    """Drop & recreate ``table_name`` from ``df``'s schema, then bulk-write.

    Parameters
    ----------
    df
        Source DataFrame. Schema is inferred directly; nested / list
        columns are not supported by the writer — flatten upstream.
    table_name
        Target Postgres table. Will be dropped if it already exists.
    connection_uri
        Optional libpq URI. Falls back to ``get_connection_uri()``.

    Returns
    -------
    int
        Number of rows written (as reported by ``write_database``).
    """
    uri = connection_uri or get_connection_uri()

    rows_written = df.write_database(
        table_name=table_name,
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    return rows_written


__all__ = ["get_connection_uri", "write_df_to_postgres"]
