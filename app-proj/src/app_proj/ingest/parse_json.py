"""Parse the Länsstyrelsen stiftelser JSON dump into a flat Polars DataFrame.

The dump has shape::

    {
      "STIFTELSER": [
        {
          "STIFTELSE": { ...scalar fields... },
          "ROLLER":    [ {"ROLL": {...}}, ... ],   # board members (dropped v1)
          "FIRMOR":    [ {"FIRMA": {...}}, ... ]   # subsidiary businesses (dropped v1)
        },
        ...
      ]
    }

For v1 we keep only the STIFTELSE scalar fields — that's where ANDAMAL
(the embed target) lives. ROLLER / FIRMOR are dropped; they can be
re-introduced as normalized side-tables later.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl


def parse_stiftelser_json(json_path: str | Path) -> pl.DataFrame:
    """Read the Länsstyrelsen stiftelser JSON dump → flat Polars DataFrame.

    Parameters
    ----------
    json_path
        Path to the JSON file (e.g. ``stiftelser_2026-05-26_1154.json``).

    Returns
    -------
    pl.DataFrame
        One row per foundation. Column names are lowercased for SQL
        friendliness. Nested ROLLER / FIRMOR are not included.
    """
    path = Path(json_path)
    with path.open("rb") as f:
        payload = orjson.loads(f.read())

    raw = payload.get("STIFTELSER", [])

    # Each item wraps the scalar dict under "STIFTELSE". A handful of
    # malformed records may be missing it — skip those.
    records: list[dict] = [
        item["STIFTELSE"]
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("STIFTELSE"), dict)
    ]

    if not records:
        return pl.DataFrame()

    df = pl.from_dicts(records, infer_schema_length=None)

    # Lowercase column names for Postgres ergonomics (case-folded
    # identifiers don't need quoting downstream).
    df = df.rename({c: c.lower() for c in df.columns})

    return df


__all__ = ["parse_stiftelser_json"]
