# app-proj

Länsstyrelsen stiftelser ingest + embed pipeline for the Stadsmissionen grant-matching hackathon.

## Modules

- `src/app_proj/ingest/parse_json.py` — read Länsstyrelsen JSON dump → Polars DataFrame (nested ROLLER / FIRMOR dropped for v1).
- `src/app_proj/ingest/load_postgres.py` — Polars DataFrame → Postgres table (drop & recreate from DF schema, ADBC writer).
- `src/app_proj/embed/chunk_embed.py` — given `(source_table, text_column)`, build a `chunks` table with pgvector embeddings + HNSW cosine index.

## Postgres

Expected to be running locally (see `.env.example`):

```
docker run --name local-postgres \
  -e POSTGRES_USER=adm \
  -e POSTGRES_PASSWORD=pwd \
  -e POSTGRES_DB=lans-db \
  -p 5432:5432 \
  -v /Users/dimka/Documents/Claude/Projects/applications-hackaton/pg-persist:/var/lib/postgresql \
  -d postgres:alpine
```

The `chunks` table requires the `vector` extension; `chunk_embed.py` runs `CREATE EXTENSION IF NOT EXISTS vector` on init. If the Alpine image doesn't ship pgvector, switch to `pgvector/pgvector:pg16` or install the extension into the running container.

## Install

```
cd app-proj
uv sync
```
