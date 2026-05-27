# Enrichment (Phase EN)

Offline batch job. Runs once (or on schedule) on the full `stiftelser` table.
Output feeds `stiftelser_enriched`. No query-time involvement.

---

## Purpose

Länsstyrelsen rows are sparse. Enrichment adds the structured fields needed
for pre-filtering **before** vector search runs in Phase 2.

Key fields that unlock filtering:
- `accepts_unsolicited` — eliminates closed foundations before ANN
- `deadline_typical` — eliminates expired/irrelevant timing
- `grant_size_sek` — optional range filter
- `target_groups` + `notes` — fed back into chunks pipeline for richer embeddings

---

## Target table

```sql
CREATE TABLE stiftelser_enriched (
    org_nr              VARCHAR PRIMARY KEY,
    website             VARCHAR,
    deadline_typical    VARCHAR,   -- Swedish month, "rolling", UNKNOWN
    grant_size_sek      VARCHAR,   -- "50000-200000" or UNKNOWN
    accepts_unsolicited VARCHAR,   -- TRUE / FALSE / UNKNOWN
    target_groups       VARCHAR,   -- comma-separated labels, feeds embeddings
    notes               VARCHAR,   -- one sentence max, feeds embeddings
    pdf_links           VARCHAR,   -- semicolon-separated URLs
    enriched_at         TIMESTAMP DEFAULT now(),
    gemini_model        VARCHAR,
    FOREIGN KEY (org_nr) REFERENCES stiftelser(org_nr)
);
```

---

## Tool

Gemini 2.0 Flash with `google_search_retrieval` grounding.
One prompt per row → one CSV row back → INSERT with ON CONFLICT DO UPDATE.

### Prompt contract

Input fields passed to Gemini: `NAMN`, `org_nr`, `ANDAMAL`, `LANKOD` (decoded
to county name via `decode.py` for search context — decoded name is NOT stored back).

Return format: single CSV row, no header, column order matches table above
(excluding `enriched_at`, `gemini_model`). All fields mandatory; use `UNKNOWN`
never blank.

---

## Run condition

```sql
WHERE e.org_nr IS NULL   -- only unenriched rows, no other filter
```

No similarity pre-selection. Enrichment is source-of-truth, not query-dependent.

---

## Back-feed into chunks pipeline

After enrichment, re-run `chunk_embed.py` with:
- `source_table = 'stiftelser_enriched'`
- `source_column = 'target_groups'` (skip if UNKNOWN)
- `source_column = 'notes'` (skip if UNKNOWN)

Uses the existing generic chunks schema — zero schema work (Decision 3).

---

## Phase 2 pre-filter (how enrichment is consumed)

```sql
SELECT s.*
FROM stiftelser s
JOIN stiftelser_enriched e USING (org_nr)
WHERE e.accepts_unsolicited != 'FALSE'
-- optionally: AND e.deadline_typical not in ('UNKNOWN') for tighter demos
```

Vector search runs on this filtered set, not the full 17k.

---

## Constraints

- `decode.py` must exist before enrichment script runs (pure function, no DB calls)
- Rate-limit: `time.sleep(0.4)` between Gemini calls
- Spot-check 10–20 rows manually before bulk run — Gemini invents websites for obscure foundations
- `gemini_model` column preserved for future re-run with better model
