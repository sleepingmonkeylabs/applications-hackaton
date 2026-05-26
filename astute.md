# Astute decisions

A collected list of small-but-load-bearing choices in this project — the ones where picking the obvious-looking alternative would have quietly cost us.

---

## 1. Pipeline separation: Ingest vs Ask

The KB-building pipeline (Phase 1) and the retrieval/drafting pipeline (Phase 2) are independent. Ingest can be re-run, swapped, or extended without touching Ask, and vice versa.

**ELI5:** It's the same reason kitchens have a prep station and a service station. You can scale them differently, swap one out, and a fire on one side doesn't burn the other.

---

## 2. Postgres as the in-app backend

Postgres runs as `docker-up` locally for dev, and ships as an extra service inside the app container when deployed. One database engine covers structured rows, vectors (pgvector), full-text search, and joins — no separate vector DB to host or sync.

**ELI5:** One filing cabinet instead of three. Postgres can do all of it, so we don't pay the cost of keeping a vector DB, a SQL DB, and a search index consistent with each other.

---

## 3. Chunks table is generic, not stiftelser-specific

The schema is `(source_id, source_table, source_column, chunk_index, chunk_text, embedding, model_name, ...)`. So the same table holds chunks for *any* source table and *any* embedding model side-by-side.

**ELI5:** Instead of one drawer labeled "foundation chunks," it's a drawer labeled "chunks of anything, tagged with where they came from." Adding programmes, news articles, or a second embedding model later is zero schema work.

---

## 4. Multilingual embedding + e5 passage/query asymmetry

The model (`intfloat/multilingual-e5-base`) was trained on Swedish, so it actually understands ANDAMAL text. It was *also* trained with `"passage: "` and `"query: "` prefixes — the helper `_e5_prefix` in `chunk_embed.py` respects this at ingest (passage), and the same helper will be reused for queries in Phase 2 (query).

**ELI5:** It's like a model that learned "stored docs" and "search questions" as two different languages. We speak both to it correctly. Skipping the prefixes still produces vectors, just slightly worse ones — and you'd never notice.

---

## 5. Normalization + cosine

`normalize_embeddings=True` paired with `vector_cosine_ops` on an HNSW index. Cosine is length-invariant by definition, so this is a performance/ergonomics choice, not a correctness one.

**ELI5:** Once every vector has length 1, cosine collapses to a plain dot product and the database has less to compute. Bonus: scores live in `[-1, 1]` so thresholds mean the same thing across queries, and we can swap to inner-product distance later without re-embedding.

---

## 6. Reranking with grain change (chunks → foundations)

ANN retrieves the top-K *chunks*. We then group by `stiftelse_id`, aggregate scores, pull the *full structured row* (NAMN, full ANDAMAL, ORT, LÄN, TYP-decoded), and hand that to the LLM for rerank.

**ELI5:** The first pass finds promising paragraphs. The rerank steps back and judges whole foundations — using structured fields the embedding model never saw. So the LLM isn't re-voting on the same evidence; it's adding new information.

---

## 7. Paragraph-first, sentence-fallback chunking of ANDAMAL

ANDAMAL texts are short and semi-structured — blank lines separate distinct purposes. `chunk_text` splits on paragraphs first, only falling back to sentence splits when a paragraph exceeds `max_chars`.

**ELI5:** Foundation purpose statements already have natural seams (the blank lines authors put there). We split there first instead of using a blind 512-token window that would smear two distinct purposes into the same chunk.

---

## 8. Reverse direction is free

Stadsmissionen programmes are modeled symmetrically to stiftelser — just another "source" feeding into chunks. So the reverse query ("which foundations would these programmes interest?" vs "which programmes would interest this foundation?") is the same infrastructure with embed sides swapped.

**ELI5:** We didn't hardwire one side as the "query" and the other as the "documents." Both are just tables of text with embeddings — point the arrow either way.

---

## 9. TYP / LANKOD / KOMMUNKOD decoded at PULL, not at ingest

The JSON stores integer codes (`TYP=6`, `LANKOD=1`, `KOMMUNKOD=180`). We keep them raw in Postgres and translate to human-readable names only in the retrieval step, right before the LLM sees the row.

**ELI5:** Keep the receipt clean; write the notes on a sticky pad. If the lookup table changes, we update one function instead of re-ingesting the whole dump. The table stays byte-faithful to the source.

**Flag for v2.** No decoder exists in code yet — Phase 2 isn't built. When it lands:

- Put it in a pure module (`app_proj/decode.py`), no DB calls — so both retrieval *and* the future table enricher (EN) can import the same function without drifting apart.
- Pull SCB's official lookup tables for LANKOD / KOMMUNKOD; don't hand-type them.
- Note: `TYP=6` covers ~96% of rows (16,754 / 17,476), `TYP=3` covers ~4% (716), `TYP=-1` is a 6-row sentinel. Worth confirming the meanings before spending LLM tokens displaying a field that's mostly one value.
