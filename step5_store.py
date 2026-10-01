"""STEP 5 · STORE  —  the vectors go somewhere that survives.

                                                          (notes: Block 2B)
    VECTORS + TEXT + SOURCE  ->  >>> THE STORE

THE STORE IS THE WALL BETWEEN THE TWO HALVES OF RAG.

    OFFLINE  (ingest.py)   load -> clean -> chunk -> embed -> STORE
                                                               |
                                                          [ the store ]
                                                               |
    ONLINE   (ask.py)                    SEARCH -> augment -> generate

Offline writes it, rarely, when documents change — nobody is waiting, so it
can be slow. Online only reads it, on every question — so it must be fast.
A system that re-embeds the whole corpus on every question has collapsed that
boundary and will be unusably slow. That is the number-one beginner mistake.

WHY THE SOURCE IS STORED NEXT TO THE VECTOR
Because that is how CITATIONS BECOME FREE. The matching chunk already knows
which file it came from, so the answer can cite it without any extra work.

TWO BACKENDS, ONE INTERFACE
    "local"     a numpy .npz file. Zero setup, persists on disk, and you can
                see it. Cosine is a dot product over unit vectors — four
                lines you can read, and it is Week 1's cosine by hand.
    "pgvector"  real Postgres with the vector extension. Same save/search
                interface. The production shape.

Switch with STORE_BACKEND in config.py. The point of having both: the
BOUNDARY is the idea, the database is just technology.
"""

from pathlib import Path

import numpy as np

from config import EMBED_DIM, PG_DSN, STORE_BACKEND, STORE_PATH


# ══════════════════════════════════════════════════════════════════════
# LOCAL BACKEND — numpy on disk. The default.
# ══════════════════════════════════════════════════════════════════════

def _local_save(chunks: list[dict], vectors: np.ndarray) -> None:
    """Write vectors + text + source to one .npz file."""
    Path(STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        STORE_PATH,
        vectors=vectors,
        contents=np.array([c["content"] for c in chunks], dtype=object),
        sources=np.array([c["source"] for c in chunks], dtype=object),
    )


def _local_load() -> tuple[np.ndarray, list[str], list[str]]:
    if not Path(STORE_PATH).exists():
        raise SystemExit(
            f"No store at {STORE_PATH}.\n"
            f"  Run  uv run ingest.py --show  first — that is the offline half."
        )
    d = np.load(STORE_PATH, allow_pickle=True)
    return d["vectors"], list(d["contents"]), list(d["sources"])


def _local_search(qvec: np.ndarray, k: int) -> list[dict]:
    """Cosine similarity over every stored vector, return the top k.

    Every vector is unit length (step 4 normalised them), so the cosine
    similarity of two vectors IS their dot product. That is the whole
    calculation — the same cosine you wrote by hand in Week 1, now as one
    matrix multiply across the corpus.
    """
    vectors, contents, sources = _local_load()

    scores = vectors @ qvec                      # <- cosine, all rows at once
    top = np.argsort(-scores)[:k]                # highest score first

    return [{"content": contents[i],
             "source": sources[i],
             "score": float(scores[i])} for i in top]


def _local_count() -> int:
    return 0 if not Path(STORE_PATH).exists() else len(_local_load()[1])


# ══════════════════════════════════════════════════════════════════════
# PGVECTOR BACKEND — the production shape. Needs a running Postgres.
# ══════════════════════════════════════════════════════════════════════
#
# The three distance operators, and the index opclass each one needs:
#
#     <=>   cosine distance          vector_cosine_ops   <- we use this
#     <->   L2 / Euclidean           vector_l2_ops
#     <#>   negative inner product   vector_ip_ops
#
# The operator you QUERY with must match the opclass your INDEX was built
# with, or Postgres silently ignores the index — a classic silent slowdown.

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    content   text   NOT NULL,
    source    text   NOT NULL,          -- metadata = ordinary columns
    embedding vector({EMBED_DIM}) NOT NULL   -- MUST equal EMBED_DIM
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# `1 - (embedding <=> %s)` turns cosine DISTANCE (smaller = closer) into a
# friendly 0-1 similarity score. We sort by the distance, ascending.
SEARCH_SQL = """
SELECT content, source, 1 - (embedding <=> %s) AS score
FROM chunks
ORDER BY embedding <=> %s
LIMIT %s
"""


def _pg_connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(PG_DSN)
    register_vector(conn)        # once per connection, or vectors go in as text
    return conn


def _pg_save(chunks: list[dict], vectors: np.ndarray) -> None:
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("TRUNCATE chunks")       # a fresh index each ingest
            cur.executemany(
                "INSERT INTO chunks (content, source, embedding) VALUES (%s,%s,%s)",
                [(c["content"], c["source"], v)
                 for c, v in zip(chunks, vectors)],
            )
        conn.commit()


def _pg_search(qvec: np.ndarray, k: int) -> list[dict]:
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SEARCH_SQL, (qvec, qvec, k))
            return [{"content": c, "source": s, "score": float(sc)}
                    for c, s, sc in cur.fetchall()]


def _pg_count() -> int:
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return cur.fetchone()[0]


# ══════════════════════════════════════════════════════════════════════
# THE INTERFACE — the rest of the system only ever calls these three.
# ══════════════════════════════════════════════════════════════════════

_BACKENDS = {
    "local":    (_local_save, _local_search, _local_count),
    "pgvector": (_pg_save,    _pg_search,    _pg_count),
}

if STORE_BACKEND not in _BACKENDS:
    raise SystemExit(f"Unknown STORE_BACKEND {STORE_BACKEND!r}. "
                     f"Pick one of: {list(_BACKENDS)}")

_save, _search, _count = _BACKENDS[STORE_BACKEND]


def save(chunks: list[dict], vectors: np.ndarray) -> None:
    """Write chunks and their vectors to the store. The end of OFFLINE."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    _save(chunks, vectors)


def search(qvec: np.ndarray, k: int) -> list[dict]:
    """Return the k nearest stored chunks. The start of ONLINE."""
    return _search(qvec, k)


def count() -> int:
    """How many chunks are in the store right now."""
    return _count()


def where() -> str:
    """A human-readable location, for --show."""
    return STORE_PATH if STORE_BACKEND == "local" else PG_DSN
