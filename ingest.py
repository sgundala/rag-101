"""INGEST — the OFFLINE half.  Run this when documents change.

    uv run ingest.py           quiet
    uv run ingest.py --show    narrated, step by step  <- use this in class

    FILE -> TEXT -> CLEAN -> TOKENS -> CHUNKS -> VECTORS -> THE STORE
     (1)     (1)     (2)       (3)       (3)       (4)        (5)

Nobody is waiting on this, so it can be slow. It runs rarely. Everything it
does ends up in the store, and the store is the wall between the two halves.
"""

import sys
import time

import numpy as np

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_DIM
from step1_load import load
from step2_clean import clean
from step3_chunk import chunk, count_tokens
from step4_embed import embed, model_facts
from step5_store import save, where

SHOW = "--show" in sys.argv


def banner(n: int, name: str, arrow: str) -> None:
    if SHOW:
        print(f"\n{'─' * 70}")
        print(f"STEP {n} · {name}")
        print(f"  {arrow}")
        print(f"{'─' * 70}")


def run() -> None:
    t0 = time.time()

    if SHOW:
        facts = model_facts()
        print("=" * 70)
        print("OFFLINE — building the index")
        print("=" * 70)
        print(f"  embedding model   {facts['name']}")
        print(f"  its tokenizer     {facts['tokenizer']}")
        print(f"  max seq length    {facts['max_seq_length']} tokens"
              f"   <- longer chunks get SILENTLY TRUNCATED")
        print(f"  dimension         {facts['dimension']}"
              f"   <- config.EMBED_DIM = {EMBED_DIM}  (these MUST match)")

    # ── STEP 1 · LOAD ────────────────────────────────────────────────────
    banner(1, "LOAD", "file  ->  text")
    docs = load(DOCS_DIR)

    if not docs:
        raise SystemExit(
            f"No .md, .txt or .pdf files found in {DOCS_DIR}/\n"
            f"  Drop a document in there and run again."
        )

    if SHOW:
        for d in docs:
            print(f"  {d['source']:<34} {len(d['text']):>7,} chars")

    all_chunks: list[dict] = []

    for d in docs:
        # ── STEP 2 · CLEAN ───────────────────────────────────────────────
        before = len(d["text"])
        text = clean(d["text"])

        if SHOW:
            banner(2, f"CLEAN   ({d['source']})", "text  ->  clean text")
            print(f"  {before:,} chars  ->  {len(text):,} chars")
            print(f"  whitespace collapsed, tags stripped, PDF hyphens joined")
            print(f"  case KEPT on purpose — modern embedders are cased")

        # ── STEP 3 · CHUNK ───────────────────────────────────────────────
        n_tokens = count_tokens(text)
        pieces = chunk(text)

        if SHOW:
            banner(3, f"CHUNK   ({d['source']})",
                   "clean text  ->  TOKENS  ->  chunks")
            print(f"  {len(text):,} chars  =  {n_tokens:,} TOKENS"
                  f"   (the tokenizer's unit, not words)")
            print(f"  size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP}"
                  f"  ->  {len(pieces)} chunks")
            for i, p in enumerate(pieces[:3]):
                print(f"    chunk {i}  {count_tokens(p):>3} tok  "
                      f"| {p[:58]}...")
            if len(pieces) > 3:
                print(f"    ... and {len(pieces) - 3} more")

        all_chunks += [{"content": p, "source": d["source"]} for p in pieces]

    # ── STEP 4 · EMBED ───────────────────────────────────────────────────
    banner(4, "EMBED", "chunks  ->  vectors")
    texts = [c["content"] for c in all_chunks]
    vectors = embed(texts)

    if SHOW:
        print(f"  {len(texts)} chunks  ->  {vectors.shape[0]} vectors"
              f" of {vectors.shape[1]} numbers each")
        v = vectors[0]
        preview = ", ".join(f"{x:+.4f}" for x in v[:6])
        print(f"  chunk 0 became:  [{preview}, ...]"
              f"   ({len(v)} numbers total)")
        print(f"  its length is {np.linalg.norm(v):.4f}"
              f"   <- normalised to 1, so cosine = a dot product")

    # ── STEP 5 · STORE ───────────────────────────────────────────────────
    banner(5, "STORE", "vectors + text + source  ->  the store")
    save(all_chunks, vectors)

    if SHOW:
        print(f"  wrote {len(all_chunks)} rows to  {where()}")
        print(f"  each row:  the vector + the chunk text + the SOURCE FILE")
        print(f"  the source travels with the vector — that is why citations"
              f" are free")

        print(f"\n{'=' * 70}")
        print(f"OFFLINE DONE in {time.time() - t0:.1f}s —"
              f" {len(all_chunks)} chunks indexed.")
        print(f"The store is the wall. Nothing above here runs again until the")
        print(f"documents change.  Now:  uv run ask.py \"your question\" --show")
        print(f"{'=' * 70}")
    else:
        print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents "
              f"into {where()} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    run()
