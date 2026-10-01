# RAG 101 — Week 3, Day 1

RAG from first principles, as **one file per step**. `ls` is the pipeline:

```
step1_load.py      file        ->  text
step2_clean.py     text        ->  clean text
step3_chunk.py     clean text  ->  TOKENS  ->  chunks
step4_embed.py     chunks      ->  vectors        (384 numbers each)
step5_store.py     vectors     ->  the store      <- THE BOUNDARY
step6_search.py    question    ->  vector  ->  nearest chunks
step7_augment.py   chunks      ->  one grounded prompt
step8_generate.py  prompt      ->  the answer
```

Two runnable halves:

```
ingest.py    OFFLINE   steps 1-5   runs when documents change
ask.py       ONLINE    steps 6-8   runs on every question
```

---

## Before class — download the model once

**This is the one thing that can stop the class.** `all-MiniLM-L6-v2` is fetched from
huggingface.co on first use, about 90 MB, cached in `~/.cache/huggingface`. On a
restricted network it fails.

Run this the night before, on the network you'll teach from:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

After that it's offline and instant forever.

The generator is local too, so pull it the same night (about 13 GB):

```bash
ollama pull gpt-oss:20b
```

## Setup

```bash
uv sync                        # installs everything
ollama pull gpt-oss:20b        # the local LLM — once, and keep Ollama running
uv run ingest.py --show        # OFFLINE — build the index, narrated
uv run ask.py "what is grounding?" --show   # ONLINE — answer, narrated
```

Default LLM is **local**: `gpt-oss:20b` served by [Ollama](https://ollama.com). No API
key, no cost, works offline. On a laptop that can't run a 20B model, point `model` at a
smaller one you've pulled (`ollama_chat/llama3.1:8b`), or change `LLM_PROVIDER` in
`config.py` to Groq, Anthropic or OpenAI and put that provider's key in `.env` — one
line, nothing else moves.

Default store is **local** — a numpy file at `store/index.npz`. Zero setup, persists on
disk, and you can see it. Switch `STORE_BACKEND` to `"pgvector"` in `config.py` for real
Postgres; `step5_store.py` has the full SQL, the HNSW index and all three distance
operators.

---

## Teaching it in class — the three commands

### 1 · Show the pipeline, offline

```bash
uv run ingest.py --show
```

Prints every stage with real numbers. The screen does the talking:

```
STEP 3 · CHUNK   (rag_101.md)
  clean text  ->  TOKENS  ->  chunks
  2,894 chars  =  1,281 TOKENS   (the tokenizer's unit, not words)
  size=256 tokens, overlap=32  ->  6 chunks
    chunk 0  256 tok  | # RAG 101 — Retrieval-Augmented Generation ## Why RAG exis...

STEP 4 · EMBED
  10 chunks  ->  10 vectors of 384 numbers each
  chunk 0 became:  [+0.0213, -0.0891, +0.0412, ...]   (384 numbers total)
  its length is 1.0000   <- normalised to 1, so cosine = a dot product

STEP 5 · STORE
  wrote 10 rows to  store/index.npz
  the source travels with the vector — that is why citations are free
```

### 2 · Show the boundary

```bash
ls -la store/
```

That file is the wall. Then point at `ask.py` and note what it does **not** import:
no loader, no chunker. It only reads. *A system that re-embeds the corpus on every
question has collapsed the boundary and will be unusably slow* — that's the
number-one beginner mistake, and the file layout makes it impossible here.

### 3 · Show the pipeline, online

```bash
uv run ask.py "what is grounding and where does it live?" --show
```

```
STEP 6 · SEARCH
  the question became 384 numbers — by the SAME embed() the documents used
  that is why they are comparable at all
  compared against 10 stored vectors, cosine, top 3:
    [1]  score 0.612  rag_101.md
```

---

## Add a PDF — do this live

Drop any PDF into `docs/` and re-run `uv run ingest.py --show`. Nothing else changes:
`step1_load.py` is the **only** file that knows what a file format is, which is the
whole point of having a loader.

It's also the best demo of why `step2_clean.py` exists. A PDF has no paragraphs, just
positioned glyphs, so the extracted text is a mess. Watch a word broken across a line
wrap get repaired:

```
RAW:      'Chunk size is measured in tokens, not words or charac-\nters. A good...'
CLEANED:  'Chunk size is measured in tokens, not words or characters. A good...'
```

A scanned PDF with no text layer is skipped with a message rather than silently
indexed as empty — that case needs OCR, which is a different tool.

---

## The number that must match

```
the model outputs 384   ==   the store holds 384   ==   the question is embedded to 384
```

One constant (`EMBED_DIM` in `config.py`) and one `embed()` function (step 4), which
both the documents and the question call. There is no way for them to drift apart.

Demo the failure deliberately: set `EMBED_DIM = 768` in `config.py` and run ingest.

```
DIMENSION MISMATCH — the silent failure, caught early.
  all-MiniLM-L6-v2 outputs 384 numbers.
  config.EMBED_DIM says 768.
  Fix EMBED_DIM and re-run ingest.py — a model switch is a full re-index.
```

In a system without that guard you'd get silent garbage instead of an error. This is
the first thing to check when retrieval returns nonsense.

---

## One place the notes contradict themselves

The memorised chain is **load → chunk → clean**. But the notes' own Block 1C says
*"structural cleaning belongs before chunking so a boundary never lands mid-tag"* —
which is correct, and the opposite order.

This code does **load → clean → chunk** (steps 1, 2, 3) because that's what's right:
clean first, then cut. Worth saying out loud in class — the mnemonic rhymes better
one way and the code has to be the other way.

---

## Where each piece came from

| This step | You built it in |
|---|---|
| step 3 · chunking in **tokens** | Week 1 — the tokenizer |
| step 4 · text → vector | Week 1 — embeddings |
| step 6 · cosine similarity | Week 1 — you wrote it by hand |
| top-K / "lost in the middle" | Week 1 — attention |
| step 7 · instruction + context | Week 2 — the system prompt block |
| step 7 · grounding as a guardrail | Week 2 — "recommend; a human decides" |
| step 8 · the model call + config | Weeks 1 & 2 — unchanged |

Nothing here is new except the assembly. Say that on the first slide and again on the
last.

**Forward to Day 3:** `retrieve()` in `step6_search.py` has a typed signature and a
tool-style docstring on purpose. On Day 3 you wrap it as a tool an agent calls — the
Week 2 Day 2 idea. RAG doesn't get rebuilt; it gets *called*. When an agent has several
tools, that docstring is what drives tool selection.

---

## Errors, decoded

| What you see | What it means |
|---|---|
| `Could not load the embedding model` | No huggingface.co access. Pre-download it (top of this file) |
| `No store at store/index.npz` | Run `ingest.py` first — that's the offline half |
| `DIMENSION MISMATCH` | `EMBED_DIM` disagrees with the model. A model switch is a full re-index |
| `generation failed` + "Is Ollama running" | Start the Ollama app and `ollama pull gpt-oss:20b` |
| `No GROQ_API_KEY found` | Only on a cloud provider: put that provider's key in `.env` |
| `No .md, .txt or .pdf files found` | `docs/` is empty — drop a document in |
| Scores all look low (~0.1) | Normal for unrelated questions. Real MiniLM gives 0.4–0.8 on a good match |
