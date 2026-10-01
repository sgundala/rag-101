"""CONFIG — every knob in the system, in one place.

Read this file first. Everything downstream imports from here, so there is
exactly one place to change a setting and no way for two parts of the system
to disagree.

THE NUMBER THAT MUST MATCH
    EMBED_DIM is the single most important value in a RAG system.

    the model outputs 384 numbers
      == the store holds vectors of 384
      == the question is embedded to 384

If any of those three disagree, you get silent garbage or a crash. It flows
from ONE constant here and ONE embed() function (step 4), so the documents
and the question can never be embedded differently.
"""

# ── step 4 · the embedder ────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384          # must equal the model's output. See the note above.

# Easy upgrade, same 384 dimension so no re-index needed:
#   EMBED_MODEL = "BAAI/bge-small-en-v1.5"
#
# Anything with a different dimension (bge-base = 768, OpenAI small = 1536)
# means: change EMBED_DIM, change the store, and RE-EMBED EVERYTHING. A model
# switch is a full re-index. That is the cost of switching.


# ── step 3 · chunking ───────────────────────────────────────────────────
# Measured in TOKENS, not words or characters — the unit from Week 1.
CHUNK_SIZE = 256         # also the model's max_seq_length; longer is truncated
CHUNK_OVERLAP = 32       # ~12%. Stops an idea being sliced in half.


# ── step 6 · retrieval ──────────────────────────────────────────────────
TOP_K = 3                # how many chunks to put in the prompt
# Too low: you miss the answer. Too high: noise crowds out signal, cost
# rises, and "lost in the middle" kicks in.


# ── step 5 · where vectors live ─────────────────────────────────────────
STORE_BACKEND = "local"  # "local" | "pgvector"

# "local"    a numpy file at STORE_PATH. Zero setup, persists on disk,
#            and you can see it. Right for learning and for small corpora.
# "pgvector" real Postgres. Needs a running server and the pgvector
#            extension. Same interface — see step5_store.py.

STORE_PATH = "store/index.npz"
PG_DSN = "postgresql://localhost/rag101"   # only used by the pgvector backend


# ── step 8 · the generator ──────────────────────────────────────────────
LLM_PROVIDER = "ollama"  # "ollama" | "groq" | "anthropic" | "openai"

# "ollama" runs on this machine: no API key, no cost, works offline. It needs
# the Ollama app running and the model pulled once. The other three are cloud
# APIs and need a key in .env.

LLM_PROVIDERS = {
    "ollama": {
        "env_var": None,     # local — there is no key
        "model": "ollama_chat/gpt-oss:20b",
        "signup": "ollama.com -> install, then:  ollama pull gpt-oss:20b",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "model": "groq/llama-3.1-8b-instant",
        "signup": "console.groq.com -> API Keys  (free tier)",
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "model": "anthropic/claude-sonnet-4-6",
        "signup": "console.anthropic.com -> API keys",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "model": "openai/gpt-4o-mini",
        "signup": "platform.openai.com -> API keys",
    },
}

TEMPERATURE = 0          # a grounded answer is a careful read, not a brainstorm
MAX_TOKENS = 600
# gpt-oss thinks before it answers, and that thinking is spent from this same
# budget. If answers come back cut off or empty, raise this.


# ── step 1 · where documents come from ──────────────────────────────────
DOCS_DIR = "docs"
# Drop .md, .txt or .pdf files in there. step1_load.py handles all three.
