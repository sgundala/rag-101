"""STEP 4 · EMBED  —  text becomes a list of numbers.     (notes: Block 2A)

    CHUNKS  ->  >>> VECTORS  (384 numbers each)

THERE IS EXACTLY ONE embed() IN THIS WHOLE SYSTEM, AND THAT IS THE POINT.

The documents call it during ingest. The question calls it during search. If
there were two, they could drift apart — a different model, a different
dimension — and the comparison becomes silent garbage. One function means the
query and the documents can never be embedded differently.

WHAT HAPPENS INSIDE THE MODEL
    tokenize -> transformer encoder (attention) -> one vector per token
             -> mean-pool to one vector -> normalize to length 1

That is Week 1's transformer, wired for READING instead of generating. Same
attention, same layers — it just never writes a word. A generation model is
decoder-only; an embedding model is an encoder.

WHY DISTANCE MEANS MEANING
These models are trained CONTRASTIVELY: pull paraphrases together, push
unrelated text apart, over a billion pairs. That training is the only reason
cosine distance tracks semantic dissimilarity. Geometry equals meaning because
somebody optimised for it — it did not come for free.

WHY normalize_embeddings=True
Every vector is scaled to length 1, so vector LENGTH — which mostly reflects
text length, not meaning — stops affecting the score. Cosine then compares
pure direction, and cosine, dot product and L2 all rank identically.
"""

import numpy as np

from config import EMBED_DIM, EMBED_MODEL

_model = None      # loaded once, lazily — the import should stay cheap


def _get_model():
    """Load the embedding model, with a readable error if it cannot download.

    First run fetches ~90MB from the HuggingFace Hub and caches it in
    ~/.cache/huggingface. Later runs are offline and instant.
    """
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    try:
        _model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        raise SystemExit(
            f"Could not load the embedding model '{EMBED_MODEL}'.\n"
            f"  {type(e).__name__}: {str(e)[:160]}\n\n"
            f"  First run downloads ~90MB from huggingface.co and caches it.\n"
            f"  If you are on a restricted network that blocks huggingface.co,\n"
            f"  download it once somewhere unrestricted with:\n"
            f"      python -c \"from sentence_transformers import SentenceTransformer;"
            f" SentenceTransformer('{EMBED_MODEL}')\"\n"
            f"  then copy ~/.cache/huggingface across."
        )

    # Fail loudly, now, if the model does not match config.
    actual = _model.get_sentence_embedding_dimension()
    if actual != EMBED_DIM:
        raise SystemExit(
            f"DIMENSION MISMATCH — the silent failure, caught early.\n"
            f"  {EMBED_MODEL} outputs {actual} numbers.\n"
            f"  config.EMBED_DIM says {EMBED_DIM}.\n"
            f"  Fix EMBED_DIM and re-run ingest.py — a model switch is a"
            f" full re-index."
        )

    return _model


def get_tokenizer():
    """The model's OWN tokenizer, so step 3 cuts in the model's units."""
    return _get_model().tokenizer


def embed(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into an array of unit-length vectors.

    Args:
        texts: any list of strings — chunks during ingest, one question
            during search. Both paths come through here on purpose.

    Returns:
        A float32 array of shape (len(texts), EMBED_DIM). Every row has
        length 1.0, so a dot product IS the cosine similarity.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # cosine-ready, length 1
        batch_size=64,               # batching is most of the speed
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def model_facts() -> dict:
    """What the model actually is — printed by --show so nothing is a mystery."""
    m = _get_model()
    return {
        "name": EMBED_MODEL,
        "tokenizer": m.tokenizer.__class__.__name__,
        "max_seq_length": m.get_max_seq_length(),
        "dimension": m.get_sentence_embedding_dimension(),
    }
