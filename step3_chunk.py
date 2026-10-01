"""STEP 3 · CHUNK  —  cut the text into pieces, measured in TOKENS.

                                                          (notes: Block 1B)
    CLEAN TEXT  ->  >>> TOKENS  ->  CHUNKS

THIS IS THE STEP WHERE THE TOKENIZER FROM WEEK 1 COMES BACK, DOING REAL WORK.

We do not count words. We do not count characters. We count TOKENS, using the
embedding model's own tokenizer — the exact same WordPiece/BPE idea from Week
1, Day 2. Three reasons, and they all matter:

    1. The model has a MAX SEQUENCE LENGTH (256 tokens for MiniLM). Text past
       that limit is SILENTLY TRUNCATED — no error, no warning, just a vector
       that ignored the end of your chunk. This is the deepest reason we chunk.

    2. Your cost and the model's context budget are both counted in tokens.

    3. One vector has to mean "about one thing". A whole document embedded as
       one vector blurs every topic in it into an average of nothing.

WHY OVERLAP
Cut at a hard boundary and an idea gets sliced in half — the sentence that
answers the question ends up split across two chunks, and neither one scores
well enough to be retrieved. Overlap means each chunk carries the tail of the
one before it, so an idea near a boundary survives intact in at least one.
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL
from step4_embed import get_tokenizer


def chunk(text: str, size: int = CHUNK_SIZE,
          overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into ~`size`-TOKEN pieces that overlap by `overlap`.

    Args:
        text: cleaned text from step 2.
        size: tokens per chunk. Keep it at or under the model's max sequence
            length, or the tail of every chunk is silently thrown away.
        overlap: tokens each chunk repeats from the previous one.

    Returns:
        A list of text strings. Note they come back as TEXT, not tokens — we
        tokenize only to find the right places to cut, then decode back, so
        step 4 can embed normal strings.
    """
    tok = get_tokenizer()

    # text -> token ids. This is literally Week 1's tokenizer.
    ids = tok.encode(text, add_special_tokens=False)

    step = size - overlap          # how far the window slides each time
    chunks = []

    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if not window:
            break
        chunks.append(tok.decode(window))

        if start + size >= len(ids):
            break                  # the window already covered the tail

    return chunks


def count_tokens(text: str) -> int:
    """How many tokens is this text? Used by --show to make the unit visible."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))
