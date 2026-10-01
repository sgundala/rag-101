"""STEP 6 · SEARCH  —  the question becomes a vector, and finds its neighbours.

                                                          (notes: Block 3A)
    QUESTION  ->  >>> VECTOR  ->  THE STORE  ->  TOP-K CHUNKS

THIS IS WHERE REAL RAG SYSTEMS LIVE OR DIE. It looks like three lines.

Notice the first line of retrieve(): it calls the SAME embed() the documents
went through in step 4. That is the whole reason there is only one embed
function — the question and the documents must land in the same vector space,
or you are measuring distance between two unrelated coordinate systems.

SEMANTIC, NOT LEXICAL
Keyword search matches STRINGS: "car" never finds "automobile". This matches
MEANING: embed the question, embed the documents, return the nearest vectors.
"How do I fix my automobile" finds a car-repair doc with no shared words.

FAILS WHEN
    - K is wrong: too low and the answer is not in the prompt; too high and
      noise crowds out the signal
    - pure vector search misses an exact token — a name, an ID, an error code
      (the fix is HYBRID search: BM25 keyword + vector, fused)
    - a "close but wrong" chunk outranks the right one (the fix is RERANKING:
      retrieve a wide net of 20, then re-score with a slower cross-encoder
      and keep 5)

THE SIGNATURE AND DOCSTRING ARE WRITTEN THE WAY THEY ARE ON PURPOSE.
On Day 3 you wrap this function as a TOOL an agent can call — the Week 2 Day 2
idea. When an agent has several tools, the docstring is what drives tool
selection. A good docstring written today is a Day 3 investment.
"""

from config import TOP_K
from step4_embed import embed
from step5_store import search as store_search


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks most semantically similar to `question`.

    Each result is a dict with:
        content  the chunk text
        source   the file it came from — this is what makes citation free
        score    cosine similarity, 0 to 1, higher is closer

    Use this to ground an answer in retrieved documents rather than the
    model's memory.
    """
    qvec = embed([question])[0]          # the SAME embed() the documents used
    return store_search(qvec, k)
