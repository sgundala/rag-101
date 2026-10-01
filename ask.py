"""ASK — the ONLINE half.  Runs on every question, so it must be fast.

    uv run ask.py "what is grounding?"
    uv run ask.py "what is grounding?" --show    <- use this in class

    QUESTION -> VECTOR -> THE STORE -> TOP-K -> PROMPT -> ANSWER
        (6)       (6)        (6)        (6)      (7)       (8)

Notice what this file does NOT do: it never loads a document, never chunks
anything, never re-embeds the corpus. It only READS the store. Offline wrote
it; online reads it. A system that re-embeds on every question has collapsed
that boundary and will be unusably slow.

Read the last three lines of ask(). That is RAG. Everything else in this
project is detail about doing those three things well.
"""

import sys
import time

from config import TOP_K
from step4_embed import embed
from step5_store import count, where
from step6_search import retrieve
from step7_augment import build_prompt
from step8_generate import MODEL, generate

SHOW = "--show" in sys.argv


def banner(n: int, name: str, arrow: str) -> None:
    if SHOW:
        print(f"\n{'─' * 70}")
        print(f"STEP {n} · {name}")
        print(f"  {arrow}")
        print(f"{'─' * 70}")


def ask(question: str) -> str:
    """The entire online path, in three calls."""
    chunks = retrieve(question)                      # step 6
    prompt = build_prompt(question, chunks)          # step 7
    return generate(prompt)                          # step 8


def ask_narrated(question: str) -> str:
    """The same three calls, with every stage printed. For teaching."""
    print("=" * 70)
    print("ONLINE — answering a question")
    print("=" * 70)
    print(f"  question    {question!r}")
    print(f"  store       {where()}  ({count()} chunks, already indexed)")
    print(f"  nothing is re-embedded here except the question itself")

    # ── STEP 6 · SEARCH ──────────────────────────────────────────────────
    banner(6, "SEARCH", "question  ->  vector  ->  nearest stored chunks")
    t = time.time()
    qvec = embed([question])[0]
    print(f"  the question became {len(qvec)} numbers — "
          f"by the SAME embed() the documents used")
    print(f"  that is why they are comparable at all\n")

    chunks = retrieve(question)
    print(f"  compared against {count()} stored vectors, cosine, top {TOP_K}:")
    for i, c in enumerate(chunks):
        print(f"    [{i + 1}]  score {c['score']:.3f}  {c['source']}")
        print(f"         {c['content'][:60]}...")
    print(f"\n  search took {(time.time() - t) * 1000:.0f} ms")

    # ── STEP 7 · AUGMENT ─────────────────────────────────────────────────
    banner(7, "AUGMENT", "chunks + question  ->  one grounded prompt")
    prompt = build_prompt(question, chunks)
    print(f"  prompt is {len(prompt):,} chars:")
    print(f"    the grounding instruction  (\"answer ONLY from the context\")")
    print(f"  + {len(chunks)} retrieved chunks, numbered [1]..[{len(chunks)}]")
    print(f"  + the question")
    print(f"\n  GROUNDING LIVES HERE, IN THE PROMPT — not in the model.")

    # ── STEP 8 · GENERATE ────────────────────────────────────────────────
    banner(8, "GENERATE", "prompt  ->  the answer")
    print(f"  model  {MODEL}")
    print(f"  this is the same API call as Week 1. Nothing about it is new.")
    print(f"  the model did not change — the context did.\n")

    t = time.time()
    answer = generate(prompt)
    print(f"  generation took {time.time() - t:.1f}s")

    print(f"\n{'=' * 70}")
    print("ANSWER")
    print("=" * 70)
    print(answer)

    cited = {c["source"] for c in chunks}
    print(f"\nretrieved from: {', '.join(sorted(cited))}")
    print("=" * 70)

    return answer


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        raise SystemExit(
            'Ask a question:\n'
            '  uv run ask.py "what is grounding?" --show'
        )

    question = " ".join(args)

    if SHOW:
        ask_narrated(question)
    else:
        print(ask(question))


if __name__ == "__main__":
    main()
