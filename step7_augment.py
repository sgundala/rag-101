"""STEP 7 · AUGMENT  —  build the grounded prompt.        (notes: Block 4A)

    TOP-K CHUNKS + QUESTION  ->  >>> ONE PROMPT

This is pure Week 2. You are writing a system prompt with a guardrail in it,
and the guardrail is aimed at hallucination.

    Week 2, Day 1   "never promise a payout"
    Week 2, Day 2   "never auto-block a transaction"
    Week 2, Day 3   "you screen, you do not enroll"
    Week 3          "answer ONLY from the context; if it is not there,
                     say you do not know"

GROUNDING IS NOT IN THE MODEL. IT IS IN THE PROMPT.
That is worth saying out loud, because students assume RAG changed the model.
It did not. The model is identical to the one you called in Week 1. What
changed is what you put in front of it. RAG is better INPUTS, not a new model.

And this is the structural cure for hallucination: if the true fact is sitting
in the prompt and the model is told to answer only from it, there is nothing
left to invent.

THE NUMBERED CITATIONS
Each chunk gets a [1], [2], [3] label and carries its source. Asking the model
to cite by number gives you attribution for free — the thing fine-tuning can
never do.
"""

GROUNDING = (
    "Answer the question using ONLY the context below.\n"
    "If the answer is not in the context, say you do not know — do not guess "
    "and do not use knowledge from outside the context.\n"
    "Cite the context you used by its number, like [1] or [2].\n"
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Format retrieved chunks and the question into one grounded prompt."""
    if not chunks:
        # No context retrieved. Say so rather than sending a bare question —
        # a bare question is an invitation to hallucinate.
        return (f"{GROUNDING}\nCONTEXT:\n(nothing was retrieved)\n\n"
                f"QUESTION: {question}")

    context = "\n\n".join(
        f"[{i + 1}] {c['content']}\n    (source: {c['source']})"
        for i, c in enumerate(chunks)
    )

    return f"{GROUNDING}\nCONTEXT:\n{context}\n\nQUESTION: {question}"
