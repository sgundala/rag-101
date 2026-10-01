"""STEP 2 · CLEAN  —  tidy the text before anyone cuts it up.  (notes: Block 1C)

    TEXT  ->  >>> CLEAN TEXT

WHY THIS RUNS BEFORE CHUNKING, NOT AFTER
The memorised chain says "load, chunk, clean" — but structural cleaning has
to happen BEFORE chunking, or a chunk boundary lands in the middle of an HTML
tag and you embed half a tag. Clean first, then cut. (The notes say this too,
in Block 1C; the mnemonic just rhymes better the other way.)

WHY CLEAN AT ALL
Embedding models are sensitive to formatting noise. A PDF gives you broken
line-wraps, page numbers and repeated headers; HTML gives you tags. None of
that carries meaning, and all of it eats your token budget.

THE ONE THING NOT TO DO
    Do NOT lowercase.
Modern embedders are cased and trained on normal text; lowercasing usually
HURTS retrieval. It is a habit left over from bag-of-words search. There is a
commented-out .lower() below purely so you can see the decision was made on
purpose rather than forgotten.
"""

import re


def clean(text: str) -> str:
    """Normalise whitespace and strip obvious noise. Keeps case and meaning."""

    # 1. strip HTML tags, if any survived the loader
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. de-hyphenate words broken across a PDF line-wrap:  "retriev-\nal"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 3. collapse every run of whitespace into one space
    text = re.sub(r"\s+", " ", text)

    # 4. drop bare page numbers left behind by PDF extraction
    text = re.sub(r"\s+\d{1,3}\s+(?=[A-Z])", " ", text)

    # text = text.lower()   # <- deliberately NOT done. Modern embedders are cased.

    return text.strip()
