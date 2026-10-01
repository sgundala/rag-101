"""STEP 1 · LOAD  —  any file becomes plain text.    (notes: Block 1A)

    >>> FILE  ->  TEXT

This is the ONLY step that knows what a file format is. Everything after it
sees plain text and does not care whether that text came from a PDF, a
markdown file, or a web page. That is why the loader is the only block you
rewrite when a new source appears.

Handles .md, .txt and .pdf today. Drop a PDF into docs/ and it just works.

FAILS WHEN
    the encoding is wrong (we force utf-8 and replace bad bytes rather than
    crash), or a scanned PDF has no text layer at all — that needs OCR, which
    is a different tool. We detect that case and say so instead of silently
    indexing an empty document.
"""

from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


def load(folder: str) -> list[dict]:
    """Read every supported file in `folder` into {text, source}.

    Returns one dict per FILE, not per page — a PDF's pages are joined, and
    chunking (step 3) decides the real boundaries later.
    """
    docs = []

    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue

        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text(path)

        if not text.strip():
            print(f"  !! {path.name} has no extractable text — skipped.")
            print(f"     (a scanned PDF needs OCR; that is a different tool)")
            continue

        docs.append({"text": text, "source": path.name})

    return docs


def _read_text(path: Path) -> str:
    """Plain text and markdown. errors='replace' so one bad byte cannot crash
    the whole ingest."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Pull the text layer out of a PDF, page by page, joined with blank lines.

    A PDF has no paragraphs — just positioned glyphs — so extracted text is
    always messier than markdown. That mess is exactly what step 2 cleans.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)
