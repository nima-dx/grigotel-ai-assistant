"""
GrigoTel AI Assistant - Step 2.2: Chunk the markdown docs and build the index

Reads the .md files produced by convert_docs.py, chunks them with
gitsource.chunk_documents (sliding window) and saves the chunks to
GrigoTel_Docs/chunks.json. The app loads that file with load_chunks()
and builds the minsearch index with build_index().

Each document carries metadata (filename, doc_title, doc_type, model);
chunk_documents keeps these fields on every chunk and adds `start`
(character offset) and `content` (the chunk text).

Location: src/grigotel_ai_assistant/build_index.py

Usage (from the project root):
    uv run src/grigotel_ai_assistant/build_index.py
"""

import json
import re
from pathlib import Path

from gitsource import chunk_documents
from minsearch import Index

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.convert_docs import DOCS_DIR, MARKDOWN_DIR, classify_doc
except ImportError:  # works when run as a script from this folder
    from convert_docs import DOCS_DIR, MARKDOWN_DIR, classify_doc

CHUNKS_PATH = DOCS_DIR / "chunks.json"

CHUNK_SIZE = 500  # characters per chunk
CHUNK_STEP = 100  # window moves 1000 chars -> 1000 chars overlap

# Lines that are conversion noise, not content
NOISE_LINES = {".", "- End of Document -"}


def clean_text(text: str) -> str:
    """Strip each line, drop noise lines and collapse runs of blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line not in NOISE_LINES]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def doc_type(md_path: Path) -> str:
    name = md_path.stem.lower()
    if "user_manual" in name:
        return "manual"
    if "corporate" in name:
        return "corporate"
    if "privacy" in name:
        return "privacy"
    return "spec"


def load_documents() -> list[dict]:
    """Read every markdown file into a document dict with metadata."""
    documents = []
    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        content = clean_text(md_path.read_text(encoding="utf-8"))
        documents.append({
            "filename": f"{md_path.stem}.pdf",
            "doc_title": content.split("\n", 1)[0],  # first line is the document title
            "doc_type": doc_type(md_path),
            "model": classify_doc(md_path),  # a model name or "general"
            "content": content,
        })
    return documents


def build_chunks() -> list[dict]:
    return chunk_documents(load_documents(), size=CHUNK_SIZE, step=CHUNK_STEP)


def load_chunks() -> list[dict]:
    """Load the saved chunks (used by the app)."""
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def build_index(chunks: list[dict]) -> Index:
    index = Index(
        text_fields=["doc_title", "content"],
        keyword_fields=["model", "doc_type", "filename"],
    )
    index.fit(chunks)
    return index


def main() -> None:
    chunks = build_chunks()
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
