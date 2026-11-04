"""
GrigoTel AI Assistant - Step 2.1: Convert PDFs to markdown

Converts every PDF in GrigoTel_Docs/ to a .md file in GrigoTel_Docs/markdown/
using markitdown. Also defines classify_doc(), which tags each document with
its phone model or "general" (used by build_index.py).

Location: src/grigotel_ai_assistant/convert_docs.py

Usage (from the project root):
    uv run src/grigotel_ai_assistant/convert_docs.py
"""

from pathlib import Path

from markitdown import MarkItDown

# This file lives in src/grigotel_ai_assistant/, so the project root is 2 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "GrigoTel_Docs"
MARKDOWN_DIR = DOCS_DIR / "markdown"

GENERAL = "general"

# Ordered rules: the FIRST rule whose keyword appears in the file name wins.
# General docs come first, because "Alexy_G26" also appears in the manual's name.
# Variants (Fold/Lite/Pro) come before plain "G26", because "G26" is in all of them.
CLASSIFICATION_RULES: list[tuple[str, str]] = [
    ("user_manual", GENERAL),
    ("corporate_profile", GENERAL),
    ("privacy", GENERAL),
    ("g26_fold", "Alexy G26 Fold"),
    ("g26_lite", "Alexy G26 Lite"),
    ("g26_pro", "Alexy G26 Pro"),
    ("g26", "Alexy G26"),
]


def classify_doc(path: Path) -> str:
    """Return the model name a document belongs to, or 'general'."""
    name = path.stem.lower()
    for keyword, tag in CLASSIFICATION_RULES:
        if keyword in name:
            return tag
    raise ValueError(f"Cannot classify '{path.name}'. Add a rule to CLASSIFICATION_RULES.")


def main() -> None:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))

    # Classify everything before converting, so a naming problem stops the run early
    for pdf in pdfs:
        classify_doc(pdf)

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for pdf in pdfs:
        md_text = converter.convert(str(pdf)).text_content
        (MARKDOWN_DIR / f"{pdf.stem}.md").write_text(md_text, encoding="utf-8")

    print(f"Converted {len(pdfs)} PDFs -> {MARKDOWN_DIR}")


if __name__ == "__main__":
    main()
