"""Document Parser — extracts text from every file kind the workspace accepts.

PDF (pypdf) is parsed page-by-page so pages survive into retrieval metadata;
DOCX (python-docx) reads paragraphs and tables; plain text formats and source
code are decoded as-is. Parser libraries are imported lazily so the workspace
still loads when an optional parser is missing (a parser error marks the file
``failed`` instead of breaking anything).
"""

from __future__ import annotations

import logging

from synapse.workspace.contracts import ParsedDocument

log = logging.getLogger("synapse.workspace.parsers")

#: Binary parsers keyed by extension. Plain text and code decode below.
_BINARY = {"pdf", "docx", "rtf"}


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(__import__("io").BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    metadata = {}
    info = reader.metadata
    if info is not None:
        metadata = {"title": info.title, "author": info.author, "pages": len(pages)}
    return ParsedDocument(text="\n".join(pages), pages=pages, metadata=metadata)


def _parse_docx(data: bytes) -> ParsedDocument:
    import io

    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return ParsedDocument(text="\n".join(parts))


def _parse_text(data: bytes) -> ParsedDocument:
    return ParsedDocument(text=_decode(data))


def parse(data: bytes, extension: str) -> ParsedDocument:
    """Extract text from raw bytes for any supported extension."""
    ext = (extension or "").lower().lstrip(".")
    if ext == "pdf":
        return _parse_pdf(data)
    if ext == "docx":
        return _parse_docx(data)
    return _parse_text(data)