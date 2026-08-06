"""Chunking — deterministic text and code segmentation for embedding/retrieval.

Documents are chunked by characters with an overlap window that respects
paragraph boundaries where possible; code is chunked by lines with a line
overlap, so "Find every TODO" scans and retrieval both keep line context.
"""

from __future__ import annotations

import re

#: Paragraph split used to keep whole paragraphs together while chunking.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split long text into overlapping character chunks on paragraph seams."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if chunk_size <= overlap:
        overlap = max(0, chunk_size // 4)

    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current and current_len + len(para) + 2 > chunk_size and current_len > chunk_size // 2:
            chunks.append("\n\n".join(current))
            tail = _tail_chars("\n\n".join(current), overlap)
            current = [tail] if tail else []
            current_len = len(tail or "")
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) == 1 and len(chunks[0]) <= chunk_size:
        return chunks
    # Fallback: chunk brutally when paragraphs are longer than the window.
    merged = [
        c for chunk in chunks for c in _hard_split(chunk, chunk_size, overlap)
    ]
    return merged or [text]


def chunk_code(text: str, lines: int = 200, overlap: int = 20) -> list[tuple[str, int, int]]:
    """Split source code into line-bounded chunks (text, start_line, end_line)."""
    source_lines = text.splitlines()
    if not source_lines:
        return []
    if len(source_lines) <= lines:
        return [("\n".join(source_lines), 1, len(source_lines))]
    step = max(1, lines - overlap)
    chunks: list[tuple[str, int, int]] = []
    for start in range(0, len(source_lines), step):
        block = source_lines[start:start + lines]
        chunks.append(("\n".join(block), start + 1, start + len(block)))
    return chunks


def _tail_chars(text: str, n: int) -> str:
    if n <= 0 or not text:
        return ""
    return text[-n:].lstrip("\n")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        parts.append(text[start:end])
        start = max(start, 1, end - overlap)
    return parts