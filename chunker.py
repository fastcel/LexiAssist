"""
chunker.py — Sentence-aware chunking with metadata preservation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    citation: str = ""
    url: str = ""
    jurisdiction: str = ""
    char_start: int = 0
    char_end: int = 0
    embedding: Optional[list] = field(default=None, repr=False)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no nltk dependency)."""
    abbrevs = r"(?:s\.|ss\.|No\.|Art\.|cl\.|para\.|Vol\.|vs\.|v\.|cf\.|ibid\.|id\.|etc\.|e\.g\.|i\.e\.|viz\.|Pt\.|Div\.|Sch\.)"

    text = re.sub(abbrevs, lambda m: m.group().replace(".", "<DOT>"), text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\(\"])", text)

    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]


def chunk_document(
    text: str,
    doc_id: str,
    citation: str = "",
    url: str = "",
    jurisdiction: str = "",
    chunk_size: int = 250,
    overlap_sentences: int = 2,
) -> list[Chunk]:

    if len(text.split()) < 30:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_words = 0
    chunk_idx = 0
    char_cursor = 0

    for sent in sentences:
        current_sentences.append(sent)
        current_words += len(sent.split())

        if current_words >= chunk_size:
            chunk_text = " ".join(current_sentences)

            start = text.find(current_sentences[0], char_cursor)
            end = start + len(chunk_text)

            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_idx,
                citation=citation,
                url=url,
                jurisdiction=jurisdiction,
                char_start=max(0, start),
                char_end=max(0, end),
            ))

            chunk_idx += 1

            current_sentences = current_sentences[-overlap_sentences:]
            current_words = sum(len(s.split()) for s in current_sentences)
            char_cursor = start

    # flush remainder
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        start = text.find(current_sentences[0], char_cursor)
        end = max(0, start) + len(chunk_text)  # fixed: guard start=-1 before adding length

        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc_id,
            chunk_index=chunk_idx,
            citation=citation,
            url=url,
            jurisdiction=jurisdiction,
            char_start=max(0, start),
            char_end=end,
        ))

    return chunks