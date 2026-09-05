"""
Structure-aware chunking.

Rather than slicing text every N characters (which routinely cuts a
sentence — or an idea — in half), we chunk along the document's own
structure first (headings / page boundaries for PDFs, paragraph groups
for web pages), and only fall back to a token-count split *within* a
section if that section is too long on its own.

Target size: 300-500 tokens, ~50 token overlap (see config.py).
Each chunk keeps its hierarchical context (organization, document,
section, age_range, topics) so retrieval never returns an orphaned,
contextless sentence.
"""
from __future__ import annotations

import re
import uuid

from config import CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from src.preprocessing.arabic_cleaner import clean_display_text
from src.schema import Chunk, RawDocument

_HEADING_PATTERN = re.compile(r"^\s*#{1,4}\s+(.*)$", re.MULTILINE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _approx_token_count(text: str) -> int:
    """Cheap token estimate (Arabic + English word split). Good enough for
    chunk sizing without pulling in a full tokenizer dependency here."""
    return len(text.split())


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown-style headings if present; otherwise treat the
    whole document as a single unnamed section."""
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end].strip()))
    return sections


_SENTENCE_SPLIT = re.compile(r"(?<=[.!؟])\s+")


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    """Fallback: split a single paragraph with no internal breaks into
    sentence-level pieces, each close to CHUNK_TARGET_TOKENS."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    if len(sentences) <= 1:
        return [paragraph]  # nothing to split on, leave as-is

    pieces = []
    current, current_tokens = [], 0
    for sentence in sentences:
        sent_tokens = _approx_token_count(sentence)
        if current_tokens + sent_tokens > CHUNK_TARGET_TOKENS and current:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sent_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _chunk_section_text(text: str) -> list[str]:
    """Group paragraphs up to ~CHUNK_TARGET_TOKENS, with a small overlap
    carried into the next chunk so context isn't lost at the boundary."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    # Break up any paragraph that's already bigger than the target on its
    # own (e.g. text with no internal line breaks) into sentence-level
    # pieces before grouping — otherwise a single oversized paragraph
    # would end up as one oversized chunk.
    pieces: list[str] = []
    for paragraph in paragraphs:
        if _approx_token_count(paragraph) > CHUNK_TARGET_TOKENS:
            pieces.extend(_split_oversized_paragraph(paragraph))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _approx_token_count(piece)

        if current_tokens + piece_tokens > CHUNK_TARGET_TOKENS and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)

            # carry the tail of the previous chunk forward as overlap
            overlap_words = chunk_text.split()[-CHUNK_OVERLAP_TOKENS:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        remaining_text = "\n\n".join(current)
        if chunks and _approx_token_count(remaining_text) < CHUNK_MIN_TOKENS:
            chunks[-1] = chunks[-1] + "\n\n" + remaining_text
        else:
            chunks.append(remaining_text)

    return chunks

def _page_for_offset(page_map: dict[int, str] | None, snippet: str) -> int | None:
    """Best-effort lookup of which PDF page a chunk snippet came from,
    used purely for citation display (see prompt.py)."""
    if not page_map:
        return None
    probe = snippet[:40].strip()
    for page_number, page_text in page_map.items():
        if probe and probe in page_text:
            return page_number
    return None


def chunk_document(doc: RawDocument) -> list[Chunk]:
    """Turn one RawDocument into a list of retrievable Chunks."""
    sections = _split_into_sections(doc.text)
    chunks: list[Chunk] = []

    for heading, section_text in sections:
        for piece in _chunk_section_text(section_text):
            cleaned = clean_display_text(piece)
            if not cleaned:
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc.doc_id,
                    text=cleaned,
                    organization=doc.organization,
                    title=doc.title,
                    section=heading or None,
                    page=_page_for_offset(doc.page_map, cleaned),
                    age_range=doc.age_range,
                    topics=doc.topics,
                    priority_stars=doc.priority_stars,
                    license_status=doc.license_status,
                    url=doc.url,
                )
            )
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
