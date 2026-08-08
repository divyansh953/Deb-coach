"""Structured extractor for `Word Power Made Easy` (Norman Lewis).

The book has a stable structure:
- 12 CHAPTERs
- 47 SESSIONs, each with a sequence of named subsections:
    TEASER PREVIEW            -> short intro / motivation
    TEASER QUESTIONS ...      -> warm-up questions
    USING THE WORDS           -> pronunciation + working definitions
    ORIGINS AND RELATED WORDS -> root / prefix / etymology body
    REVIEW OF ETYMOLOGY       -> root recap table
    SAY THESE WORDS ALOUD     -> pronunciation drill
    CAN YOU MATCH ...?        -> matching exercise
    DO YOU UNDERSTAND ...?    -> recall exercise

Generic recursive chunking discards that structure and mixes drill
material with prose. This extractor splits the book along those
markers, tags each chunk with `session`, `chapter`, and `section_type`,
and skips the noisy answer-key / blank-line drill chunks so the
english_db retrieves *teaching* prose first.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import fitz
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    fitz = None

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    try:
        from langchain.schema import Document  # type: ignore
    except ImportError:
        Document = None


SECTION_HEADERS = [
    "TEASER PREVIEW",
    "TEASER QUESTIONS FOR THE AMATEUR",
    "ETYMOLOGIST",
    "USING THE WORDS",
    "ORIGINS AND RELATED WORDS",
    "REVIEW OF ETYMOLOGY",
    "SAY THESE WORDS ALOUD",
    "CAN YOU PRONOUNCE THE WORDS",
    "CAN YOU WORK WITH THE WORDS",
    "CAN YOU TELL THE DIFFERENCE",
    "DO YOU UNDERSTAND THE WORDS",
    "CHAPTER REVIEW",
    "IDEAS",
]

SECTION_TYPE = {
    "TEASER PREVIEW": "teaser",
    "TEASER QUESTIONS FOR THE AMATEUR": "teaser",
    "ETYMOLOGIST": "teaser",
    "USING THE WORDS": "definitions",
    "ORIGINS AND RELATED WORDS": "etymology",
    "REVIEW OF ETYMOLOGY": "etymology",
    "SAY THESE WORDS ALOUD": "pronunciation",
    "CAN YOU PRONOUNCE THE WORDS": "pronunciation",
    "CAN YOU WORK WITH THE WORDS": "exercise",
    "CAN YOU TELL THE DIFFERENCE": "exercise",
    "DO YOU UNDERSTAND THE WORDS": "exercise",
    "CHAPTER REVIEW": "review",
    "IDEAS": "ideas",
}

SESSION_RE = re.compile(r"^SESSION\s+(\d+)\s*$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)\s*$", re.MULTILINE)
HEADER_RE = re.compile(
    r"^(?P<h>" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$",
    re.MULTILINE,
)

MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 120


def _extract_text(pdf_path: str | Path) -> str:
    if fitz is None:
        return ""
    pdf = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in pdf)


def _split_sessions(text: str) -> list[tuple[int | None, str]]:
    """Yield (session_number, body) pairs. Pre-session preface is None."""
    matches = list(SESSION_RE.finditer(text))
    if not matches:
        return [(None, text)]

    blocks = []
    if matches[0].start() > 0:
        blocks.append((None, text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((int(m.group(1)), text[m.end() : end]))
    return blocks


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    matches = list(HEADER_RE.finditer(body))
    if not matches:
        return [(None, body)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, body[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((m.group("h"), body[m.end() : end]))
    return sections


def _chapter_at(text: str, before_pos: int) -> int | None:
    last = None
    for m in CHAPTER_RE.finditer(text):
        if m.start() <= before_pos:
            last = int(m.group(1))
        else:
            break
    return last


def _trim_drill_noise(content: str) -> str:
    """Drop multi-blank-line drill rows that hurt retrieval quality."""
    lines = content.splitlines()
    cleaned = []
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(line)
            continue
        blank_run = 0
        # Skip pure answer-key rows like "1. T  2. F  3. T"
        if re.fullmatch(r"(\d+\.\s*[A-Z]\s*){3,}", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _split_long(body: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces = []
    remaining = body
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n\n", 0, max_chars)
        if split_at == -1:
            split_at = remaining.rfind(". ", 0, max_chars)
        if split_at == -1 or split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _make_doc(content: str, metadata: dict):
    if Document is None:
        return {"page_content": content, "metadata": metadata}
    return Document(page_content=content, metadata=metadata)


def extract_word_power(pdf_path: str | Path, doc_type: str = "english_vocab") -> list:
    text = _extract_text(pdf_path)
    if not text:
        return []

    documents = []
    for session_num, body in _split_sessions(text):
        # Best-effort chapter lookup using session start position in original text.
        session_pos = text.find(body[:200]) if body else 0
        chapter_num = _chapter_at(text, session_pos)

        for header, content in _split_sections(body):
            content = _trim_drill_noise(content)
            if len(content) < MIN_CHUNK_CHARS:
                continue

            section_type = SECTION_TYPE.get(header or "", "prose")
            base_meta = {
                "source_path": str(pdf_path),
                "doc_type": doc_type,
                "session": session_num,
                "chapter": chapter_num,
                "section": header,
                "section_type": section_type,
            }

            for piece in _split_long(content):
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                documents.append(_make_doc(piece, dict(base_meta)))

    return documents
