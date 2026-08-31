"""
Chunk cleaned book-chapter text (output of convert_pdf_to_text.py) into
overlapping, structure-aware chunks ready for embedding.

Two-layer approach:
  1. Split by document structure (optional heading regex) into sections.
  2. Within each section, build token-based chunks along paragraph
     boundaries, with paragraph-based overlap between consecutive chunks.

Usage:
    python chunking.py ../data/chapter.txt
    python chunking.py ../data/chapter.txt --chunk-size 400 --overlap 60
    python chunking.py ../data/chapter.txt --heading-pattern "^Chapter \\d+"
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None


def count_tokens(text: str) -> int:
    """Real token count via tiktoken when available; otherwise a rough
    word-count fallback so the script still runs without the dependency.
    """
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text.split()))


@dataclass
class Chunk:
    text: str
    section: str
    chunk_id: int
    token_count: int


def split_by_structure(text: str, heading_pattern: Optional[str] = None) -> List[Tuple[str, str]]:
    """Split text into (section_title, section_text) pairs using a heading
    regex (matched at the start of a line). If no pattern is given or no
    matches are found, the whole text is treated as a single section.
    """
    if not heading_pattern:
        return [("document", text)]

    pattern = re.compile(heading_pattern, flags=re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("document", text)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))

    # keep any text before the first heading as its own "intro" section
    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.insert(0, ("intro", intro))

    return sections


def keep_last_paragraphs_within_tokens(paragraphs: List[str], overlap_tokens: int) -> List[str]:
    """Keep as many trailing paragraphs as fit within overlap_tokens, so the
    next chunk starts with real paragraph-level context, not a raw
    character cut mid-word.
    """
    kept: List[str] = []
    total = 0
    for para in reversed(paragraphs):
        para_tokens = count_tokens(para)
        if kept and total + para_tokens > overlap_tokens:
            break
        kept.insert(0, para)
        total += para_tokens
    return kept


def chunk_section(section_text: str, section_title: str, chunk_size: int, overlap: int, start_id: int) -> List[Chunk]:
    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_tokens = 0
    chunk_id = start_id

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if buffer and buffer_tokens + para_tokens > chunk_size:
            chunks.append(Chunk(
                text="\n\n".join(buffer),
                section=section_title,
                chunk_id=chunk_id,
                token_count=buffer_tokens,
            ))
            chunk_id += 1
            buffer = keep_last_paragraphs_within_tokens(buffer, overlap)
            buffer_tokens = count_tokens("\n\n".join(buffer)) if buffer else 0

        buffer.append(para)
        buffer_tokens += para_tokens

    if buffer:
        chunks.append(Chunk(
            text="\n\n".join(buffer),
            section=section_title,
            chunk_id=chunk_id,
            token_count=buffer_tokens,
        ))

    return chunks


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 60, heading_pattern: Optional[str] = None) -> List[Chunk]:
    sections = split_by_structure(text, heading_pattern)
    all_chunks: List[Chunk] = []
    next_id = 0

    for title, section_text in sections:
        section_chunks = chunk_section(section_text, title, chunk_size, overlap, next_id)
        all_chunks.extend(section_chunks)
        next_id += len(section_chunks)

    return all_chunks


def save_chunks_to_jsonl(chunks: List[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chunk cleaned text into overlapping, structure-aware chunks.")
    parser.add_argument("input", type=Path, help="Path to the cleaned .txt file (output of convert_pdf_to_text.py).")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .jsonl path. Defaults to <input>_chunks.jsonl next to the input file."
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Target chunk size in tokens (default: 400).")
    parser.add_argument("--overlap", type=int, default=60, help="Overlap between chunks in tokens (default: 60).")
    parser.add_argument(
        "--heading-pattern", type=str, default=None,
        help=r"Regex for section headings, e.g. '^Chapter \d+'. If omitted, the whole document is one section."
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    chunks = chunk_document(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        heading_pattern=args.heading_pattern,
    )

    output_path = args.output or args.input.with_name(args.input.stem + "_chunks.jsonl")
    save_chunks_to_jsonl(chunks, output_path)

    section_count = len(set(c.section for c in chunks)) if chunks else 0
    print(f"Created {len(chunks)} chunks across {section_count} section(s).")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
