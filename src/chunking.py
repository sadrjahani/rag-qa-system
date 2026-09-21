"""
Chunk cleaned book-chapter text (output of convert_pdf_to_text.py) into
overlapping, structure-aware chunks ready for embedding.

Two-layer approach:
  1. Split by document structure into sections, using ONE of:
       - an explicit --heading-pattern regex, or
       - headings auto-detected by convert_pdf_to_text.py's .meta.json
         sidecar (auto-discovered next to the input file, or given via
         --metadata), or
       - nothing, in which case the whole document is one section.
  2. Within each section, build token-based chunks along paragraph
     boundaries, with paragraph-based overlap between consecutive chunks.

Usage:
    python chunking.py ../data/chapter.txt
    python chunking.py ../data/chapter.txt --chunk-size 400 --overlap 60
    python chunking.py ../data/chapter.txt --heading-pattern "^Chapter \\d+"
    python chunking.py ../data/chapter.txt --metadata ../data/chapter.meta.json
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
    page: Optional[int] = None


def split_by_structure(text: str, heading_pattern: Optional[str] = None) -> List[Tuple[str, str, Optional[int]]]:
    """Split text into (section_title, section_text, page) triples using a
    heading regex (matched at the start of a line). page is always None
    here since a regex match carries no page information.
    If no pattern is given or no matches are found, the whole text is
    treated as a single section.
    """
    if not heading_pattern:
        return [("document", text, None)]

    pattern = re.compile(heading_pattern, flags=re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("document", text, None)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip(), None))

    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.insert(0, ("intro", intro, None))

    return sections


def split_by_metadata_headings(text: str, headings: List[dict]) -> List[Tuple[str, str, Optional[int]]]:
    """Split text into (section_title, section_text, page) triples using
    heading candidates detected by convert_pdf_to_text.py (saved in the
    .meta.json sidecar). Each heading's exact text is located inside the
    extracted text (in order) to find the split points.

    Headings that can't be found verbatim (e.g. filtered out of the body
    text for some reason) are skipped rather than breaking the split.
    """
    if not headings:
        return [("document", text, None)]

    positions = []
    search_start = 0
    for heading in headings:
        title = heading["text"]
        idx = text.find(title, search_start)
        if idx == -1:
            continue  # not found verbatim past this point, skip it
        positions.append((idx, title, heading.get("page")))
        search_start = idx + len(title)

    if not positions:
        return [("document", text, None)]

    sections = []
    for i, (idx, title, page) in enumerate(positions):
        start = idx + len(title)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections.append((title, text[start:end].strip(), page))

    if positions[0][0] > 0:
        intro = text[: positions[0][0]].strip()
        if intro:
            sections.insert(0, ("intro", intro, None))

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


def chunk_section(
    section_text: str, section_title: str, chunk_size: int, overlap: int, start_id: int, page: Optional[int] = None
) -> List[Chunk]:
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
                page=page,
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
            page=page,
        ))

    return chunks


def chunk_document(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
    heading_pattern: Optional[str] = None,
    headings: Optional[List[dict]] = None,
) -> List[Chunk]:
    """
    Structure-aware paragraph-based chunking with token limits and overlap.
    Structure source priority: explicit heading_pattern regex first (if
    given), then metadata headings (if given), then no split (whole
    document as one section).
    """
    if heading_pattern:
        sections = split_by_structure(text, heading_pattern)
    elif headings:
        sections = split_by_metadata_headings(text, headings)
    else:
        sections = [("document", text, None)]

    all_chunks: List[Chunk] = []
    next_id = 0

    for title, section_text, page in sections:
        section_chunks = chunk_section(section_text, title, chunk_size, overlap, next_id, page=page)
        all_chunks.extend(section_chunks)
        next_id += len(section_chunks)

    return all_chunks


def load_headings_from_metadata(metadata_path: Path) -> List[dict]:
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata.get("headings", [])


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
        help=r"Regex for section headings, e.g. '^Chapter \d+'. Overrides --metadata if both are given."
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Path to a .meta.json sidecar (from convert_pdf_to_text.py) to use its detected headings. "
             "If omitted, auto-looked-up next to the input file (<input_stem>.meta.json)."
    )
    parser.add_argument(
        "--no-metadata-lookup", action="store_true",
        help="Disable auto-discovery of a .meta.json sidecar next to the input file."
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")

    headings: Optional[List[dict]] = None
    if not args.heading_pattern:
        metadata_path = args.metadata
        if metadata_path is None and not args.no_metadata_lookup:
            candidate = args.input.with_suffix(".meta.json")
            if candidate.exists():
                metadata_path = candidate
                print(f"Auto-discovered metadata: {metadata_path}")

        if metadata_path is not None:
            headings = load_headings_from_metadata(metadata_path)
            print(f"Loaded {len(headings)} heading candidate(s) from metadata.")

    chunks = chunk_document(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        heading_pattern=args.heading_pattern,
        headings=headings,
    )

    output_path = args.output or args.input.with_name(args.input.stem + "_chunks.jsonl")
    save_chunks_to_jsonl(chunks, output_path)

    section_count = len(set(c.section for c in chunks)) if chunks else 0
    print(f"Created {len(chunks)} chunks across {section_count} section(s).")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
