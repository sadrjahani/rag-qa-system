"""
Chunk cleaned book-chapter text (output of convert_pdf_to_text.py) into
overlapping, structure-aware chunks ready for embedding.

Two-layer approach:
  1. Split by document structure using metadata headings or regex.
  2. Within each section, build token-based chunks along paragraph boundaries.

Usage:
    python chunking.py ../data/chapter.txt
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


def split_by_structure(
    text: str,
    heading_pattern: Optional[str] = None
) -> List[Tuple[str, str, Optional[int]]]:

    if not heading_pattern:
        return [("document", text, None)]

    pattern = re.compile(
        heading_pattern,
        flags=re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    if not matches:
        return [("document", text, None)]

    sections = []

    for i, match in enumerate(matches):
        title = match.group().strip()

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        sections.append(
            (
                title,
                text[start:end].strip(),
                None
            )
        )

    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()

        if intro:
            sections.insert(
                0,
                ("intro", intro, None)
            )

    return sections


def split_by_metadata_headings(
    text: str,
    headings: List[dict]
) -> List[Tuple[str, str, Optional[int]]]:

    if not headings:
        return [("document", text, None)]

    positions = []

    search_start = 0

    for heading in headings:

        title = heading["text"]

        idx = text.find(
            title,
            search_start
        )

        if idx == -1:
            continue

        positions.append(
            (
                idx,
                title,
                heading.get("page")
            )
        )

        search_start = idx + len(title)

    if not positions:
        return [("document", text, None)]

    sections = []

    for i, (idx, title, page) in enumerate(positions):

        start = idx + len(title)

        end = (
            positions[i + 1][0]
            if i + 1 < len(positions)
            else len(text)
        )

        sections.append(
            (
                title,
                text[start:end].strip(),
                page
            )
        )

    return sections


def keep_last_paragraphs_within_tokens(
    paragraphs: List[str],
    overlap_tokens: int
) -> List[str]:

    if overlap_tokens <= 0:
        return []

    kept = []
    total = 0

    for para in reversed(paragraphs):

        tokens = count_tokens(para)

        if kept and total + tokens > overlap_tokens:
            break

        kept.insert(0, para)
        total += tokens

        if total >= overlap_tokens:
            break

    return kept


def split_large_paragraph(
    paragraph: str,
    max_tokens: int
) -> List[str]:

    if count_tokens(paragraph) <= max_tokens:
        return [paragraph]

    if _ENCODING is None:

        words = paragraph.split()

        chunks = []
        current = []

        for word in words:

            candidate = current + [word]

            if (
                current
                and count_tokens(" ".join(candidate)) > max_tokens
            ):
                chunks.append(
                    " ".join(current)
                )

                current = [word]

            else:
                current = candidate

        if current:
            chunks.append(
                " ".join(current)
            )

        return chunks


    tokens = _ENCODING.encode(paragraph)

    result = []

    for start in range(
        0,
        len(tokens),
        max_tokens
    ):

        piece = tokens[start:start + max_tokens]

        result.append(
            _ENCODING.decode(piece)
        )

    return result



def chunk_section(
    section_text: str,
    section_title: str,
    chunk_size: int,
    overlap: int,
    start_id: int,
    page: Optional[int] = None
) -> List[Chunk]:


    paragraphs = [
        p.strip()
        for p in section_text.split("\n\n")
        if p.strip()
    ]


    normalized = []

    for paragraph in paragraphs:

        normalized.extend(
            split_large_paragraph(
                paragraph,
                chunk_size
            )
        )


    chunks = []

    buffer = []

    buffer_tokens = 0

    chunk_id = start_id


    for paragraph in normalized:

        paragraph_tokens = count_tokens(paragraph)


        if (
            buffer
            and buffer_tokens + paragraph_tokens > chunk_size
        ):

            chunks.append(
                Chunk(
                    text="\n\n".join(buffer),
                    section=section_title,
                    chunk_id=chunk_id,
                    token_count=buffer_tokens,
                    page=page,
                )
            )

            chunk_id += 1


            buffer = keep_last_paragraphs_within_tokens(
                buffer,
                overlap
            )


            buffer_tokens = (
                count_tokens(
                    "\n\n".join(buffer)
                )
                if buffer
                else 0
            )


        buffer.append(paragraph)

        buffer_tokens += paragraph_tokens



    if buffer:

        chunks.append(
            Chunk(
                text="\n\n".join(buffer),
                section=section_title,
                chunk_id=chunk_id,
                token_count=buffer_tokens,
                page=page,
            )
        )


    return chunks



def chunk_document(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
    heading_pattern: Optional[str] = None,
    headings: Optional[List[dict]] = None,
) -> List[Chunk]:


    if heading_pattern:

        sections = split_by_structure(
            text,
            heading_pattern
        )

    elif headings:

        sections = split_by_metadata_headings(
            text,
            headings
        )

    else:

        sections = [
            ("document", text, None)
        ]


    all_chunks = []

    next_id = 0


    for title, section_text, page in sections:

        section_chunks = chunk_section(
            section_text,
            title,
            chunk_size,
            overlap,
            next_id,
            page
        )

        all_chunks.extend(section_chunks)

        next_id += len(section_chunks)


    return all_chunks



def load_headings_from_metadata(
    metadata_path: Path
) -> List[dict]:

    with open(
        metadata_path,
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return metadata.get(
        "headings",
        []
    )



def save_chunks_to_jsonl(
    chunks: List[Chunk],
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    asdict(chunk),
                    ensure_ascii=False
                )
                + "\n"
            )



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=60
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=None
    )

    args = parser.parse_args()


    text = args.input.read_text(
        encoding="utf-8"
    )


    headings = None


    if args.metadata:

        headings = load_headings_from_metadata(
            args.metadata
        )


    chunks = chunk_document(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        headings=headings,
    )


    output = (
        args.output
        or args.input.with_name(
            args.input.stem + "_chunks.jsonl"
        )
    )


    save_chunks_to_jsonl(
        chunks,
        output
    )


    print(
        f"Created {len(chunks)} chunks"
    )

    print(
        f"Saved to {output}"
    )



if __name__ == "__main__":
    main()