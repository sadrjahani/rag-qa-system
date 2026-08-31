"""
Convert a PDF into a cleaned .txt file, stripping headers/footers and
footnotes based on vertical position and font size. Also detects heading
candidates (font noticeably larger than body text) and saves them, along
with page counts and detection settings, to a sidecar .meta.json file for
downstream use (e.g. chunking.py's structure-based splitting).

Usage:
    python convert_pdf_to_text.py path/to/file.pdf
    python convert_pdf_to_text.py path/to/file.pdf -o path/to/output.txt
    python convert_pdf_to_text.py path/to/file.pdf --min-font-size 9
    python convert_pdf_to_text.py path/to/file.pdf --heading-ratio 1.2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pymupdf  # pymupdf


def detect_body_font_size(path, sample_pages=5):
    """Guess the main body-text font size by finding the size with the most
    characters across a sample of pages. Footnotes/captions are almost
    always smaller than this, headings are almost always bigger.
    """
    doc = pymupdf.open(path)
    char_counts_by_size = Counter()

    for i, page in enumerate(doc):
        if i >= sample_pages:
            break
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # only text blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = span.get("size")
                    if size is not None and text.strip():
                        char_counts_by_size[round(size, 1)] += len(text)

    doc.close()

    if not char_counts_by_size:
        return 9.0  # fallback if nothing was found

    return char_counts_by_size.most_common(1)[0][0]


def extract_text_with_filters(
    path,
    min_font_size=None,
    y_margin_top=0.15,
    y_margin_bottom=0.85,
    heading_ratio=1.15,
    max_heading_words=12,
):
    """Extract text from a PDF with font-size and position filters, and
    collect heading candidates + page-level metadata along the way.

    - Lines outside the [y_margin_top, y_margin_bottom] band of the page
      height are dropped (removes running headers/footers).
    - Spans smaller than min_font_size are dropped from the body text
      (removes footnotes).
    - Line breaks are preserved, so paragraph structure survives.
    - Any line whose largest span is >= body_font_size * heading_ratio,
      and short enough to plausibly be a title (<= max_heading_words),
      is recorded as a heading candidate with its page number.

    Returns (text, metadata) where metadata is a dict ready to be saved
    as JSON.
    """
    body_font_size = detect_body_font_size(path)

    if min_font_size is None:
        min_font_size = round(body_font_size - 0.5, 1)
        print(f"Auto-detected min_font_size: {min_font_size}")

    doc = pymupdf.open(path)
    pages_text = []
    headings = []

    for page_index, page in enumerate(doc):
        height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        page_lines = []

        for block in blocks:
            if block.get("type") != 0:  # only text blocks
                continue

            for line in block.get("lines", []):
                y = line["bbox"][1]  # top position

                # filter by position (remove header/footer)
                if not (y_margin_top * height < y < y_margin_bottom * height):
                    continue

                spans = line.get("spans", [])
                if not spans:
                    continue

                max_span_size = max(
                    (s["size"] for s in spans if s.get("size") is not None),
                    default=0,
                )
                line_text_raw = "".join(s.get("text", "") for s in spans).strip()

                # heading candidate: noticeably bigger font than body, short line
                if (
                    line_text_raw
                    and max_span_size >= body_font_size * heading_ratio
                    and len(line_text_raw.split()) <= max_heading_words
                ):
                    headings.append({
                        "text": line_text_raw,
                        "page": page_index + 1,
                        "font_size": round(max_span_size, 1),
                    })

                # body text: font-size filter (removes footnotes), keep line breaks
                line_spans = [
                    s["text"] for s in spans
                    if s.get("size") is not None and s["size"] >= min_font_size
                ]
                if line_spans:
                    page_lines.append(" ".join(line_spans))

        pages_text.append("\n".join(page_lines))

    num_pages = doc.page_count
    doc.close()

    text = "\n\n".join(pages_text)  # blank line between pages

    metadata = {
        "source_pdf": str(path),
        "num_pages": num_pages,
        "body_font_size": body_font_size,
        "min_font_size_used": min_font_size,
        "y_margin_top": y_margin_top,
        "y_margin_bottom": y_margin_bottom,
        "heading_ratio": heading_ratio,
        "headings": headings,
    }

    return text, metadata


def save_text_to_file(text, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


def save_metadata_to_file(metadata, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def inspect_font_sizes(path, num_pages=1):
    """Print font size/font/y-position for each span, useful for manually
    picking a min_font_size or heading_ratio if auto-detection needs a
    sanity check.
    """
    doc = pymupdf.open(path)

    for i, page in enumerate(doc):
        if i >= num_pages:
            break

        print(f"\n=== Page {i + 1} ===")
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"][:50]
                    size = span.get("size")
                    font = span.get("font")
                    y = line["bbox"][1]
                    print(f"Size: {size:5.2f}, Font: {font}, Y: {y:6.2f}, Text: {text}")

    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Extract cleaned body text (+ heading/page metadata) from a PDF.")
    parser.add_argument("input", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to the output .txt file. Defaults to the input path with a .txt extension."
    )
    parser.add_argument(
        "--min-font-size", type=float, default=None,
        help="Minimum font size to keep in body text. If omitted, it's auto-detected from the PDF."
    )
    parser.add_argument(
        "--y-margin-top", type=float, default=0.15,
        help="Top margin fraction of page height to exclude (default: 0.15)."
    )
    parser.add_argument(
        "--y-margin-bottom", type=float, default=0.85,
        help="Bottom margin fraction of page height to exclude below (default: 0.85)."
    )
    parser.add_argument(
        "--heading-ratio", type=float, default=1.15,
        help="A line is a heading candidate if its font size >= body_font_size * this ratio (default: 1.15)."
    )
    parser.add_argument(
        "--max-heading-words", type=int, default=12,
        help="Max word count for a line to still qualify as a heading candidate (default: 12)."
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Skip writing the .meta.json sidecar file; only write the .txt output."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print span font sizes/positions for the first page instead of converting."
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_font_sizes(args.input)
        return

    output_path = args.output or args.input.with_suffix(".txt")

    text, metadata = extract_text_with_filters(
        args.input,
        min_font_size=args.min_font_size,
        y_margin_top=args.y_margin_top,
        y_margin_bottom=args.y_margin_bottom,
        heading_ratio=args.heading_ratio,
        max_heading_words=args.max_heading_words,
    )
    save_text_to_file(text, output_path)
    print(f"Done. Saved text to {output_path}")

    if not args.no_metadata:
        meta_path = output_path.with_suffix(".meta.json")
        save_metadata_to_file(metadata, meta_path)
        print(f"Saved metadata ({len(metadata['headings'])} heading candidates) to {meta_path}")


if __name__ == "__main__":
    main()
