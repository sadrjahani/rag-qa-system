"""
Convert a PDF into a cleaned .txt file, stripping headers/footers and
footnotes based on vertical position and font size.

Usage:
    python convert_pdf_to_text.py path/to/file.pdf
    python convert_pdf_to_text.py path/to/file.pdf -o path/to/output.txt
    python convert_pdf_to_text.py path/to/file.pdf --min-font-size 9
"""

import argparse
from collections import Counter
from pathlib import Path

import pymupdf  # pymupdf


def auto_detect_body_font_size(path, sample_pages=5):
    """Guess the main body-text font size by finding the size with the most
    characters across a sample of pages. Footnotes/captions are almost
    always smaller than this, so it's a good default filter threshold.
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

    body_size = char_counts_by_size.most_common(1)[0][0]
    # keep the body size itself and anything bigger (headings); footnotes
    # are typically ~1-2pt smaller, so cut just below the body size
    return round(body_size - 0.5, 1)


def extract_text_with_filters(path, min_font_size=None, y_margin_top=0.15, y_margin_bottom=0.85):
    """Extract text from a PDF with font-size and position filters.

    - Lines outside the [y_margin_top, y_margin_bottom] band of the page
      height are dropped (removes running headers/footers).
    - Spans smaller than min_font_size are dropped (removes footnotes).
    - Line breaks are preserved, so paragraph structure survives.
    """
    if min_font_size is None:
        min_font_size = auto_detect_body_font_size(path)
        print(f"Auto-detected min_font_size: {min_font_size}")

    doc = pymupdf.open(path)
    pages_text = []

    for page in doc:
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

                # filter by font size (remove footnotes), keep line breaks
                line_spans = [
                    span["text"]
                    for span in line.get("spans", [])
                    if span.get("size") is not None and span["size"] >= min_font_size
                ]
                if line_spans:
                    page_lines.append(" ".join(line_spans))

        pages_text.append("\n".join(page_lines))

    doc.close()
    return "\n\n".join(pages_text)  # blank line between pages


def save_text_to_file(text, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


def inspect_font_sizes(path, num_pages=1):
    """Print font size/font/y-position for each span, useful for manually
    picking a min_font_size threshold if auto-detection needs a sanity check.
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
    parser = argparse.ArgumentParser(description="Extract cleaned body text from a PDF.")
    parser.add_argument("input", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to the output .txt file. Defaults to the input path with a .txt extension."
    )
    parser.add_argument(
        "--min-font-size", type=float, default=None,
        help="Minimum font size to keep. If omitted, it's auto-detected from the PDF."
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
        "--inspect", action="store_true",
        help="Print span font sizes/positions for the first page instead of converting."
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_font_sizes(args.input)
        return

    output_path = args.output or args.input.with_suffix(".txt")

    text = extract_text_with_filters(
        args.input,
        min_font_size=args.min_font_size,
        y_margin_top=args.y_margin_top,
        y_margin_bottom=args.y_margin_bottom,
    )
    save_text_to_file(text, output_path)
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
