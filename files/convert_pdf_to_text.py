# import libraries
import pymupdf  # pymupdf
import re

# filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.pdf"
# output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.txt"
filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.pdf"
output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.txt"

# filename = r"E:\0-daily\Projects\rag-qa-system\data\5.pdf"
# output_file = r"E:\0-daily\Projects\rag-qa-system\data\5.txt"


def extract_text_with_filters(path, min_font_size=9, y_margin_top=0.15, y_margin_bottom=0.85):
    """Extract text from PDF with font size and position filters."""
    doc = pymupdf.open(path)
    pages_text = []
    
    for page in doc:
        height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        lines_text = []
        
        for block in blocks:
            if block.get("type") != 0:  # Only text blocks
                continue
                
            for line in block.get("lines", []):
                y = line["bbox"][1]  # top position
                
                # Filter by position (remove header/footer)
                if not (y_margin_top * height < y < y_margin_bottom * height):
                    continue
                
                # Filter by font size (remove footnotes)
                for span in line.get("spans", []):
                    size = span.get("size")
                    if size is not None and size >= min_font_size:
                        lines_text.append(span["text"])
        
        pages_text.append(" ".join(lines_text))
    
    doc.close()
    return "\n".join(pages_text)




# Optional: Inspect font sizes first to find the right threshold
def inspect_font_sizes(path, num_pages=1):
    """Inspect font sizes in the PDF to determine the right min_font_size threshold."""
    doc = pymupdf.open(path)
    
    for i, page in enumerate(doc):
        if i >= num_pages:
            break
            
        print(f"\n=== Page {i+1} ===")
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
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


def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)



# Extract text with font size and position filters
text_pages = extract_text_with_filters(filename, min_font_size=9)


# Save to file
save_text_to_file(text_pages, output_file)

print("Done.")