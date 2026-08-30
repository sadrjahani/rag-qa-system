# # importing required modules
# import PyPDF2
# import re

# # filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.pdf"
# # filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.pdf"
# # output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.txt"
# # output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.txt"
# filename = r"E:\0-daily\Projects\rag-qa-system\data\5.pdf"
# output_file = r"E:\0-daily\Projects\rag-qa-system\data\5.txt"

# # #ignore the first 150 lines and the last 150 lines for eliminate header and footer
# def visitor_body(text, cm, tm, fontDict, fontSize):
#     y = tm[5]
#     if  120 < y < 720:
#         # return text

#         if fontSize is not None and float(fontSize) > 3.0:
#             return text
#     return ""

# #convert pdf to text
# def convert_pdf_to_text(filename):
#     reader = PyPDF2.PdfReader( filename,strict=True)
#     text_pages = []

#     for page in reader.pages:
#         text_body = page.extract_text(visitor_text=visitor_body)
#         text_pages.append(text_body or "")

#     final_text = "\n".join(text_pages)
#     return final_text


# def remove_header_footer_patterns(text):
#     # Remove lines that are only a page number
#     text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

#     # Remove lines like: "252 | Chapter 5: Prompt Engineering"
#     text = re.sub(
#         r'^\s*\d+\s*\|\s*Chapter.*\n',
#         '',
#         text,
#         flags=re.MULTILINE | re.IGNORECASE
#     )

#     # Remove lines like: "Chapter 5: Prompt Engineering | 252"
#     text = re.sub(
#         r'^\s*Chapter.*\|\s*\d+\s*\n',
#         '',
#         text,
#         flags=re.MULTILINE | re.IGNORECASE
#     )

#     # Remove any line containing "| Chapter" (extra safety)
#     text = re.sub(
#         r'^\s*.*?\|\s*Chapter.*\n',
#         '',
#         text,
#         flags=re.MULTILINE | re.IGNORECASE
#     )

#     # Remove number attached to the beginning of text (e.g., "1In its short...")
#     text = re.sub(r'^\s*(\d+)([A-Z])', r'\2', text, flags=re.MULTILINE)

#     # Remove excessive consecutive blank lines
#     text = re.sub(r'\n{3,}', '\n\n', text)

#     return text

# #save text to file
# def save_text_to_file(text, filename):
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write(text)


# import pdfplumber

# def extract_text_with_pdfplumber(path):
#     pages_text = []
#     with pdfplumber.open(path) as pdf:
#         for page in pdf.pages:
#             # فقط متن‌هایی که در ناحیهٔ وسط صفحه هستند
#             height = page.height
#             words = page.extract_words()
#             body_words = [
#                 w for w in words
#                 if 0.15 * height < w["top"] < 0.85 * height  # حذف هدر/فوتر
#             ]
#             # wfor w in body_words
#             print(body_words[0].get("size"))
#             # body_words = [
#             #     w.get("size") for w in body_words
#             #     if w.get("size") >= 11 # حذف هدر/فوتر
#             # ]
#             text = " ".join(w["text"] for w in body_words)
#             pages_text.append(text)
#     return "\n".join(pages_text)

# tplumber = extract_text_with_pdfplumber(filename)
# print(tplumber)
# #convert pdf to text and save to file
# text_pages = convert_pdf_to_text(filename)
# text_pages = remove_header_footer_patterns(text_pages)
# save_text_to_file(text_pages, output_file)

# # print(text_pages)
import pymupdf  # pymupdf
import re

# filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.pdf"
# filename = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.pdf"
# output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 6 Rag & Agents.txt"
# output_file = r"E:\0-daily\Projects\rag-qa-system\data\chapter 5 prompt engineering.txt"
filename = r"E:\0-daily\Projects\rag-qa-system\data\5.pdf"
output_file = r"E:\0-daily\Projects\rag-qa-system\data\5.txt"


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



def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


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


# Extract text with font size and position filters
text_pages = extract_text_with_filters(filename, min_font_size=9)


# Save to file
save_text_to_file(text_pages, output_file)

print("Done.")