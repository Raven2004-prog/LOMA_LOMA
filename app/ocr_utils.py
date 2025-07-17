# app/ocr_utils.py

import os
from collections import defaultdict

import fitz                    # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# If Poppler isn’t on your system PATH (Windows), set this to its bin folder:
POPPLER_PATH = None  # e.g. r"C:\poppler-24.02.0\Library\bin"
# DPI resolution for fallback OCR
OCR_DPI = 100
# If PyMuPDF returns fewer than this many lines, we trigger the OCR fallback
MIN_MUPDF_LINES = 3

# ─── PyMuPDF TEXT PARSER ──────────────────────────────────────────────────────

def extract_lines_mupdf(pdf_path):
    """
    Extract line-level text from each page of a PDF via PyMuPDF.
    Returns a list of pages, where each page is a list of dicts:
      { text, left, top, width, height, font_size, font_name }
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        lines = []

        for block in blocks:
            for line in block.get("lines", []):
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue

                span0 = line["spans"][0]
                x0, y0, x1, y1 = span0["bbox"]
                lines.append({
                    "text":      text,
                    "left":      x0,
                    "top":       y0,
                    "width":     x1 - x0,
                    "height":    y1 - y0,
                    "font_size": span0.get("size", 0),
                    "font_name": span0.get("font", "")
                })

        pages.append(lines)

    return pages

# ─── PDF→IMAGE & TESSERACT FALLBACK ───────────────────────────────────────────

def pdf_to_images(pdf_path, dpi=OCR_DPI):
    """
    Convert each PDF page to a PIL Image for Tesseract OCR fallback.
    """
    opts = {"dpi": dpi}
    if POPPLER_PATH:
        opts["poppler_path"] = POPPLER_PATH
    return convert_from_path(pdf_path, **opts)

def extract_ocr_data(image):
    """
    Run Tesseract OCR on a PIL Image; return the dict from image_to_data().
    """
    return pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

def ocr_dict_to_lines(ocr):
    """
    Group Tesseract’s word-level output into line-level entries:
      { text, left, top, width, height }
    """
    lines = defaultdict(lambda: {
        "words": [], "lefts": [], "tops": [], "rights": [], "bottoms": []
    })
    count = len(ocr.get("text", []))

    for i in range(count):
        txt = ocr["text"][i].strip()
        if not txt:
            continue
        key = (ocr["block_num"][i], ocr["par_num"][i], ocr["line_num"][i])
        x, y = ocr["left"][i], ocr["top"][i]
        w, h = ocr["width"][i], ocr["height"][i]
        lines[key]["words"].append(txt)
        lines[key]["lefts"].append(x)
        lines[key]["tops"].append(y)
        lines[key]["rights"].append(x + w)
        lines[key]["bottoms"].append(y + h)

    result = []
    for info in lines.values():
        text = " ".join(info["words"])
        left   = min(info["lefts"])
        top    = min(info["tops"])
        right  = max(info["rights"])
        bottom = max(info["bottoms"])
        result.append({
            "text":   text,
            "left":   left,
            "top":    top,
            "width":  right - left,
            "height": bottom - top
        })
    return result

# ─── UNIFIED LINE GENERATOR ───────────────────────────────────────────────────

def get_page_lines(pdf_path):
    """
    Generator yielding line dicts for each page in the PDF.
    Attempts PyMuPDF first; if a page has fewer than MIN_MUPDF_LINES lines,
    falls back to Tesseract OCR on that page’s image.

    Yields dicts:
      { text, left, top, width, height, page }
    """
    mupdf_pages = extract_lines_mupdf(pdf_path)

    for page_num, lines in enumerate(mupdf_pages, start=1):
        # If PyMuPDF detected too few lines, use OCR fallback
        if len(lines) < MIN_MUPDF_LINES:
            images = pdf_to_images(pdf_path)
            ocr = extract_ocr_data(images[page_num - 1])
            lines = ocr_dict_to_lines(ocr)

        # Annotate and clean up each line dict
        for line in lines:
            line["page"] = page_num
            line.pop("font_name", None)  # drop if unused downstream

        yield from lines
