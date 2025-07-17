# app/features.py
import re

def extract_features_from_line(line_dict, page_number):
    """
    Given a single line dict (from PyMuPDF or OCR fallback), return feature dict.
    Assumes keys: text, left, top, width, height, (optional: font_size, font_name).
    """
    text = line_dict.get("text", "").strip()
    if not text:
        return None

    height = line_dict.get("height", 0)
    width  = line_dict.get("width", 0)
    top    = line_dict.get("top", 0)
    left   = line_dict.get("left", 0)
    font_size = line_dict.get("font_size", 0)
    font_name = line_dict.get("font_name", "").lower()

    # feature engineering
    text_len = len(text)
    num_words = len(text.split())
    upper_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1e-5)
    has_colon = ":" in text
    ends_with_numbering = bool(re.search(r"\d+[.)]$", text))
    is_title_case = text.istitle()
    is_bold = int("bold" in font_name)

    return {
        "text": text,
        "page": page_number,
        "text_len": text_len,
        "num_words": num_words,
        "height": height,
        "width": width,
        "top": top,
        "left": left,
        "font_size": font_size,
        "is_bold": is_bold,
        "area": height * width,
        "aspect_ratio": width / (height + 1e-5),
        "upper_ratio": upper_ratio,
        "has_colon": int(has_colon),
        "ends_with_numbering": int(ends_with_numbering),
        "is_title_case": int(is_title_case),
    }
