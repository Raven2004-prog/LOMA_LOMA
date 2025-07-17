# app/features.py
import re

def extract_features_from_line(line_dict, page_number):
    """
    Given a single line dict (with keys: text, left, top, width, height)
    and page number, return a features dict.
    """
    text = line_dict['text'].strip()
    if not text:
        return None

    height = line_dict['height']
    width  = line_dict['width']
    top    = line_dict['top']
    left   = line_dict['left']

    # feature engineering
    text_len = len(text)
    num_words = len(text.split())
    upper_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1e-5)
    has_colon = ":" in text
    ends_with_numbering = bool(re.search(r"\d+[.)]$", text))
    is_title_case = text.istitle()

    return {
        "text": text,
        "page": page_number,
        "text_len": text_len,
        "num_words": num_words,
        "height": height,
        "width": width,
        "top": top,
        "left": left,
        "area": height * width,
        "aspect_ratio": width / (height + 1e-5),
        "upper_ratio": upper_ratio,
        "has_colon": int(has_colon),
        "ends_with_numbering": int(ends_with_numbering),
        "is_title_case": int(is_title_case),
    }
