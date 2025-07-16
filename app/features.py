import re


def extract_features(ocr_dict, page_number):
    """
    Extracts features from OCR dictionary for each detected text line.
    Returns a list of dicts with features for each text box.
    """
    features = []
    n = len(ocr_dict['text'])

    for i in range(n):
        text = ocr_dict['text'][i].strip()
        if not text:
            continue

        height = int(ocr_dict['height'][i])
        width = int(ocr_dict['width'][i])
        top = int(ocr_dict['top'][i])
        left = int(ocr_dict['left'][i])

        # Feature engineering
        text_len = len(text)
        num_words = len(text.split())
        upper_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1e-5)
        has_colon = ":" in text
        ends_with_numbering = bool(re.search(r"\d+[.)]$", text))
        is_title_case = text.istitle()

        features.append({
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
            "is_title_case": int(is_title_case)
        })

    return features
