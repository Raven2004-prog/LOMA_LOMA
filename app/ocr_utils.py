from pdf2image import convert_from_path
import pytesseract
from collections import defaultdict
def pdf_to_images(pdf_path, dpi=150):

    images = convert_from_path(pdf_path, dpi=dpi)  #Convert PDF to images
    return images

def extract_ocr_data(image):
    
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)  #Extract OCR data from image

def ocr_dict_to_lines(ocr):
    """
    Group Tesseract word-level output into line-level dicts.
    """
    lines = defaultdict(lambda: {'words': [], 'lefts': [], 'tops': [], 'rights': [], 'bottoms': []})
    n = len(ocr['text'])
    for i in range(n):
        txt = ocr['text'][i].strip()
        if not txt:
            continue
        key = (ocr['block_num'][i], ocr['par_num'][i], ocr['line_num'][i])
        x, y, w, h = (ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i])
        lines[key]['words'].append(txt)
        lines[key]['lefts'].append(x)
        lines[key]['tops'].append(y)
        lines[key]['rights'].append(x + w)
        lines[key]['bottoms'].append(y + h)

    result = []
    for (_, _, _), info in lines.items():
        text = " ".join(info['words'])
        left, top = min(info['lefts']), min(info['tops'])
        right, bottom = max(info['rights']), max(info['bottoms'])
        width, height = right - left, bottom - top
        result.append({
            'text': text,
            'left': left, 'top': top,
            'width': width, 'height': height
        })
    return result

