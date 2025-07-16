from pdf2image import convert_from_path
import pytesseract

def pdf_to_images(pdf_path, dpi=150):

    images = convert_from_path(pdf_path, dpi=dpi)  #Convert PDF to images
    return images

def extract_ocr_data(image):
    
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)  #Extract OCR data from image



