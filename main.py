# main.py

import argparse
import json
from app.ocr_utils import pdf_to_images, extract_ocr_data, ocr_dict_to_lines
from app.features import extract_features_from_line
from app.classifier import predict_labels
from app.output_format import structure_output


def process_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    feature_dicts = []

    for page_num, img in enumerate(images, start=1):
        ocr = extract_ocr_data(img)
        lines = ocr_dict_to_lines(ocr)
        for line in lines:
            feats = extract_features_from_line(line, page_num)
            if feats:
                feature_dicts.append(feats)

    predictions = predict_labels(feature_dicts)
    structured = structure_output(predictions)

    return structured


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", required=False, help="Path to output JSON (optional)")
    args = parser.parse_args()

    output = process_pdf(args.pdf)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ Output saved to {args.out}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))
