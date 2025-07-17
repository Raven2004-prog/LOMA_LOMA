# main.py

import argparse
import json
import os
import pickle
from app.ocr_utils import pdf_to_images, extract_ocr_data, ocr_dict_to_lines
from app.features import extract_features_from_line
from app.classifier import predict_labels
from app.output_format import structure_output


def process_pdf(pdf_path, model, label_mapping):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images = pdf_to_images(pdf_path)
    feature_dicts = []

    for page_num, img in enumerate(images, start=1):
        print(f"📄 Processing page {page_num}")
        ocr_data = extract_ocr_data(img)
        lines = ocr_dict_to_lines(ocr_data)

        for line in lines:
            feats = extract_features_from_line(line, page_num)
            if feats:
                feature_dicts.append(feats)

    if not feature_dicts:
        print("⚠️  No features extracted. Check OCR or feature logic.")
        return {}

    predictions = predict_labels(feature_dicts, model, label_mapping)
    return structure_output(predictions)


def main():
    parser = argparse.ArgumentParser(description="PDF OCR + Classification Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--model", default="app/model/sk_model.pkl", help="Path to model pickle file")
    parser.add_argument("--out", help="Path to save output JSON")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"❌ Model file not found: {args.model}")
        exit(1)

    with open(args.model, "rb") as mf:
        model_bundle = pickle.load(mf)

    if not (isinstance(model_bundle, dict) and "model" in model_bundle and "labels" in model_bundle):
        print("❌ Invalid model format: expected dict with 'model' and 'labels'")
        exit(1)

    model = model_bundle["model"]
    label_mapping = model_bundle["labels"]

    try:
        result = process_pdf(args.pdf, model, label_mapping)
        output_json = json.dumps(result, indent=2, ensure_ascii=False)

        if args.out:
            out_dir = os.path.dirname(args.out) or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"✅ Output saved to {args.out}")
        else:
            print(output_json)

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
