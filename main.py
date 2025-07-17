import argparse
import json
import os
import pickle

from app.ocr_utils import get_page_lines
from app.features import extract_features_from_line
from app.classifier import predict_labels
from app.output_format import structure_output

def process_pdf(pdf_path, model, label_mapping):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    feature_dicts = []
    for line in get_page_lines(pdf_path):
        feats = extract_features_from_line(line, line["page"])
        if feats:
            feature_dicts.append(feats)

    if not feature_dicts:
        print("⚠️  No features extracted. Check OCR or feature logic.")
        return {}

    predictions = predict_labels(feature_dicts, model, label_mapping)
    return structure_output(predictions)


def main():
    parser = argparse.ArgumentParser(description="PDF Heading Extraction Pipeline")
    parser.add_argument("--pdf",   required=True, help="Path to input PDF file")
    parser.add_argument("--model", default="app/model/sk_model.pkl",
                        help="Path to trained model pickle (contains 'model' and 'labels')")
    parser.add_argument("--out",   help="Path to save output JSON (optional)")
    args = parser.parse_args()

    # Load model bundle
    if not os.path.isfile(args.model):
        print(f"❌ Model file not found: {args.model}")
        exit(1)

    with open(args.model, "rb") as mf:
        bundle = pickle.load(mf)

    if not (isinstance(bundle, dict) and "model" in bundle and "labels" in bundle):
        print("❌ Invalid model format: expected dict with 'model' and 'labels'")
        exit(1)

    model = bundle["model"]
    label_mapping = bundle["labels"]

    try:
        result = process_pdf(args.pdf, model, label_mapping)
        json_str = json.dumps(result, indent=2, ensure_ascii=False)

        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"✅ Output saved to {args.out}")
        else:
            print(json_str)

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
