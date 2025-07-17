# label_helper.py

import argparse
import json
from app.ocr_utils import get_page_lines
from app.features import extract_features_from_line

def main(pdf_path, out_path):
    all_feats = []
    line_count = 0

    for line_dict in get_page_lines(pdf_path):
        feat = extract_features_from_line(line_dict, page_number=line_dict["page"])
        if feat:
            all_feats.append(feat)
            line_count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_feats, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {line_count} lines from {pdf_path}")
    print(f"📄 Saved labeled feature file to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unlabeled features from PDF for manual labeling")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", required=True, help="Output JSON file for extracted features")
    args = parser.parse_args()

    main(args.pdf, args.out)
