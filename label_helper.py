# label_helper.py
import argparse, json
from app.ocr_utils import pdf_to_images, extract_ocr_data, ocr_dict_to_lines
from app.features import extract_features_from_line

def main(pdf_path, out_path):
    images = pdf_to_images(pdf_path)
    all_feats = []

    for page_num, img in enumerate(images, start=1):
        ocr = extract_ocr_data(img)
        lines = ocr_dict_to_lines(ocr)

        for line in lines:
            feat = extract_features_from_line(line, page_num)
            if feat:
                all_feats.append(feat)

    with open(out_path, 'w') as f:
        json.dump(all_feats, f, indent=2)
    print(f"Saved {len(all_feats)} line‑level entries to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pdf, args.out)
