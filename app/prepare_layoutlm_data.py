import json
from sklearn.model_selection import train_test_split

# Map your string labels to integer IDs
LABEL2ID = {"TITLE": 0, "H1": 1, "H2": 2, "H3": 3, "BODY": 4}

def main(input_json, train_out="lm_finetune/lm_train.json", test_out="lm_finetune/lm_test.json"):
    # 1) Load your fully labeled feature list
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Build records suitable for LayoutLM: text, bbox, label
    records = []
    for rec in data:
        x0, y0 = rec["left"], rec["top"]
        x1, y1 = x0 + rec["width"], y0 + rec["height"]
        label_id = LABEL2ID.get(rec["label"], LABEL2ID["BODY"])
        records.append({
            "text": rec["text"],
            "bbox": [x0, y0, x1, y1],
            "label": label_id
        })

    # 3) Split into train/test with stratification
    train, test = train_test_split(
        records,
        test_size=0.1,
        random_state=42,
        stratify=[r["label"] for r in records]
    )

    # 4) Write JSON files for Huggingface loading
    for arr, path in ((train, train_out), (test, test_out)):
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(arr)} examples to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare JSON for LayoutLM fine‑tuning")
    parser.add_argument("--input", required=True, help="Labeled features JSON (list of recs)")
    parser.add_argument("--train_out", default="lm_finetune/lm_train.json", help="Training output JSON path")
    parser.add_argument("--test_out",  default="lm_finetune/lm_test.json", help="Test output JSON path")
    args = parser.parse_args()
    main(args.input, args.train_out, args.test_out)
