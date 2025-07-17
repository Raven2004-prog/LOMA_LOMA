# app/classifier.py

import pickle
import os

# Load the trained model and label mapping on import
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/sk_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    label_mapping = saved['labels']  # e.g., {0: 'BODY', 1: 'H1', 2: 'TITLE'}

# Reverse mapping to get label → index (if needed later)
label_inverse = {v: k for k, v in label_mapping.items()}


def predict_labels(feature_dicts):
    """
    Given a list of feature dicts (from feature extractor),
    returns a list of (text, predicted_label) pairs.
    """

    results = []

    for feat in feature_dicts:
        if feat is None:
            continue

        text = feat.get("text", "")
        # Build feature vector (exclude keys not used in training)
        input_features = [
            feat[key]
            for key in sorted(feat.keys())
            if key not in ["text", "page", "label"]
        ]
        pred = model.predict([input_features])[0]
        label = label_mapping[pred]

        results.append((text, label))

    return results
