# app/classifier.py

import pandas as pd

def predict_labels(feature_dicts, model, label_mapping):
    """
    Predict labels using the provided model. Return list of dicts with text, label, and page.
    """

    # Filter out non-numeric columns
    X = pd.DataFrame([
        {k: v for k, v in feat.items() if k not in ["text", "page", "label"]}
        for feat in feature_dicts
    ])

    preds = model.predict(X)

    predictions = [
        {
            "text": feat["text"],
            "label": label_mapping.get(pred, str(pred)),
            "page": feat.get("page", -1)
        }
        for feat, pred in zip(feature_dicts, preds)
    ]

    return predictions
