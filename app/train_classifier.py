import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_labeled_features(json_path):
    
    """
    Loads a labeled feature set from a JSON file where each entry is:
    {
        "text": ..., "page": ..., "text_len": ..., ..., "label": "H1" or "BODY" etc.
    }
    """
    
    return pd.read_json(json_path)


def train_and_save_model(data_path, save_path=r"model\sk_model.pkl"):
    # Load data
    df = load_labeled_features(data_path)

    # Prepare features and labels
    feature_cols = [c for c in df.columns if c not in ("text", "page", "label")]
    if not feature_cols:
        raise ValueError("No feature columns found in the JSON.")
    X = df[feature_cols]
    y = df["label"]

    # Encode labels
    y_encoded = y.astype("category").cat.codes
    label_mapping = dict(enumerate(y.astype("category").cat.categories))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict & Evaluate (only on classes present in y_test)
    y_pred = model.predict(X_test)
    present_codes = sorted(set(y_test))
    present_names = [label_mapping[c] for c in present_codes]

    print(classification_report(
        y_test,
        y_pred,
        labels=present_codes,
        target_names=present_names
    ))

    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Save model and label mapping
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "labels": label_mapping, "feature_names": feature_cols}, f)
    print(f"✅ Model and label mapping saved to {save_path}")


if __name__ == "__main__":
    # Avoid unicode-escape issues on Windows
    data_path = r"C:\Users\lenovo\Desktop\loma_loma\LOMA_LOMA\examples\labeled_features_1.json"
    train_and_save_model(data_path)