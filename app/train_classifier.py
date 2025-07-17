import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def load_labeled_features(json_path):
    """
    Loads a labeled feature set from a JSON file where each entry is:
    {
        "text": ..., "page": ..., "text_len": ..., ..., "label": "H1" or "BODY" etc.
    }
    """
    return pd.read_json(json_path)


def train_and_save_model(data_path, save_path="model/sk_model.pkl"):
    df = pd.read_json(data_path)

    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in ["text", "page", "label"]]
    X = df[feature_cols]
    y = df["label"]

    # Encode labels
    y_encoded = y.astype("category").cat.codes
    label_mapping = dict(enumerate(y.astype('category').cat.categories))

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

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.values())))

    # Save model and label mapping
    with open(save_path, 'wb') as f:
        pickle.dump({'model': model, 'labels': label_mapping}, f)
    print(f"✅ Model and label mapping saved to {save_path}")


if __name__ == "__main__":
    data_path = "examples/labeled_features.json"  # adjust if needed
    train_and_save_model(data_path)
