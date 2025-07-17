import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


def train_and_save_model(data_path, save_path="model/sk_model.pkl"):
    df = pd.read_json(data_path)

    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in ["text", "page", "label"]]
    X = df[feature_cols]
    y = df["label"]

    # Encode labels to integers
    y_encoded = y.astype("category").cat.codes
    label_mapping = dict(enumerate(y.astype("category").cat.categories))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    unique_classes = np.unique(y_test)
    target_names = [label_mapping[i] for i in unique_classes]

    print("📊 Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=unique_classes,
        target_names=target_names
    ))

    # Save model and label mapping
    with open(save_path, "wb") as f:
        pickle.dump({
            "model": model,
            "labels": label_mapping
        }, f)

    print(f"✅ Trained model saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/labeled_features_1.json", help="Path to labeled features JSON")
    args = parser.parse_args()

    train_and_save_model(args.data)
