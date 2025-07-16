import json
import pickle
import xgboost as xgb
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
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def train_and_save_model(data, save_path="model/xgb_model.pkl"):
    df = pd.DataFrame(data)

    # Extract features and labels
    feature_cols = [col for col in df.columns if col not in ["text", "page", "label"]]
    X = df[feature_cols]
    y = df["label"]

    # Encode labels
    y = y.astype("category").cat.codes

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    labeled_data_path = "examples/labeled_features.json"  # Adjust path as needed
    data = load_labeled_features(labeled_data_path)
    train_and_save_model(data)
