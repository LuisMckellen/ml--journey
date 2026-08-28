import pickle
from pathlib import Path

import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.json"
MEDIAN_PATH = MODEL_DIR / "medians.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

SEED = 42


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # split first, then learn medians on train only — imputing before the
    # split leaks test-set distribution into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, y_train, y_test, medians


def train():
    print(f"xgboost {xgboost.__version__}")
    print(f"Looking for CSV at: {DATA_PATH}")

    X_train, X_test, y_train, y_test, medians = load_and_prepare_data()
    feature_order = X_train.columns.tolist()
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)

    print(f"\nmacro F1: {macro_f1:.4f}")
    print(f"recall_0: {recall_0:.4f} | recall_1: {recall_1:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("prediction distribution:")
    print(pd.Series(y_pred).value_counts().to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(MEDIAN_PATH, "wb") as f:
        pickle.dump(medians, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_order, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Medians saved to {MEDIAN_PATH}")
    print(f"Feature order saved to {FEATURES_PATH}")

    return {
        "macro_f1": macro_f1,
        "recall_0": recall_0,
        "recall_1": recall_1,
        "xgboost_version": xgboost.__version__,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    train()