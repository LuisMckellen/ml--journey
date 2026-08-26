import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "water_potability.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
print("Script started")
print("Looking for CSV at:", DATA_PATH)
print("CSV exists:", DATA_PATH.exists())


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()