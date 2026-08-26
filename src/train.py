import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "water_potability.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
MEDIAN_PATH = Path(__file__).resolve().parent.parent / "models" / "medians.pkl"

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, y_train, y_test, medians

def train():
    print("Script started")
    print(f"Looking for CSV at: {DATA_PATH}")
    X_train, X_test, y_train, y_test, medians = load_and_prepare_data()

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # metric you were throwing away
    y_pred = model.predict(X_test)
    print(f"macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(MEDIAN_PATH, "wb") as f:
        pickle.dump(medians, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Medians saved to {MEDIAN_PATH}")

if __name__ == "__main__":
    train()


