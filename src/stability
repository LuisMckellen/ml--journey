"""Stability check for the water-potability model.

Three questions, one loop:
  1. How much does macro F1 move with the train/test split seed?
  2. How much does it move with the model seed alone (split fixed)?
  3. Does the leaky-vs-clean gap hold up across seeds, or was +0.0513 a
     seed-42 fluke?

Run:  python -m src.stability
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier
import xgboost

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "water_potability.csv"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.3)


def load():
    df = pd.read_csv(DATA_PATH)
    return df.drop("Potability", axis=1), df["Potability"]


def run(X, y, split_seed, model_seed, leaky):
    """Train once. If leaky, impute before the split (the old bug)."""
    if leaky:
        X_used = X.fillna(X.median())
        X_train, X_test, y_train, y_test = train_test_split(
            X_used, y, test_size=0.2, random_state=split_seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_seed, stratify=y
        )
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    model = XGBClassifier(random_state=model_seed, **PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "recall_0": recall_score(y_test, y_pred, pos_label=0),
        "recall_1": recall_score(y_test, y_pred, pos_label=1),
    }


def summarize(name, scores):
    arr = np.array(scores)
    print(
        f"{name:<28} mean {arr.mean():.4f}  std {arr.std(ddof=1):.4f}  "
        f"min {arr.min():.4f}  max {arr.max():.4f}  range {arr.max() - arr.min():.4f}"
    )


def main():
    print(f"xgboost {xgboost.__version__}\n")
    X, y = load()

    # --- 1 & 3: vary the split seed, model seed fixed at 42 -------------
    print("Split seed varies, model seed fixed at 42")
    print(f"{'seed':<6}{'clean':<10}{'leaky':<10}{'delta':<10}")
    clean_scores, leaky_scores, deltas = [], [], []
    for s in SEEDS:
        c = run(X, y, split_seed=s, model_seed=42, leaky=False)["macro_f1"]
        l = run(X, y, split_seed=s, model_seed=42, leaky=True)["macro_f1"]
        clean_scores.append(c)
        leaky_scores.append(l)
        deltas.append(l - c)
        print(f"{s:<6}{c:<10.4f}{l:<10.4f}{l - c:+.4f}")

    print()
    summarize("clean macro F1", clean_scores)
    summarize("leaky macro F1", leaky_scores)
    summarize("leaky - clean delta", deltas)

    # --- 2: fix the split at 42, vary only the model seed ---------------
    print("\nSplit seed fixed at 42, model seed varies")
    model_scores = []
    for s in SEEDS:
        m = run(X, y, split_seed=42, model_seed=s, leaky=False)["macro_f1"]
        model_scores.append(m)
        print(f"  model_seed={s}: {m:.4f}")

    print()
    summarize("model-seed-only macro F1", model_scores)

    # --- verdict --------------------------------------------------------
    split_std = np.std(clean_scores, ddof=1)
    model_std = np.std(model_scores, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)

    print("\n--- read-out ---")
    print(f"Split-driven spread : {split_std:.4f}")
    print(f"Model-driven spread : {model_std:.4f}")
    print(f"Leak effect         : {delta_mean:+.4f} ± {delta_std:.4f}")
    print(
        "\nIf the leak effect straddles zero, the +0.0513 at seed 42 was "
        "split noise, not a leakage advantage."
    )


if __name__ == "__main__":
    main()