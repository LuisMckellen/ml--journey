import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report
from xgboost import XGBClassifier
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"
SEED = 42

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    return train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)


def validate():
    print(f"xgboost {xgboost.__version__}")
    print(f"Looking for CSV at: {DATA_PATH}")
    X_train,X_test,y_train,y_test=load_and_prepare_data()
   
    recall_0_scores = []
    recall_1_scores = []
    macro_f1_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        fold_medians = X_tr.median()          
        X_tr = X_tr.fillna(fold_medians)
        X_val = X_val.fillna(fold_medians)     
 
        model = XGBClassifier(random_state=42,n_estimators=100, max_depth=6, learning_rate=0.3)
        model.fit(X_tr,y_tr)
        y_val_pred=model.predict(X_val)   
        recall_0_scores.append(recall_score(y_val,y_val_pred,pos_label=0))
        recall_1_scores.append(recall_score(y_val,y_val_pred,pos_label=1))
        macro_f1_scores.append(f1_score(y_val,y_val_pred,average='macro'))
        print(f"Fold {fold}: recall_0={recall_0_scores[-1]:.4f}, recall_1={recall_1_scores[-1]:.4f}, macro_f1={macro_f1_scores[-1]:.4f}")
    print(f"\nrecall_0: {np.mean(recall_0_scores):.4f} ± {np.std(recall_0_scores, ddof=1):.4f}")
    print(f"recall_1: {np.mean(recall_1_scores):.4f} ± {np.std(recall_1_scores, ddof=1):.4f}")
    print(f"macro_f1: {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores, ddof=1):.4f}")      
if __name__ == "__main__":
    validate()
