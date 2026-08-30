import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/water_potability.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

comparison = pd.DataFrame({
    "full_data": X.median(),
    "train_only": X_train.median(),
    "n_missing": X.isna().sum(),
})
comparison["diff"] = comparison["full_data"] - comparison["train_only"]
print(comparison.to_string())