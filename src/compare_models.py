"""
compare_models.py
------------------
Throwaway script: compares RF vs Logistic Regression, with and without
class_weight balancing. Delete once you've picked a direction.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from train_model import prepare_xy
from sklearn.preprocessing import LabelEncoder, StandardScaler

features = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
    index_col=0, parse_dates=True,
)

X, y, cols = prepare_xy(features)
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

configs = {
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest (unweighted)": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Logistic Regression (unweighted)": LogisticRegression(max_iter=1000),
}

for name, model in configs.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))