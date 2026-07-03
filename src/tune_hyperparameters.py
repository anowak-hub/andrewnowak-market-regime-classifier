"""
tune_hyperparameters.py
------------------------
Throwaway script: grid-searches Random Forest hyperparameters using
time-series-aware cross-validation (TimeSeriesSplit), then checks the
best config against the real held-out test set. Delete once done.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from train_model import prepare_xy

features = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
    index_col=0, parse_dates=True,
)

X, y, cols = prepare_xy(features)
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Hold out the same final 20% as train_model.py, untouched by CV entirely
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "min_samples_leaf": [1, 5, 10],
}

tscv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid=param_grid,
    cv=tscv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)

print("Running grid search (this may take a few minutes)...\n")
grid.fit(X_train_s, y_train)

print("=" * 60)
print(f"Best params: {grid.best_params_}")
print(f"Best CV macro-F1: {grid.best_score_:.3f}")
print("=" * 60)

print("\nPerformance on the REAL held-out test set:")
best_model = grid.best_estimator_
preds = best_model.predict(X_test_s)
print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))