"""
compare_xgboost.py
--------------------
Throwaway script: compares XGBoost against the current Random Forest
config, on both the 4-class ("regime") and binary ("regime_binary")
targets, using the same fixed test-period split as train_model.py.
Delete once you've decided whether to adopt XGBoost.
"""

from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from train_model import prepare_xy, DEFAULT_TEST_START_DATE

features = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
    index_col=0, parse_dates=True,
)


def run_comparison(target_col: str):
    X, y, cols = prepare_xy(features, target_col=target_col)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    train_mask = X.index < pd.Timestamp(DEFAULT_TEST_START_DATE)
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_enc[train_mask], y_enc[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weights: RF gets class_weight="balanced" natively; XGBoost
    # doesn't support that directly for multiclass, so we compute
    # per-sample weights manually to give it the same treatment.
    class_counts = pd.Series(y_train).value_counts()
    weight_per_class = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = pd.Series(y_train).map(weight_per_class).values

    configs = {
        "Random Forest (current)": (
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                random_state=42, class_weight="balanced"
            ),
            None,  # RF doesn't need manual sample_weight
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=42, eval_metric="mlogloss"
            ),
            sample_weights,
        ),
    }

    print("\n" + "#" * 70)
    print(f"TARGET: {target_col}")
    print("#" * 70)

    for name, (model, weights) in configs.items():
        if weights is not None:
            model.fit(X_train_s, y_train, sample_weight=weights)
        else:
            model.fit(X_train_s, y_train)

        preds = model.predict(X_test_s)
        print("=" * 60)
        print(name)
        print("=" * 60)
        print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))


run_comparison("regime")
run_comparison("regime_binary")