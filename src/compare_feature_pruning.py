"""
compare_feature_pruning.py
----------------------------
Throwaway script: tests dropping the two weakest-importance features
(rel_volume, vix_change_5d) against the full feature set, on both
targets, using the same fixed test-period split as train_model.py.
Delete once you've decided whether to prune.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from train_model import FEATURE_COLUMNS, DEFAULT_TEST_START_DATE

features = pd.read_csv(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
    index_col=0, parse_dates=True,
)

PRUNED_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("rel_volume", "vix_change_5d")]


def run_comparison(target_col: str):
    y_full = features[target_col]
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_full)

    train_mask = features.index < pd.Timestamp(DEFAULT_TEST_START_DATE)
    test_mask = ~train_mask

    print("\n" + "#" * 70)
    print(f"TARGET: {target_col}")
    print("#" * 70)

    for label, cols in [("Full feature set", FEATURE_COLUMNS), ("Pruned (no rel_volume/vix_change_5d)", PRUNED_COLUMNS)]:
        X = features[cols]
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_enc[train_mask], y_enc[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=42, class_weight="balanced"
        )
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        print("=" * 60)
        print(label)
        print("=" * 60)
        print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))


run_comparison("regime")
run_comparison("regime_binary")