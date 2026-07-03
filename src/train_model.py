"""
train_model.py
---------------
Trains a classifier (RandomForest by default) to predict market regime
from the engineered features, and saves the fitted model + label encoder.

Train/test split is done by a FIXED DATE, not a percentage of rows. This
matters: a percentage-based split means the test period silently shifts
whenever you change how much historical data you load (e.g. extending
the start date pulls in more rows, which moves the 80% mark). A
date-based split keeps the test period a controlled constant across
experiments, so results stay genuinely comparable.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_50d",
    "ma20_ma50_diff",
    "vol_20d", "vol_50d",
    "rsi_14", "macd_hist",
    "vix_level", "vix_change_5d",
    "rel_volume",
]

# Fixed test-period boundary. 2018-01-01 onward deliberately covers a mix
# of conditions: the Dec 2018 selloff, the COVID crash (2020), the 2022
# bear market, and calm bull-market stretches -- so evaluation isn't
# accidentally too easy (all-calm) or too hard (all-crisis).
DEFAULT_TEST_START_DATE = "2018-01-01"


def prepare_xy(df: pd.DataFrame):
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols]
    y = df["regime"]
    return X, y, cols


def train(df: pd.DataFrame, model_type: str = "random_forest",
          test_start_date: str = DEFAULT_TEST_START_DATE, random_state: int = 42):
    """
    Trains a model on a chronological split: everything before
    `test_start_date` is training data, everything on/after it is the
    held-out test set. Never shuffled, since this is time-series data
    and shuffling would leak future information into training.
    """
    X, y, feature_cols = prepare_xy(df)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    train_mask = X.index < pd.Timestamp(test_start_date)
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=random_state, class_weight="balanced"
        )

    model.fit(X_train_scaled, y_train)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "test_index": X_test.index,
        "test_start_date": test_start_date,
    }
    return artifacts


def save_model(artifacts: dict, filename: str = "regime_model.pkl") -> Path:
    out_path = MODELS_DIR / filename
    joblib.dump(
        {
            "model": artifacts["model"],
            "scaler": artifacts["scaler"],
            "encoder": artifacts["encoder"],
            "feature_cols": artifacts["feature_cols"],
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    features = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
        index_col=0, parse_dates=True,
    )
    artifacts = train(features, model_type="random_forest")
    path = save_model(artifacts)
    print(f"Model saved -> {path}")
    print(f"Train period: {artifacts['X_test'].index.min()} is first test date")
    print(f"Train rows: {len(artifacts['X_test']) == 0}")
    print(f"Test rows: {len(artifacts['X_test'])}")

    importances = pd.Series(
        artifacts["model"].feature_importances_, index=artifacts["feature_cols"]
    ).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)