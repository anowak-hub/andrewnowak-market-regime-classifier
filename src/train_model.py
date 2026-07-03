"""
train_model.py
---------------
Trains a classifier (RandomForest by default) to predict market regime
from the engineered features, and saves the fitted model + label encoder.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def prepare_xy(df: pd.DataFrame):
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols]
    y = df["regime"]
    return X, y, cols


def train(df: pd.DataFrame, model_type: str = "random_forest", test_size: float = 0.2,
          random_state: int = 42):
    """
    Trains a model on a chronological (non-shuffled) train/test split,
    since this is time-series data and shuffling would leak future info.
    """
    X, y, feature_cols = prepare_xy(df)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=random_state, class_weight="balanced"
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

    importances = pd.Series(
        artifacts["model"].feature_importances_, index=artifacts["feature_cols"]
    ).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)