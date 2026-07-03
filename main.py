"""
main.py
-------
Runs the full pipeline end to end:
  1. Load data (SPY, VIX)
  2. Engineer features + labels
  3. Train model
  4. Evaluate model
  5. Backtest regime-based strategy vs buy & hold (OUT-OF-SAMPLE,
     using the model's PREDICTED regimes on the held-out test period)

Run from the project root:
    python main.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_all, download_ticker  # noqa: E402
from feature_engineering import build_features, save_features  # noqa: E402
from train_model import train, save_model  # noqa: E402
from evaluate import evaluate  # noqa: E402
from backtest import run_backtest, summarize, plot_equity_curves  # noqa: E402


def main():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    data = load_all()

    print("\n" + "=" * 60)
    print("STEP 2: Engineering features")
    print("=" * 60)
    features = build_features(data["SPY"], data["VIX"])
    save_features(features)
    print(f"Feature set: {features.shape[0]} rows, {features.shape[1]} cols")
    print(features["regime"].value_counts())

    print("\n" + "=" * 60)
    print("STEP 3: Training model")
    print("=" * 60)
    artifacts = train(features, model_type="random_forest")
    save_model(artifacts)

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating model")
    print("=" * 60)
    results = evaluate(artifacts)

    print("\n" + "=" * 60)
    print("STEP 5: Backtesting (out-of-sample, using PREDICTED regimes)")
    print("=" * 60)

    predicted_labels = artifacts["encoder"].inverse_transform(results["predictions"])
    predicted_series = pd.Series(predicted_labels, index=artifacts["test_index"])

    spy_close = data["SPY"]["Close"].reindex(features.index)
    bt = run_backtest(spy_close, predicted_series)
    summary = summarize(bt)
    print(summary.to_string())
    plot_equity_curves(bt)

    print("\n" + "=" * 60)
    print("STEP 6: Training BINARY model (Risk_Off vs Risk_On)")
    print("=" * 60)
    binary_artifacts = train(features, model_type="random_forest", target_col="regime_binary")
    save_model(binary_artifacts, filename="regime_model_binary.pkl")

    print("\n" + "=" * 60)
    print("STEP 7: Evaluating BINARY model")
    print("=" * 60)
    binary_results = evaluate(binary_artifacts, filename="confusion_matrix_binary.png")

    print("\n" + "=" * 60)
    print("STEP 8: Backtesting BINARY model (out-of-sample)")
    print("=" * 60)
    binary_pred_labels = binary_artifacts["encoder"].inverse_transform(binary_results["predictions"])
    binary_pred_series = pd.Series(binary_pred_labels, index=binary_artifacts["test_index"])

    from backtest import ALLOCATION_BINARY
    binary_bt = run_backtest(spy_close, binary_pred_series, allocation_map=ALLOCATION_BINARY)
    binary_summary = summarize(binary_bt)
    print(binary_summary.to_string())
    plot_equity_curves(binary_bt, filename="equity_curve_binary.png")

    print("\nPipeline complete. See figures/ and models/ for outputs.")


if __name__ == "__main__":
    main()