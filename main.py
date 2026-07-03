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
    print("STEP 5: Backtesting (out-of-sample, confidence-filtered)")
    print("=" * 60)

    from backtest import confidence_filtered_regimes  # add to the import line at top instead if you prefer

    predicted_series = confidence_filtered_regimes(
        model=artifacts["model"],
        X_test_scaled=artifacts["X_test_scaled"],
        encoder=artifacts["encoder"],
        test_index=artifacts["test_index"],
        threshold=0.45,
    )

    predicted_series = confidence_filtered_regimes(
        model=artifacts["model"],
        X_test_scaled=artifacts["X_test_scaled"],
        encoder=artifacts["encoder"],
        test_index=artifacts["test_index"],
        threshold=0.45,
    )

    # --- diagnostic: how often does the model actually clear the threshold? ---
    probs = artifacts["model"].predict_proba(artifacts["X_test_scaled"])
    max_conf = probs.max(axis=1)
    print(f"% of predictions above 0.45 confidence: {(max_conf >= 0.45).mean():.2%}")
    print(f"Mean confidence: {max_conf.mean():.3f}")
    # --- end diagnostic ---

    spy_close = data["SPY"]["Close"].reindex(features.index)

    spy_close = data["SPY"]["Close"].reindex(features.index)
    bt = run_backtest(spy_close, predicted_series)
    summary = summarize(bt)
    print(summary)
    plot_equity_curves(bt)

    print("\nPipeline complete. See figures/ and models/ for outputs.")


if __name__ == "__main__":
    main()