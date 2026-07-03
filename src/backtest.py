"""
backtest.py
-----------
Turns predicted regimes into a simple allocation strategy and compares
it against buy-and-hold SPY.

Allocation rule:
    Bull            -> 100% SPY
    Bear            -> 0% SPY (cash)
    Sideways        -> 50% SPY
    High_Volatility -> 25% SPY
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ALLOCATION = {
    "Bull": 1.0,
    "Bear": 0.0,
    "Sideways": 0.5,
    "High_Volatility": 0.25,
}

ALLOCATION_BINARY = {
    "Risk_On": 1.0,
    "Risk_Off": 0.0
}


def run_backtest(spy_close: pd.Series, predicted_regimes: pd.Series, allocation_map: dict = None) -> pd.DataFrame:
    """
    spy_close: daily close prices, indexed by date
    predicted_regimes: predicted regime label per date (same index, or subset)
    """
    allocation_map = allocation_map or ALLOCATION
    df = pd.DataFrame(index=predicted_regimes.index)
    df["close"] = spy_close.reindex(df.index)
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["regime"] = predicted_regimes
    df["allocation"] = df["regime"].map(allocation_map).fillna(0.5)

    # Strategy return = allocation applied to *that day's* return
    df["strategy_return"] = df["allocation"].shift(1).fillna(0) * df["daily_return"]

    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    return df


def performance_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    strat_stats = performance_stats(df["strategy_return"])
    bh_stats = performance_stats(df["daily_return"])
    summary = pd.DataFrame({"Buy & Hold": bh_stats, "Regime Strategy": strat_stats}).T
    return summary


def plot_equity_curves(df: pd.DataFrame, save: bool = True, filename: str = "equity_curve.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["buy_hold_equity"], label="Buy & Hold SPY")
    ax.plot(df.index, df["strategy_equity"], label="Regime Strategy")
    ax.set_title("Regime Strategy vs. Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, dpi=150)
        print(f"Saved equity curve -> {out_path}")

    plt.close(fig)

def confidence_filtered_regimes(model, X_test_scaled, encoder, test_index,
                                  threshold: float = 0.45, default_regime: str = "Sideways") -> pd.Series:
    """
    Only trust the model's predicted regime when its confidence
    (max predicted probability) clears `threshold`. Otherwise, fall
    back to `default_regime` (neutral) rather than acting on a guess
    the model itself isn't sure about.
    """
    probs = model.predict_proba(X_test_scaled)
    max_conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred_labels = encoder.inverse_transform(pred_idx)

    final_labels = np.where(max_conf >= threshold, pred_labels, default_regime)
    return pd.Series(final_labels, index=test_index)

if __name__ == "__main__":
    # Example standalone run using actual (not predicted) regimes,
    # just to sanity check the mechanics.
    features = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
        index_col=0, parse_dates=True,
    )
    from data_loader import download_ticker
    spy_raw = download_ticker("SPY")
    spy_close = spy_raw["Close"].reindex(features.index)

    bt = run_backtest(spy_close, features["regime"])
    print(summarize(bt))
    plot_equity_curves(bt)