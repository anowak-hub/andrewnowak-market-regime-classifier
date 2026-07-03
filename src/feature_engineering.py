"""
feature_engineering.py
-----------------------
Turns raw SPY / VIX price data into a clean, labeled ML dataset:

  - Trend features    (5/20/50-day returns, MA20-MA50 diff)
  - Volatility features (20/50-day rolling std of returns)
  - Momentum features  (RSI, MACD)
  - Fear features      (VIX level, 5-day VIX change)
  - Volume features    (relative volume)
  - Label              (Bull / Bear / Sideways, from FORWARD 20-day return)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    spy, vix: raw DataFrames from data_loader (must contain 'Close' and,
    for spy, 'Volume').
    Returns a single feature DataFrame indexed by date.
    """
    df = pd.DataFrame(index=spy.index)
    close = spy["Close"]

    # --- Trend features ---
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_50d"] = close.pct_change(50)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma20_ma50_diff"] = (ma20 - ma50) / close

    # --- Volatility features ---
    daily_ret = close.pct_change()
    df["vol_20d"] = daily_ret.rolling(20).std()
    df["vol_50d"] = daily_ret.rolling(50).std()

    # --- Momentum features ---
    df["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line = _macd(close)
    df["macd_hist"] = macd_line - signal_line

    # --- Fear features (VIX) ---
    vix_close = vix["Close"].reindex(df.index).ffill()
    df["vix_level"] = vix_close
    df["vix_change_5d"] = vix_close.pct_change(5)

    # --- Volume features ---
    if "Volume" in spy.columns:
        vol = spy["Volume"]
        df["rel_volume"] = vol / vol.rolling(20).mean()

    # --- Label: forward 20-day return -> regime ---
    fwd_return = close.shift(-20) / close - 1
    df["future_return_20d"] = fwd_return
    df["regime"] = np.select(
        [fwd_return > 0.05, fwd_return < -0.05],
        ["Bull", "Bear"],
        default="Sideways",
    )
    # Optional: flag high-volatility periods regardless of direction
    high_vol_threshold = df["vol_20d"].quantile(0.85)
    df.loc[df["vol_20d"] > high_vol_threshold, "regime"] = "High_Volatility"

    df = df.dropna()
    return df


def save_features(df: pd.DataFrame, filename: str = "features.csv") -> Path:
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    from data_loader import load_all

    data = load_all()
    features = build_features(data["SPY"], data["VIX"])
    path = save_features(features)
    print(f"Saved {features.shape[0]} rows x {features.shape[1]} cols -> {path}")
    print(features["regime"].value_counts())