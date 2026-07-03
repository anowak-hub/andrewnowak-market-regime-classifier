"""
data_loader.py
---------------
Downloads market data (SPY, QQQ, VIX by default) via yfinance and
caches it to data/raw/ as CSV so repeated runs don't hit the network.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf 

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX"
}

def download_ticker(ticker: str, start: str = "1993-01-01", end: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download a single ticker's OHLCV data, using a local CSV cache
    in data/raw/ when available.
    """

    safe_name = ticker.replace("^", "")
    cache_path = RAW_DIR / f"{safe_name}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.to_csv(cache_path)
    return df

def load_all(tickers: dict[str, str] = None, start: str = "1993-01-01", end: str | None = None, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download/load every ticker in `tickers` (default: SPY, QQQ, VIX).
    Returns a dict of {name: DataFrame}.
    """
    tickers = tickers or DEFAULT_TICKERS
    data = {}
    for name, symbol in tickers.items():
        print(f"Loading {name} ({symbol})...")
        data[name] = download_ticker(symbol, start=start, end=end, force_refresh=force_refresh)
    return data


if __name__ == "__main__":
    all_data = load_all()
    for name, df in all_data.items():
        print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} cols")
        print(df.tail(3))