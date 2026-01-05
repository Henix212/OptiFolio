import yfinance as yf
import pandas as pd
import os

tickers_list = ["^NDX","^FCHI","^GDAXI","^N225","^HSI","^SSMI"]

def dl_market_data(tickers_list : list) -> None:

    df = yf.download(tickers_list, period='5y')

    df.to_csv("data/raw/yahoo/yfin_data.csv")

def clean_data(path):
    df = pd.read_csv(path)

    return df.dropna()

def ensure_clean_dirs():
    """Create expected cleaned subfolders if they don't exist."""
    dirs = [
        "data/raw/yahoo/"
        "data/cleaned/volumes",
        "data/cleaned/returns",
        "data/cleaned/prices",
        "data/cleaned/volatility",
        "data/cleaned/macro",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_yf_csv(path: str) -> pd.DataFrame:
    """Load raw yfinance CSV. Try reading the two-row header produced by yfinance
    (header=[0,1]). Fall back to a simple single-header read if needed."""
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        return df
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df


def _save_with_date(df: pd.DataFrame, path: str) -> None:
    """Reset index to a column named 'Date' and save without index."""
    out = df.reset_index()
    first_col = out.columns[0]
    if 'date' not in str(first_col).lower():
        out.rename(columns={first_col: 'Date'}, inplace=True)
    out.to_csv(path, index=False)


def prices_data(path: str) -> None:
    df = load_yf_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df.loc[:, df.columns.str.contains('Close', case=False)]
    _save_with_date(prices, 'data/cleaned/prices/prices.csv')


def returns_data(path: str) -> None:
    df = load_yf_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df.loc[:, df.columns.str.contains('Close', case=False)]
    rets = prices.pct_change().dropna()
    _save_with_date(rets, 'data/cleaned/returns/returns.csv')


def volumes_data(path: str) -> None:
    df = load_yf_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        vols = df['Volume']
    else:
        vols = df.loc[:, df.columns.str.contains('Volume', case=False)]
    _save_with_date(vols, 'data/cleaned/volumes/volumes.csv')


def volatility_data(path: str, window: int = 21) -> None:
    """Compute rolling volatility (annualized) from returns and save it."""
    df = load_yf_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df.loc[:, df.columns.str.contains('Close', case=False)]
    rets = prices.pct_change().dropna()
    vol = rets.rolling(window).std() * (252 ** 0.5)
    _save_with_date(vol, 'data/cleaned/volatility/volatility.csv')


def main(tickers_list) -> None:
    ensure_clean_dirs()

    if not os.path.isfile("data/raw/yahoo/yfin_data.csv"):
        dl_market_data(tickers_list)
        
    volumes_data(path="data/raw/yahoo/yfin_data.csv")
    prices_data(path="data/raw/yahoo/yfin_data.csv")
    returns_data(path="data/raw/yahoo/yfin_data.csv")
    volatility_data(path="data/raw/yahoo/yfin_data.csv")


if __name__ == '__main__':
    main(tickers_list)