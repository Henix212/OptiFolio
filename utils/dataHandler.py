import yfinance as yf
import pandas as pd
import os

# List of market indices to download
tickers_list = ["^NDX", "^FCHI", "^GDAXI", "^N225", "^HSI", "^SSMI"]

def dl_market_data(tickers_list: list) -> None:
    """
    Download historical market data from Yahoo Finance
    and save it as a raw CSV file.
    """
    df = yf.download(tickers_list, period='max')
    df.to_csv("data/raw/yahoo/yfin_data.csv")


def clean_data(path):
    """
    Load a CSV file and drop missing values.
    """
    df = pd.read_csv(path)
    return df.dropna()


def ensure_clean_dirs():
    """
    Create required raw and cleaned data directories if they do not exist.
    """
    dirs = [
        "data/raw/yahoo/",
        "data/cleaned/volumes",
        "data/cleaned/returns",
        "data/cleaned/prices",
        "data/cleaned/macro"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_yf_csv(path: str) -> pd.DataFrame:
    """
    Load a Yahoo Finance CSV file.
    First try reading a two-level header (typical yfinance format).
    If it fails, fall back to a standard single-header CSV.
    """
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        return df
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df


def _save_with_date(df: pd.DataFrame, path: str) -> None:
    """
    Reset the index to a 'Date' column and save the DataFrame to CSV.
    """
    out = df.reset_index()
    first_col = out.columns[0]

    if 'date' not in str(first_col).lower():
        out.rename(columns={first_col: 'Date'}, inplace=True)

    out.to_csv(path, index=False)


def prices_data(path: str) -> None:
    """
    Extract closing prices from raw Yahoo Finance data
    and save them to a cleaned CSV file.
    """
    df = load_yf_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df.loc[:, df.columns.str.contains('Close', case=False)]

    _save_with_date(prices, 'data/cleaned/prices/prices.csv')


def returns_data(path: str) -> None:
    """
    Compute daily returns from closing prices
    and save them to a cleaned CSV file.
    """
    df = load_yf_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df.loc[:, df.columns.str.contains('Close', case=False)]

    rets = prices.pct_change().dropna()
    _save_with_date(rets, 'data/cleaned/returns/returns.csv')


def volumes_data(path: str) -> None:
    """
    Extract trading volumes from raw Yahoo Finance data
    and save them to a cleaned CSV file.
    """
    df = load_yf_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        vols = df['Volume']
    else:
        vols = df.loc[:, df.columns.str.contains('Volume', case=False)]

    _save_with_date(vols, 'data/cleaned/prices/volumes.csv')


def main(tickers_list) -> None:
    """
    Main pipeline:
    - Ensure directories exist
    - Download data if missing
    - Generate cleaned prices, returns, and volumes
    """
    ensure_clean_dirs()

    if not os.path.isfile("data/raw/yahoo/yfin_data.csv"):
        dl_market_data(tickers_list)

    volumes_data(path="data/raw/yahoo/yfin_data.csv")
    prices_data(path="data/raw/yahoo/yfin_data.csv")
    returns_data(path="data/raw/yahoo/yfin_data.csv")


if __name__ == '__main__':
    main(tickers_list)
