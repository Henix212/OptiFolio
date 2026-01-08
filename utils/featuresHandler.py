import pandas as pd
import numpy as np
import os

# Decay factors for short-term and long-term EWMA
lambda_short = 0.94   
lambda_long = 0.97    
annualization = np.sqrt(252)  # Annualization factor for daily returns

def ensure_clean_dirs():
    """Create expected cleaned subfolders if they don't exist."""
    dirs = [
        "data/features/",
        "data/features/volatility",
        "data/features/volatility/",
        "data/features/returns/",
        "data/features/correlation"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def ewma(df, decay):
    """
    Compute Exponentially Weighted Moving Average (EWMA) volatility.
    
    Args:
        df (pd.DataFrame): Return series of assets
        decay (float): Decay factor (lambda)
    
    Returns:
        pd.DataFrame: EWMA volatility, annualized
    """
    vol = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    
    # Initialize first row with standard deviation of first 20 observations
    vol.iloc[0] = df.iloc[:20].var() ** 0.5

    # Compute EWMA iteratively
    for t in range(1, len(df)):
        vol.iloc[t] = np.sqrt(
            decay * (vol.iloc[t-1] ** 2) + (1 - decay) * (df.iloc[t-1] ** 2)
        )

    return vol * annualization

def ewma_std(df, decay, eps=1e-8):
    """
    Alternative EWMA standard deviation using pandas ewm.
    """
    mu = ewma(df, decay)
    var = ((df - mu) ** 2).ewm(alpha=(1 - decay), adjust=False).mean()
    return (var + eps) ** 0.5

def ewma_avg_corr(returns, decay=0.97):
    """
    Compute EWMA average correlation across assets.
    
    Args:
        returns (pd.DataFrame): Asset returns
        decay (float): Decay factor
    
    Returns:
        pd.DataFrame: Time series of average correlations
    """
    assets = returns.columns
    n = len(assets)
    cov_ewma = None
    avg_corr = []

    for t in range(len(returns)):
        r = returns.iloc[t].values.reshape(-1, 1)

        if cov_ewma is None:
            cov_ewma = r @ r.T
            avg_corr.append(np.nan)
            continue

        cov_ewma = decay * cov_ewma + (1 - decay) * (r @ r.T)
        std = np.sqrt(np.diag(cov_ewma))
        corr = cov_ewma / np.outer(std, std)

        # Take upper triangle values excluding diagonal
        upper = corr[np.triu_indices(n, k=1)]
        avg_corr.append(np.nanmean(upper))

    return pd.DataFrame(avg_corr, index=returns.index)

if __name__ == '__main__':
    # Ensure all feature directories exist
    ensure_clean_dirs()

    # Load cleaned returns dataset
    returns = pd.read_csv("data/cleaned/returns/returns.csv", parse_dates=['Date'], index_col='Date')

    # Compute short-term and long-term EWMA volatility
    short_vol = ewma(df=returns, decay=lambda_short)
    long_vol = ewma(df=returns, decay=lambda_long)
    
    # Volatility ratio
    vol_ratio = short_vol / long_vol

    # Normalize returns by short-term volatility
    norm_return = returns / short_vol

    # Compute short-term and long-term average correlations
    short_corr = ewma_avg_corr(returns, decay=lambda_short)
    long_corr  = ewma_avg_corr(returns, decay=lambda_long)

    # Correlation ratio
    corr_ratio = short_corr / long_corr

    # Save computed features to CSV
    short_vol.to_csv("data/features/volatility/short_ratio.csv")
    long_vol.to_csv("data/features/volatility/long_ratio.csv")
    vol_ratio.to_csv("data/features/volatility/vol_ratio.csv")
    norm_return.to_csv("data/features/returns/norm_returns.csv")
    short_corr.to_csv("data/features/correlation/short_corr.csv")
    long_corr.to_csv("data/features/correlation/long_corr.csv")
    corr_ratio.to_csv("data/features/correlation/corr_ratio.csv")
