import pandas as pd
import numpy as np
import os

lambda_short = 0.94   
lambda_long = 0.97    
annualization = np.sqrt(252)

def ensure_clean_dirs():
    """Create expected cleaned subfolders if they don't exist."""
    dirs = [
        "data/features/",
        "data/features/volatility",
        "data/features/volatility/",
        "data/features/returns/",
        "data/features/regime"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def ewma(df,decay):
    
    vol = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    vol.iloc[0] = df.iloc[:20].var() ** 0.5

    for t in range(1, len(df)):
        vol.iloc[t] = np.sqrt(
            decay * (vol.iloc[t-1] ** 2) + (1 - decay) * (df.iloc[t-1] ** 2)
        )

    return vol * annualization

def vol_zscore(df,decay):
    mu = ewma(df=df,decay=decay)
    sigma = ewma(df=(df-mu)**2,decay=decay)

    zscore = (df - mu) / sigma

    return zscore

def avrg_coor(df):
    pass

def 

if __name__ == '__main__':
    ensure_clean_dirs()

    returns = pd.read_csv("data/cleaned/returns/returns.csv", parse_dates=['Date'], index_col='Date')

    short_vol = ewma(df=returns,decay= lambda_short)
    long_vol = ewma(df=returns, decay= lambda_long)
    
    vol_ratio = short_vol / long_vol

    norm_return = returns / short_vol

    short_vol_zscore = vol_zscore(short_vol,0.97)

    avrg_coor(returns)

    short_vol.to_csv("data/features/volatility/short_ratio.csv")
    long_vol.to_csv("data/features/volatility/long_ratio.csv")
    vol_ratio.to_csv("data/features/volatility/vol_ratio.csv")
    short_vol_zscore.to_csv("data/features/volatility/zscore.csv")
    norm_return.to_csv("data/features/returns/norm_returns.csv")
   