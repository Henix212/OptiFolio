import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    weighting_factor = 2 / (window + 1)
    ema = np.zeros(len(series))
    sma = np.mean(series[:window])
    ema[window - 1] = sma
    for i in range(window, len(series)):
        ema[i] = (series.iloc[i] * weighting_factor) + (ema[i-1] * (1 - weighting_factor))
    return ema

def relative_strong_index(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

    rs = gain / loss

    return 100 - (100 / (1+rs))

def moving_average_convergence_divergence(series: pd.Series) -> pd.Series:
    ema12 = exponential_moving_average(series=series,window=12)

    ema26 = exponential_moving_average(series=series,window=26)

    macd = ema12 - ema26

    macd_signal = exponential_moving_average(series=pd.Series(macd),window=9)

    return macd_signal

def standar_deviation(series: pd.Series, window : int):
    series.rolling(window=window).std()


df_close = pd.read_csv(
    "datasets/marketData/adj_close_prices.csv",
    index_col="Date",
    parse_dates=True
)

df_volume = pd.read_csv(
    "datasets/marketData/volume.csv",
    index_col="Date",
    parse_dates=True
)

for col in df_volume.columns:

    # === Adj_closes_prices Indicators === #

    sma_close = simple_moving_average(df_close[col], 20)
    ema_close = exponential_moving_average(df_close[col], 21)
    rsi_close = relative_strong_index(df_close[col], 14)
    macd_close = moving_average_convergence_divergence(df_close[col])

    # === Volume Indicators === #

    sma_volume = simple_moving_average(df_volume[col], 20)
    ema_volume = exponential_moving_average(df_volume[col], 21)
    std_volume = standar_deviation(df_volume[col], 20)

    result_df = pd.DataFrame({
        "Adj_Close": df_close[col],
        "Volume" : df_volume[col],
        "SMA_20": sma_close,
        "EMA_21": ema_close,
        "RSI_14": rsi_close,
        "MACD_9": macd_close,
        "SMA_20" : sma_volume,
        "EMA_21" : ema_volume,
        "STD_20" : std_volume
    })

    result_df.to_csv(f"./datasets/dataFrames/{col}.csv")