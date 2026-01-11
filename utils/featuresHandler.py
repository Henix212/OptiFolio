import talib as ta
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import os

def calculate_and_save_indicators(df, tickers, output_dir="data/indicators"):
    os.makedirs(output_dir, exist_ok=True)

    indicator_names = [
        'macd', 'rsi', 'cci', 'adx', 'stoch', 'willr', 
        'bb_upper', 'bb_middle', 'bb_lower', 
        'mfi', 'ema', 'atr', 'sar', 'obv'
    ]
    temp_data = {name: {} for name in indicator_names}

    is_multi = isinstance(df.columns, pd.MultiIndex)

    for ticker in tickers:
        if is_multi:
            if ticker not in df.columns.get_level_values(0): 
                continue
            sub_df = df[ticker]
        else:
            sub_df = df 

        sub_df = sub_df.ffill().bfill()

        close = sub_df['Close'].values.astype(float)
        high = sub_df['High'].values.astype(float)
        low = sub_df['Low'].values.astype(float)
        volume = sub_df['Volume'].values.astype(float)

        temp_data['macd'][ticker] = ta.MACD(close)[0]
        temp_data['rsi'][ticker] = ta.RSI(close)
        temp_data['cci'][ticker] = ta.CCI(high, low, close)
        temp_data['adx'][ticker] = ta.ADX(high, low, close)
        
        k, _ = ta.STOCH(high, low, close)
        temp_data['stoch'][ticker] = k
        
        temp_data['willr'][ticker] = ta.WILLR(high, low, close)
        
        u, m, l = ta.BBANDS(close)
        temp_data['bb_upper'][ticker] = u
        temp_data['bb_middle'][ticker] = m
        temp_data['bb_lower'][ticker] = l
        
        temp_data['ema'][ticker] = ta.EMA(close)
        temp_data['atr'][ticker] = ta.ATR(high, low, close)
        temp_data['sar'][ticker] = ta.SAR(high, low)
        temp_data['obv'][ticker] = ta.OBV(close, volume)
        temp_data['mfi'][ticker] = ta.MFI(high, low, close, volume)
        
    for name, data_dict in temp_data.items():
        if not data_dict:
            continue 
            
        indicator_df = pd.DataFrame(data_dict, index=df.index)
        
        file_path = f"{output_dir}/{name}.csv"
        indicator_df.to_csv(file_path)
    
    normalize_indicators("data/indicators",indicator_names)
    for macro in ["vix",'gspc']:
        df = pd.read_csv(f"data/macro/{macro}.csv")

        df = df[df.iloc[:, 0] != "Ticker"]
        close = pd.to_numeric(df["Close"], errors="coerce").values

        normalize_macro_data = normalize_macro(close)

        macro_df = pd.DataFrame(normalize_macro_data, index=df.index) 

        macro_df.to_csv(f"data/macro/{macro}.csv")
    
def handle_nans(indicators : pd.DataFrame, fill_value = 0):
    return indicators.fillna(value=fill_value)

def normalize_indicators(indicators_path, indicators_name):
    for name in indicators_name:
        df = pd.read_csv(
            f"{indicators_path}/{name}.csv",
            index_col=0
        ).astype(float)

        df = handle_nans(df)

        indicators = df.to_numpy()

        if indicators.ndim == 1:
            indicators = indicators.reshape(-1, 1)

        min_val = np.min(indicators, axis=0, keepdims=True)
        max_val = np.max(indicators, axis=0, keepdims=True)

        epsilon = 1e-8
        normalized = (indicators - min_val) / (max_val - min_val + epsilon)

        normalized_df = pd.DataFrame(
            normalized,
            index=df.index,
            columns=df.columns
        )

        normalized_df.to_csv(f"{indicators_path}/{name}.csv")

def normalize_macro(indicators):
    indicators = handle_macro_nans(indicators)  
    if indicators.ndim == 1:
        indicators = indicators.reshape(-1, 1) 

    min_val = np.min(indicators, axis=0, keepdims=True)
    max_val = np.max(indicators, axis=0, keepdims=True)

    epsilon = 1e-8
    normalized = (indicators - min_val) / (max_val - min_val + epsilon)
    
    return normalized

def handle_macro_nans(indicators, fill_value=0):
    inds_nan = np.isnan(indicators)
    if inds_nan.any():
        indicators = np.where(inds_nan, fill_value, indicators)
    return indicators