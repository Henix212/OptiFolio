import talib as ta
import numpy as np
import pandas as pd
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
    