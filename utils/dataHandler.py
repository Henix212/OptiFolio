import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import os

from featuresHandler import calculate_and_save_indicators 

os.makedirs("raw_data", exist_ok=True)
os.makedirs("data/returns", exist_ok=True)
os.makedirs("data/volatility", exist_ok=True)
os.makedirs("data/macro", exist_ok=True)
os.makedirs("data/indicators", exist_ok=True) 

start_date = dt.date.today() - dt.timedelta(days=365*10)
end_date = dt.date.today()

commodities_stock = ["GLD", "USO"]
equity_index_stock = ["URTH"] 
combined_stock_list = list(set(commodities_stock + equity_index_stock))

raw_data = yf.download(combined_stock_list, start=start_date, end=end_date, group_by='ticker', progress=False)

training_days = pd.date_range(start=start_date, end=end_date, freq='B')
raw_data = raw_data.reindex(training_days, method='ffill').dropna()

gspc_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False).reindex(raw_data.index, method='ffill')
vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False).reindex(raw_data.index, method='ffill')

close_prices = pd.DataFrame()
for ticker in combined_stock_list:
    close_prices[ticker] = raw_data[ticker]['Close']

returns = close_prices.pct_change().dropna()

raw_data = raw_data.loc[returns.index] 

volatility = returns.std() * np.sqrt(252)
sorted_volatility = volatility.sort_values(ascending=False)
print("\nAnnual Volatility :")
print(sorted_volatility)

calculate_and_save_indicators(raw_data, combined_stock_list)

raw_data.to_csv("raw_data/raw_data.csv")
returns.to_csv("data/returns/data_returns.csv")
volatility.to_csv("data/volatility/data_volatility.csv")
gspc_df.to_csv("data/macro/gspc.csv")
vix_df.to_csv("data/macro/vix.csv")