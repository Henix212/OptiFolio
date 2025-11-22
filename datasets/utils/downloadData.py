import yfinance as yf
import os
import datetime
import pandas as pd

output_dir = "datasets" 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir,"marketData"), exist_ok=True)

tickers_list = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","SAFRY","AIR","GD"]
end_date = datetime.date.today().strftime('%Y-%m-%d')
start_date = datetime.date(2010,1,1).strftime('%Y-%m-%d')

print("\n--- Download ---")

all_data = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True)

adj_close_data = all_data['Close']

volume_data = all_data['Volume']

adj_close_data = adj_close_data.dropna(how='all').ffill()
volume_data = volume_data.dropna(how='all').ffill()

close_filepath = os.path.join(output_dir,"marketData","adj_close_prices.csv")
volume_filepath = os.path.join(output_dir,"marketData","volume.csv")

adj_close_data.to_csv(close_filepath)
volume_data.to_csv(volume_filepath)

print(f"Data 'Adj Close' saved in {close_filepath}")
print(f"Data 'Volume' saved in {volume_filepath}")