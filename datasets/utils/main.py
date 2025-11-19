import yfinance as yf
import os
import datetime
import pandas as pd

output_dir = "datasets" 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir,"marketData"), exist_ok=True)

tickers_list = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","AVGO","JPM","V","WMT","HD","MC","OR","SIE","JNJ","SAFRY","AIR","BA","GD"]
start_date = "2010-01-01"
end_date = datetime.date.today().strftime('%Y-%m-%d')

print("\n--- Téléchargement ---")

all_data = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True)

adj_close_data = all_data['Close']

volume_data = all_data['Volume']

adj_close_data = adj_close_data.dropna(how='all').ffill()
volume_data = volume_data.dropna(how='all').ffill()

close_filepath = os.path.join(output_dir,"marketData","adj_close_prices.csv")
volume_filepath = os.path.join(output_dir,"marketData","volume.csv")

adj_close_data.to_csv(close_filepath)
volume_data.to_csv(volume_filepath)

print(f"✅ Données 'Adj Close' enregistrées dans {close_filepath}")
print(f"✅ Données 'Volume' enregistrées dans {volume_filepath}")

print("\n--- Aperçu des données de clôture (adj_close_prices.csv) ---")
print(adj_close_data.tail()) 