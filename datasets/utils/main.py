import yfinance as yf
import os
import datetime
import pandas as pd

output_dir_v2 = "datasets" 
os.makedirs(output_dir_v2, exist_ok=True)

tickers_list = ["SPY", "QQQ", "EFA", "EEM", "GLD", "TLT"]
start_date = "2010-01-01"
end_date = datetime.date.today().strftime('%Y-%m-%d')

print("\n--- Téléchargement (Style 2 : 1 Fichier Principal) ---")

all_data = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True)

adj_close_data = all_data['Close']

volume_data = all_data['Volume']

adj_close_data = adj_close_data.dropna(how='all').ffill()
volume_data = volume_data.dropna(how='all').ffill()

close_filepath = os.path.join(output_dir_v2, "adj_close_prices.csv")
volume_filepath = os.path.join(output_dir_v2, "volume.csv")

adj_close_data.to_csv(close_filepath)
volume_data.to_csv(volume_filepath)

print(f"✅ Données 'Adj Close' enregistrées dans {close_filepath}")
print(f"✅ Données 'Volume' enregistrées dans {volume_filepath}")

print("\n--- Aperçu des données de clôture (adj_close_prices.csv) ---")
print(adj_close_data.tail()) 