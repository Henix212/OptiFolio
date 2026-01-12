import pandas as pd
import os

data_dir = "data/"

df = None

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file == 'data_volatility.csv':
                continue
            else:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.endswith(".csv"):
                    temp_df = pd.read_csv(file_path)
                    if df is None:
                        df = temp_df
                    else:
                        df = df.merge(temp_df, on="Date", how="inner", suffixes=('', f'_{folder}'))

df.to_csv("data/dataset/dataset.csv")
