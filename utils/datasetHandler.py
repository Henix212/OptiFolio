import os
import pandas as pd

def create_dataset(path):
    all_dfs = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                prefix = os.path.splitext(file)[0] 
                df = df.add_prefix(f"{prefix}_")

                all_dfs.append(df)
    
    if not all_dfs:
        return "Aucun fichier trouvé."

    combined_df = pd.concat(all_dfs, axis=1, join='inner')
    combined_df.sort_index(inplace=True)

    output_dir = "data/dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")
    
    combined_df.to_csv(output_path)
    
    return f"Dataset synchronisé créé : {output_path} ({len(combined_df)} lignes communes)"