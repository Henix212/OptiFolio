import os
import pandas as pd

def create_dataset(path):
    """
    Combine all CSV files in a folder (and subfolders) into a single dataset.
    Each column is prefixed by the filename to avoid collisions.
    Only keeps rows with common dates across all files.
    
    Args:
        path (str): Folder containing CSV files
    
    Returns:
        str: Message indicating success or no files found
    """
    all_dfs = []
    
    # Walk through the folder and subfolders
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                # Convert 'Date' column to datetime and set as index
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Prefix column names with filename (without extension)
                prefix = os.path.splitext(file)[0] 
                df = df.add_prefix(f"{prefix}_")

                all_dfs.append(df)
    
    if not all_dfs:
        return "No CSV files found."

    # Concatenate all dataframes on common dates (inner join)
    combined_df = pd.concat(all_dfs, axis=1, join='inner')
    combined_df.sort_index(inplace=True)

    # Ensure output directory exists
    output_dir = "data/dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")
    
    # Save combined dataset to CSV
    combined_df.to_csv(output_path)
    
    return f"Synchronized dataset created: {output_path} ({len(combined_df)} common rows)"

if __name__ == '__main__':
    create_dataset("data/features")