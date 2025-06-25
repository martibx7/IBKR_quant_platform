# tools/preprocess_data.py

import pandas as pd
import os
import yaml
from tqdm import tqdm


def main():
    """
    Preprocess CSV price data into Feather format for fast read/write.
    For each CSV in the backtest data directory, this script will:
      1. Read the CSV (parsing 'date' to datetime).
      2. Standardize columns (lowercase, rename 'date' to 'timestamp').
      3. Create 'date' column as a normalized datetime object and add 'symbol'.
      4. Reorder to ['timestamp','date','symbol','open','high','low','close','volume'].
      5. Write out a corresponding .feather file in data/feather/.

    Can be run from either the project root or the tools folder.
    """
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    config_path = os.path.join(project_root, 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(project_root, config['backtest']['data_dir'])
    feather_dir = os.path.join(project_root, config['backtest'].get('feather_dir', 'data/feather'))
    os.makedirs(feather_dir, exist_ok=True)

    all_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    if not all_files:
        print(f"No CSV files found in '{data_dir}'")
        return

    print(f"Processing {len(all_files)} CSV files to Feather format...")
    for filename in tqdm(all_files, desc='Preprocessing'):
        csv_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])

            df.columns = [col.strip().lower() for col in df.columns]
            df.rename(columns={'date': 'timestamp'}, inplace=True)

            # --- FIX: Store date as a datetime object, not a string ---
            df['date'] = df['timestamp'].dt.normalize()

            symbol = os.path.splitext(filename)[0]
            df['symbol'] = symbol

            df = df[['timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            out_path = os.path.join(feather_dir, f"{symbol}.feather")
            df.to_feather(out_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nAll files have been preprocessed into Feather format.")


if __name__ == '__main__':
    main()