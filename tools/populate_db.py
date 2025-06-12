# tools/populate_db.py

import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine, text
from tqdm import tqdm
import yaml

def get_data_dir_from_config(project_root: str) -> str:
    config_path = os.path.join(project_root, 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('backtest', {}).get('data_dir', 'data/historical')
    except FileNotFoundError:
        print(f"Warning: config.yaml not found at '{config_path}'. Using default 'data/historical'.")
        return 'data/historical'

def main():
    print("--- Database Population Tool ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root identified as: {project_root}")

    relative_data_dir = get_data_dir_from_config(project_root)
    data_dir = os.path.join(project_root, relative_data_dir)
    db_path = os.path.join(project_root, 'data', 'price_data.db')

    # --- NEW: Delete existing database to ensure a clean build ---
    if os.path.exists(db_path):
        print(f"Deleting existing database at {db_path} to rebuild.")
        os.remove(db_path)

    print(f"Data directory: {data_dir}")
    print(f"Database will be created at: {db_path}")

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        return

    engine = create_engine(f'sqlite:///{db_path}')
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found to process.")
        return

    print(f"Found {len(all_files)} CSV files to import.")

    for filename in tqdm(all_files, desc="Populating Database"):
        symbol = os.path.splitext(filename)[0]
        file_path = os.path.join(data_dir, filename)

        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df.columns = [col.strip().lower() for col in df.columns]
            df = df.rename(columns={'date': 'timestamp'})
            df['symbol'] = symbol

            # --- NEW: Remove duplicate timestamps before inserting ---
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df.reset_index(inplace=True)
            # --- END FIX ---

            df.to_sql(
                'price_data',
                con=engine,
                if_exists='append',
                index=False
            )
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("Creating index for faster queries... (This may take a moment)")
    with engine.connect() as connection:
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON price_data (symbol, timestamp);"))
        connection.commit()

    print("\n--- Database population complete! ---")

if __name__ == '__main__':
    main()