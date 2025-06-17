# tools/populate_db.py

import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine, text
from tqdm import tqdm
import yaml

def get_data_dir_from_config(project_root: str) -> str:
    """
    Reads the data directory path from the project's config file.
    """
    config_path = os.path.join(project_root, 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Navigate through the config structure to find the data_dir
        return config.get('backtest', {}).get('data_dir', 'data/historical')
    except FileNotFoundError:
        print(f"Warning: config.yaml not found at '{config_path}'. Using default 'data/historical'.")
        return 'data/historical'

def main():
    """
    Main function to orchestrate the database population process.
    """
    print("--- Database Population Tool ---")

    # Determine project root assuming this script is in the 'tools' folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root identified as: {project_root}")

    # Get data directory from config and set database path
    relative_data_dir = get_data_dir_from_config(project_root)
    data_dir = os.path.join(project_root, relative_data_dir)
    db_path = os.path.join(project_root, 'data', 'price_data.db')

    # Delete the existing database to ensure a clean, optimized build
    if os.path.exists(db_path):
        print(f"Deleting existing database at {db_path} to rebuild with new schema.")
        os.remove(db_path)

    print(f"Data directory: {data_dir}")
    print(f"Database will be created at: {db_path}")

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        return

    # Set up the database engine
    engine = create_engine(f'sqlite:///{db_path}')
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not all_files:
        print("No CSV files found to process.")
        return

    print(f"Found {len(all_files)} CSV files to import.")

    # Process each CSV file and load it into the database
    for filename in tqdm(all_files, desc="Populating Database"):
        symbol = os.path.splitext(filename)[0]
        file_path = os.path.join(data_dir, filename)

        try:
            # Load and standardize the data
            df = pd.read_csv(file_path, parse_dates=['date'])
            df.columns = [col.strip().lower() for col in df.columns]
            df = df.rename(columns={'date': 'timestamp'})

            # --- PERFORMANCE OPTIMIZATION ---
            # Create a dedicated 'date' column in text format ('YYYY-MM-DD')
            # This is the key change for dramatically faster daily lookups.
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            # --- END OPTIMIZATION ---

            df['symbol'] = symbol

            # Remove duplicate timestamps for data integrity
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df.reset_index(inplace=True)

            # Ensure a consistent column order
            df = df[['timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            # Append the processed data to the SQL table
            df.to_sql(
                'price_data',
                con=engine,
                if_exists='append',
                index=False
            )
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("Creating optimized index for faster queries... (This may take a moment)")
    with engine.connect() as connection:
        # Create a new, more efficient index on the date and symbol columns
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_date_symbol ON price_data (date, symbol);"))
        connection.commit()

    print("\n--- Database population complete! ---")
    print("IMPORTANT: To realize the speed benefits, you must now update your backtest engine's query to use this new 'date' column for lookups.")

if __name__ == '__main__':
    main()