# tools/populate_db.py

import pandas as pd
import os
import sqlalchemy
from sqlalchemy import create_engine, text
from tqdm import tqdm
import yaml

# You can tune this number based on your system's RAM.
# Higher numbers are faster but use more memory. 100 is a safe starting point.
CHUNK_SIZE = 500

def get_db_engine(config: dict):
    """Creates a SQLAlchemy engine from the config file."""
    db_config = config['database']
    db_type = db_config['type']

    if db_type == 'postgresql':
        conn_str = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        return create_engine(conn_str)
    elif db_type == 'sqlite':
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, db_config.get('db_path', 'data/price_data.db'))
        return create_engine(f"sqlite:///{db_path}")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def process_and_write_chunk(df_list: list, engine):
    """
    Concatenates a list of dataframes and writes the chunk to the database,
    using a safe chunksize depending on the database dialect.
    """
    if not df_list:
        return

    # Combine all dataframes in the chunk
    master_chunk_df = pd.concat(df_list, ignore_index=True)

    # Drop true duplicates based on both timestamp and symbol
    master_chunk_df = master_chunk_df.drop_duplicates(
        subset=['timestamp', 'symbol'], keep='first'
    )

    # Ensure the desired column order
    master_chunk_df = master_chunk_df[
        ['timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    ]

    # Use larger chunk sizes for PostgreSQL, smaller for SQLite
    db_chunk_size = 5000 if engine.dialect.name == 'postgresql' else 120

    master_chunk_df.to_sql(
        'price_data',
        con=engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=db_chunk_size
    )

def main():
    """
    Main function to orchestrate the database population process.
    Uses chunking to balance memory usage and speed.
    """
    print("--- Database Population Tool (Chunk-Optimized) ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(project_root, config['backtest']['data_dir'])
    engine = get_db_engine(config)

    print(f"Connecting to {config['database']['type']} database...")

    with engine.connect() as connection:
        print("Dropping existing 'price_data' table for a clean import...")
        connection.execute(text("DROP TABLE IF EXISTS price_data;"))
        connection.commit()

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found to process.")
        return

    print(f"Found {len(all_files)} CSV files. Processing in chunks of {CHUNK_SIZE}...")

    chunk_dfs = []
    for filename in tqdm(all_files, desc="Processing files"):
        symbol = os.path.splitext(filename)[0]
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df.columns = [col.strip().lower() for col in df.columns]
            df = df.rename(columns={'date': 'timestamp'})
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            chunk_dfs.append(df)

            if len(chunk_dfs) >= CHUNK_SIZE:
                print(f"Writing chunk of {len(chunk_dfs)} files to database...")
                process_and_write_chunk(chunk_dfs, engine)
                print("Chunk write complete.")
                chunk_dfs.clear()

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    # Write any remaining frames
    if chunk_dfs:
        print(f"Writing final chunk of {len(chunk_dfs)} files to database...")
        process_and_write_chunk(chunk_dfs, engine)
        print("Final chunk write complete.")

    print("\nCreating optimized index for faster queries...")
    with engine.connect() as connection:
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_date_symbol ON price_data (date, symbol);"
        ))
        connection.commit()
    print("Index creation complete.")

    # For PostgreSQL, VACUUM must run outside a transaction block
    if engine.dialect.name == 'postgresql':
        print("Optimizing table statistics for PostgreSQL (VACUUM ANALYZE)...")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text("VACUUM ANALYZE price_data;"))
        print("VACUUM ANALYZE complete.")

    print("\n--- Database population complete! ---")

if __name__ == '__main__':
    main()
