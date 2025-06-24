import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from sqlalchemy import create_engine, text

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from analytics.profiles import VolumeProfiler, MarketProfiler

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# --- DB Connection Logic ---
def get_db_engine(config: dict):
    db_config = config['database']
    db_type = db_config['type']
    if db_type == 'postgresql':
        conn_str = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        return create_engine(conn_str)
    elif db_type == 'sqlite':
        db_path = Path(project_root) / db_config.get('db_path', 'data/price_data.db')
        return create_engine(f"sqlite:///{db_path}")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

# --- OPTIMIZED: New analysis function ---
def analyze_symbol_stability_fast(df: pd.DataFrame, lookback_days: int, profiler, symbol: str) -> list:
    """
    Efficiently calculates daily POC dispersion using a rolling window approach.
    """
    if df.empty:
        return []

    # 1. Pre-calculate the POC for every single day
    daily_pocs_data = []
    for date, day_df in df.groupby(pd.Grouper(freq='D')):
        if day_df.empty:
            continue
        stats = profiler.calculate(day_df)
        if stats and 'poc_price' in stats:
            daily_pocs_data.append({'date': date, 'poc': stats['poc_price']})

    if not daily_pocs_data:
        return []

    poc_df = pd.DataFrame(daily_pocs_data).set_index('date')

    # 2. Use pandas' highly optimized rolling functions
    poc_df['rolling_std'] = poc_df['poc'].rolling(window=lookback_days).std()
    poc_df['rolling_mean'] = poc_df['poc'].rolling(window=lookback_days).mean()

    # 3. Calculate dispersion and format results
    poc_df['poc_dispersion'] = poc_df['rolling_std'] / poc_df['rolling_mean']

    # Drop rows where the rolling window was not yet full & FIX a copy warning
    final_df = poc_df.dropna().copy()
    final_df['symbol'] = symbol

    # Convert to the list of tuples format the main function expects
    return [(idx.date(), row['symbol'], row['poc_dispersion']) for idx, row in final_df.iterrows()]


def main(args):
    """Main function to run the analysis."""
    logging.info("Starting POC Stability Analysis...")

    start_date_str = input("Enter analysis start date (YYYY-MM-DD) or press Enter for all history: ")
    end_date_str = input("Enter analysis end date (YYYY-MM-DD) or press Enter for all history: ")

    try:
        start_date = pd.to_datetime(start_date_str) if start_date_str else None
        end_date = pd.to_datetime(end_date_str) if end_date_str else None
    except Exception as e:
        logging.error(f"Invalid date format: {e}. Please use YYYY-MM-DD.")
        return

    config_path = Path(project_root) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    symbol_file = Path(project_root) / args.symbols
    with open(symbol_file, 'r') as f:
        symbols = {line.strip() for line in f if line.strip() and not line.startswith('#')}

    profile_type = args.profile_type.lower()
    tick_size = config['backtest'].get('tick_size_volume_profile', 0.01) if profile_type == 'volume' else config['backtest'].get('tick_size_market_profile', 0.01)

    profiler = VolumeProfiler(tick_size=tick_size) if profile_type == 'volume' else MarketProfiler(tick_size=tick_size)
    logging.info(f"Using {profile_type} profiler with a {args.lookback} day lookback.")

    all_results = []
    engine = get_db_engine(config)

    for symbol in tqdm(symbols, desc="Analyzing Symbols"):
        try:
            # We fetch a slightly larger window to ensure the lookback period is valid
            query_start_date = start_date - pd.Timedelta(days=args.lookback * 2) if start_date else None
            query_end_date = end_date

            query_parts = ["SELECT timestamp, open, high, low, close, volume, date FROM price_data WHERE symbol = :symbol"]
            params = {'symbol': symbol}

            if query_start_date:
                query_parts.append("AND date >= :start_date")
                params['start_date'] = query_start_date.date()
            if query_end_date:
                query_parts.append("AND date <= :end_date")
                params['end_date'] = query_end_date.date()

            query = " ".join(query_parts)

            df = pd.read_sql(text(query), engine, params=params, index_col='timestamp', parse_dates=['timestamp'])

            if df.empty:
                continue

            # Call the new, faster analysis function
            symbol_results = analyze_symbol_stability_fast(df, args.lookback, profiler, symbol)

            # Filter results to only include dates within the user's requested range
            if start_date:
                symbol_results = [res for res in symbol_results if res[0] >= start_date.date()]

            all_results.extend(symbol_results)
        except Exception as e:
            logging.error(f"Failed to process {symbol}: {e}")

    if not all_results:
        logging.warning("No results generated. Check data path and symbol list.")
        return

    results_df = pd.DataFrame(all_results, columns=['date', 'symbol', 'poc_dispersion'])
    output_path = Path(project_root) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Analysis complete. Results saved to '{output_path}'")

    logging.info("--- Summary Statistics for POC Dispersion ---")
    print(results_df['poc_dispersion'].describe(percentiles=[.25, .50, .75, .90, .95, .99]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the historical stability of daily POCs for a list of symbols.")
    parser.add_argument('--symbols', type=str, required=True, help="Path to the symbol list file (e.g., strategies/sp_500.txt).")
    parser.add_argument('--profile_type', type=str, default='volume', choices=['volume', 'market'], help="The type of profile to use for POC calculation.")
    parser.add_argument('--lookback', type=int, default=10, help="The lookback period in days for the stability calculation.")
    parser.add_argument('--output', type=str, default='diagnostics/poc_dispersion_analysis.csv', help="Path to save the output CSV file.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the main configuration file.")

    args = parser.parse_args()
    main(args)