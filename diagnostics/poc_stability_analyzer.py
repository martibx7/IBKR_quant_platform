import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

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

# --- REMOVED: Database connection logic is no longer needed ---
# def get_db_engine(config: dict): ...

# --- NEW: Function to load data from Feather files ---
def load_data_from_feather(symbol: str, feather_dir: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    """
    Loads and filters historical data for a symbol from a .feather file.
    """
    feather_path = feather_dir / f"{symbol}.feather"
    if not feather_path.exists():
        logging.warning(f"Data file not found for symbol {symbol} at {feather_path}, skipping.")
        return pd.DataFrame()

    df = pd.read_feather(feather_path)

    # Ensure timestamp column is a datetime type and set it as the index
    if 'timestamp' not in df.columns:
        raise ValueError(f"'timestamp' column not found in {feather_path}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    if df.empty:
        return pd.DataFrame()

    # Determine the full date range needed, including the lookback period
    query_start_date = start_date - pd.Timedelta(days=lookback_days * 2) if start_date else None
    query_end_date = end_date

    # Filter the DataFrame by date range, similar to the old SQL query
    mask = pd.Series(True, index=df.index)
    if query_start_date:
        mask &= (df.index.normalize() >= query_start_date.normalize())
    if query_end_date:
        mask &= (df.index.normalize() <= query_end_date.normalize())

    return df.loc[mask]


# --- OPTIMIZED: New analysis function (Unchanged) ---
def analyze_symbol_stability_fast(df: pd.DataFrame, lookback_days: int, profiler, symbol: str) -> list:
    """
    Efficiently calculates daily POC dispersion using a rolling window approach.
    """
    if df.empty:
        return []

    # 1. Pre-calculate the POC for every single day
    daily_pocs_data = []
    # Group by the date part of the DatetimeIndex
    for date, day_df in df.groupby(df.index.date):
        if day_df.empty:
            continue
        stats = profiler.calculate(day_df)
        if stats and 'poc_price' in stats:
            # Use pd.to_datetime to ensure the date is a timestamp for the DataFrame index
            daily_pocs_data.append({'date': pd.to_datetime(date), 'poc': stats['poc_price']})

    if not daily_pocs_data:
        return []

    poc_df = pd.DataFrame(daily_pocs_data).set_index('date')

    # 2. Use pandas' highly optimized rolling functions
    poc_df['rolling_std'] = poc_df['poc'].rolling(window=lookback_days, min_periods=lookback_days).std()
    poc_df['rolling_mean'] = poc_df['poc'].rolling(window=lookback_days, min_periods=lookback_days).mean()

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
        # --- MODIFIED: Localize input dates to UTC to match the data's timezone ---
        if start_date_str:
            # Convert string to a naive timestamp, then localize to UTC
            start_date = pd.to_datetime(start_date_str).tz_localize('UTC')
        else:
            start_date = None

        if end_date_str:
            # Do the same for the end date
            end_date = pd.to_datetime(end_date_str).tz_localize('UTC')
        else:
            end_date = None
        # --- END MODIFICATION ---

    except Exception as e:
        logging.error(f"Invalid date format: {e}. Please use YYYY-MM-DD.")
        return

    # Load config and get feather directory path
    config_path = Path(project_root) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    feather_dir = Path(project_root) / config['backtest']['feather_dir']
    logging.info(f"Using data from Feather directory: '{feather_dir}'")

    symbol_file = Path(project_root) / args.symbols
    with open(symbol_file, 'r') as f:
        symbols = {line.strip() for line in f if line.strip() and not line.startswith('#')}

    profile_type = args.profile_type.lower()
    tick_size = config['backtest'].get('tick_size_volume_profile', 0.01) if profile_type == 'volume' else config['backtest'].get('tick_size_market_profile', 0.01)

    profiler = VolumeProfiler(tick_size=tick_size) if profile_type == 'volume' else MarketProfiler(tick_size=tick_size)
    logging.info(f"Using {profile_type} profiler with a {args.lookback} day lookback.")

    all_results = []

    for symbol in tqdm(symbols, desc="Analyzing Symbols"):
        try:
            # Replaced SQL query with Feather file loading function
            df = load_data_from_feather(symbol, feather_dir, start_date, end_date, args.lookback)

            if df.empty:
                continue

            # Call the analysis function (logic is unchanged)
            symbol_results = analyze_symbol_stability_fast(df, args.lookback, profiler, symbol)

            # Filter results to only include dates within the user's requested range
            if start_date:
                # Compare the date part of the result with the date part of the start_date
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
    # Config argument is no longer needed as the path is now standardized
    # parser.add_argument('--config', type=str, default='config.yaml', help="Path to the main configuration file.")

    args = parser.parse_args()
    main(args)