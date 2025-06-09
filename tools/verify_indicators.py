# tools/verify_indicators.py

import os
import sys
import pandas as pd
from datetime import datetime
import pytz

# --- FIX 1: Add the project root to the Python path ---
# This allows the script to find the 'analytics' and 'core' modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import necessary components from the existing framework
from analytics.profiles import get_session, VolumeProfiler, MarketProfiler
from analytics.indicators import calculate_vwap

def load_data_for_ticker(data_dir: str, ticker: str) -> pd.DataFrame:
    """Loads and prepares historical data for a single ticker."""
    filepath = os.path.join(data_dir, f'{ticker}.csv')
    if not os.path.exists(filepath):
        print(f"ERROR: Data file not found at {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        # Standardize column names
        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True, errors='ignore')

        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None

def analyze_day(day_df: pd.DataFrame, timezone_str: str = 'America/New_York', tick_size: float = 0.01):
    """
    Analyzes a single day's data, breaking it down by session and
    calculating indicators for each.
    """
    if day_df.empty:
        return

    date_str = day_df.index[0].date().strftime('%Y-%m-%d')
    print(f"\n--- Analysis for {date_str} ---")

    timezone = pytz.timezone(timezone_str)
    day_df.index = day_df.index.tz_convert(timezone)

    # Define the sessions to analyze
    sessions = ['Pre-Market', 'Regular', 'After-Hours']

    for session_name in sessions:
        # Filter the day's data for the specific session
        session_df = day_df[day_df.index.to_series().apply(
            lambda dt: get_session(dt) == session_name
        )]

        print(f"\n  -- {session_name} Session --")

        if session_df.empty:
            print("     No data for this session.")
            continue

        # --- Calculate Indicators for the session ---
        # 1. VWAP
        vwap_df = calculate_vwap(session_df.copy())
        final_vwap = vwap_df['vwap'].iloc[-1] if not vwap_df.empty else 'N/A'

        # 2. Volume Profile
        vol_profiler = VolumeProfiler(session_df, tick_size=tick_size)

        # 3. Market Profile (TPO)
        mkt_profiler = MarketProfiler(session_df, tick_size=tick_size)

        # --- Print Results ---
        print(f"     Time Range : {session_df.index[0].time()} - {session_df.index[-1].time()}")
        print(f"     Ending VWAP: {final_vwap:.2f}" if isinstance(final_vwap, float) else f"     Ending VWAP: {final_vwap}")

        print("     Volume Profile:")
        if vol_profiler.poc_price is not None:
            print(f"       - POC: {vol_profiler.poc_price:.2f}")
            print(f"       - VAH: {vol_profiler.vah:.2f}")
            print(f"       - VAL: {vol_profiler.val:.2f}")
        else:
            print("       - Not enough data to calculate.")

        print("     Market Profile (TPO):")
        if mkt_profiler.poc_price is not None:
            print(f"       - POC: {mkt_profiler.poc_price:.2f}")
            print(f"       - VAH: {mkt_profiler.vah:.2f}")
            print(f"       - VAL: {mkt_profiler.val:.2f}")
        else:
            print("       - Not enough data to calculate.")


def main():
    """Main function to drive the indicator verification script."""
    print("--- Indicator Verification Tool ---")

    # --- FIX 2: Build the path to the data directory from the project root ---
    DATA_DIR = os.path.join(project_root, 'data', 'historical')
    TICK_SIZE = 0.01 # Default tick size for profiles

    # --- User Input ---
    ticker = input("Enter the ticker symbol to analyze (e.g., PLTR): ").upper()
    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    # --- Load Data ---
    full_data = load_data_for_ticker(DATA_DIR, ticker)
    if full_data is None:
        return

    # --- Process each day in the date range ---
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for trade_date in date_range:
        day_data = full_data[full_data.index.date == trade_date.date()]
        if not day_data.empty:
            analyze_day(day_data, tick_size=TICK_SIZE)

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    main()
