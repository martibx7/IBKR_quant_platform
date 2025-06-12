# tools/verify_indicators.py

import os
import sys
import pandas as pd
from datetime import datetime
import pytz

# Add the project root to the Python path to find modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import necessary components from the framework
from analytics.profiles import get_session, VolumeProfiler, MarketProfiler
from analytics.indicators import calculate_vwap, calculate_moving_averages, calculate_200_day_sma

def load_data_for_ticker(data_dir: str, ticker: str) -> pd.DataFrame:
    """Loads and prepares historical data for a single ticker."""
    filepath = os.path.join(data_dir, f'{ticker}.csv')
    if not os.path.exists(filepath):
        print(f"ERROR: Data file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        df.columns = [col.strip().lower() for col in df.columns]
        df = df.rename(columns={'date': 'timestamp'})
        df.set_index(pd.to_datetime(df['timestamp'], utc=True), inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df.apply(pd.to_numeric)
    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None

def analyze_day(day_df: pd.DataFrame, full_data_with_200d_sma: pd.DataFrame, timezone_str: str = 'America/New_York', tick_size: float = 0.01):
    """
    Analyzes a single day's data, calculating indicators for each session.
    """
    if day_df.empty:
        return

    date = day_df.index[0].date()
    print(f"\n--- Analysis for {date.strftime('%Y-%m-%d')} ---")

    timezone = pytz.timezone(timezone_str)
    day_df_ny = day_df.tz_convert(timezone)

    end_of_day_timestamp = day_df.index[-1]
    val_200d_sma = 'N/A'
    if end_of_day_timestamp in full_data_with_200d_sma.index:
        val_200d_sma = full_data_with_200d_sma.loc[end_of_day_timestamp]['sma_close_200d']

    print(f"  200-Day SMA (End of Day): {val_200d_sma:.2f}" if isinstance(val_200d_sma, float) else f"  200-Day SMA (End of Day): {val_200d_sma}")

    sessions = ['Pre-Market', 'Regular', 'Post-Market']
    for session_name in sessions:
        print(f"\n  -- {session_name} Session --")
        session_df = get_session(day_df_ny, date, session_name, tz_str=timezone_str)

        if session_df.empty:
            print("     No data for this session.")
            continue

        session_df = calculate_moving_averages(session_df, field='Close', periods=[9, 20, 50, 200])
        vwap_df = calculate_vwap(session_df.copy())
        final_vwap = vwap_df['vwap'].iloc[-1] if not vwap_df.empty else 'N/A'

        vol_profiler = VolumeProfiler(tick_size=tick_size)
        vol_profile = vol_profiler.calculate(session_df)
        mkt_profiler = MarketProfiler(tick_size=tick_size)
        mkt_profile = mkt_profiler.calculate(session_df)

        print(f"     Time Range : {session_df.index[0].time()} - {session_df.index[-1].time()} (ET)")
        print(f"     Ending VWAP: {final_vwap:.2f}" if isinstance(final_vwap, float) else f"     Ending VWAP: {final_vwap}")

        print("     Ending Moving Averages (Minutes):")
        for period in [9, 20, 50, 200]:
            sma_col = f'sma_close_{period}m'
            ema_col = f'ema_close_{period}m'
            sma_val = session_df[sma_col].iloc[-1] if sma_col in session_df.columns and not session_df[sma_col].empty else 'N/A'
            ema_val = session_df[ema_col].iloc[-1] if ema_col in session_df.columns and not session_df[ema_col].empty else 'N/A'
            print(f"       - {period}-min SMA: {sma_val:.2f}" if isinstance(sma_val, float) else f"       - {period}-min SMA: {sma_val}")
            print(f"       - {period}-min EMA: {ema_val:.2f}" if isinstance(ema_val, float) else f"       - {period}-min EMA: {ema_val}")

        # --- FIX: Restored the missing print logic for profiles ---
        print("     Volume Profile:")
        if vol_profile:
            print(f"       - POC: {vol_profile['poc_price']:.2f}")
            print(f"       - VAH: {vol_profile['value_area_high']:.2f}")
            print(f"       - VAL: {vol_profile['value_area_low']:.2f}")
        else:
            print("       - Not enough data to calculate.")

        print("     Market Profile (TPO):")
        if mkt_profile:
            print(f"       - POC: {mkt_profile['poc_price']:.2f}")
            print(f"       - VAH: {mkt_profile['value_area_high']:.2f}")
            print(f"       - VAL: {mkt_profile['value_area_low']:.2f}")
        else:
            print("       - Not enough data to calculate.")

def main():
    """Main function to drive the indicator verification script."""
    print("--- Indicator Verification Tool ---")

    DATA_DIR = os.path.join(project_root, 'data', 'historical')
    TICK_SIZE = 0.01

    ticker = input("Enter the ticker symbol to analyze (e.g., PLTR): ").upper()
    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    full_data = load_data_for_ticker(DATA_DIR, ticker)
    if full_data is None or full_data.empty:
        print("Could not load data.")
        return

    print("\nCalculating 200-day SMA across all available data...")
    full_data = calculate_200_day_sma(full_data, field='Close')
    print("...calculation complete.")

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for trade_date in date_range:
        day_data = full_data[full_data.index.date == trade_date.date()]
        if not day_data.empty:
            analyze_day(day_data, full_data, tick_size=TICK_SIZE)

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()