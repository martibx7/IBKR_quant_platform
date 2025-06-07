import os
import time
import datetime
from polygon import RESTClient
import pandas as pd
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("POLYGON_API_KEY")
TICKERS_FILE = 'tickers.txt'
DATA_DIR = 'data/historical_polygon'
# Polygon's free tier allows 5 requests per minute, so we must pause at least 12 seconds.
PAUSE_SEC = 13

def load_tickers(file_path: str) -> list[str]:
    """Loads a list of tickers from a text file."""
    try:
        with open(file_path, 'r') as f:
            # .strip() removes whitespace, .upper() standardizes to uppercase
            tickers = [line.strip().upper() for line in f if line.strip()]
        if not tickers:
            print(f"Warning: Tickers file at {file_path} is empty.")
        return tickers
    except FileNotFoundError:
        print(f"ERROR: Tickers file not found at {file_path}")
        return []

def download_polygon_data(api_key: str, tickers: list, start_date_str: str, end_date_str: str, data_dir: str):
    """
    Downloads 1-minute historical data for a list of tickers from Polygon.io
    and saves it into CSV files, respecting free-tier rate limits.
    """
    if not api_key or "YOUR_API_KEY" in api_key:
        print("ERROR: POLYGON_API_KEY not found. Please set it in your .env file.")
        return

    client = RESTClient(api_key)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Starting download for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        print(f"\n({i+1}/{len(tickers)}) Processing ticker: {ticker}")

        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=start_date_str,
                to=end_date_str,
                adjusted=True,
                limit=50000
            )

            if not aggs:
                print(f"  - No data found for {ticker} in the specified date range.")
                continue

            # The client returns a list of objects with attributes like .timestamp, .open, etc.
            # We can convert this directly to a DataFrame.
            df = pd.DataFrame(aggs)

            # --- FIX IS HERE ---
            # 1. The timestamp field is named 'timestamp', not 't'.
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # 2. The other column names are already correct (open, high, low, close, volume),
            #    so the rename step is no longer needed.

            # Select and reorder columns to our desired format
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            filepath = os.path.join(data_dir, f'{ticker}.csv')
            df.to_csv(filepath, index=False)
            print(f"  - Successfully saved data for {ticker} to {filepath}")

        except KeyError as e:
            # Add a specific error message for this common issue
            print(f"  - A KeyError occurred for {ticker}: {e}. This may be due to an unexpected response format from the API.")
        except Exception as e:
            print(f"  - An error occurred for {ticker}: {e}")

        # Pause to respect the API rate limit
        if i < len(tickers) - 1:
            print(f"  - Pausing for {PAUSE_SEC} seconds to respect API rate limit...")
            time.sleep(PAUSE_SEC)

    print("\nData download complete.")

def main():
    """
    Main function to drive the script.
    """
    tickers = load_tickers(TICKERS_FILE)
    if not tickers:
        return

    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")

    # Basic validation for date format
    try:
        datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    download_polygon_data(API_KEY, tickers, start_date_str, end_date_str, DATA_DIR)

if __name__ == "__main__":
    main()