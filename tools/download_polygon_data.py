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
PAUSE_SEC = 13      # Pause to respect the 5 requests/minute free tier limit
API_LIMIT = 50000   # Max results per Polygon API call

def load_tickers(file_path: str) -> list[str]:
    """Loads a list of tickers from a text file."""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        if not tickers:
            print(f"Warning: Tickers file at {file_path} is empty.")
        return tickers
    except FileNotFoundError:
        print(f"ERROR: Tickers file not found at {file_path}")
        return []

def download_polygon_data(api_key: str, tickers: list, start_date_str: str, end_date_str: str, data_dir: str):
    """
    Downloads 1-minute historical data for a list of tickers from Polygon.io,
    handling API pagination to retrieve the full date range.
    """
    if not api_key or "YOUR_API_KEY" in api_key:
        print("ERROR: POLYGON_API_KEY not found. Please set it in your .env file.")
        return

    client = RESTClient(api_key)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Starting download for {len(tickers)} tickers from {start_date_str} to {end_date_str}...")

    for i, ticker in enumerate(tickers):
        print(f"\n({i+1}/{len(tickers)}) Processing ticker: {ticker}")

        all_aggs_df = pd.DataFrame()
        # Use the provided start_date_str for the first iteration
        current_start_date = start_date_str

        while True:
            # Fetch a chunk of data
            print(f"  - Fetching data for {ticker} from {current_start_date} to {end_date_str}...")

            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=current_start_date,
                    to=end_date_str,
                    adjusted=True,
                    limit=API_LIMIT
                )

                if not aggs:
                    print(f"  - No more data found for {ticker} in this range. Download for this ticker is complete.")
                    break

                chunk_df = pd.DataFrame(aggs)
                all_aggs_df = pd.concat([all_aggs_df, chunk_df])

                # Check if we've hit the API limit, indicating more data may be available
                if len(chunk_df) < API_LIMIT:
                    print(f"  - Received {len(chunk_df)} bars. This is the final chunk.")
                    break
                else:
                    # We hit the limit, so we need to paginate.
                    # The next request will start AFTER the last timestamp we received.
                    last_timestamp_ms = chunk_df.iloc[-1]['timestamp']

                    # To avoid re-requesting the same last bar, we add one minute (60000 ms)
                    next_timestamp_ms = last_timestamp_ms + 60000
                    current_start_date = pd.to_datetime(next_timestamp_ms, unit='ms').strftime('%Y-%m-%d')

                    print(f"  - Hit API limit of {API_LIMIT} bars. Will fetch next chunk.")
                    print(f"  - Pausing for {PAUSE_SEC} seconds to respect API rate limit...")
                    time.sleep(PAUSE_SEC)


            except Exception as e:
                print(f"  - An error occurred for {ticker}: {e}")
                # Break the loop for this ticker on error to avoid infinite loops
                break

        # After the loop, process the complete DataFrame
        if not all_aggs_df.empty:
            # Drop duplicates that might occur at chunk boundaries
            all_aggs_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

            # Convert timestamp and format the dataframe
            all_aggs_df['date'] = pd.to_datetime(all_aggs_df['timestamp'], unit='ms', utc=True)
            all_aggs_df = all_aggs_df[['date', 'open', 'high', 'low', 'close', 'volume']]
            all_aggs_df.sort_values(by='date', inplace=True)

            filepath = os.path.join(data_dir, f'{ticker}.csv')
            all_aggs_df.to_csv(filepath, index=False)
            print(f"  - Successfully saved {len(all_aggs_df)} total bars for {ticker} to {filepath}")
        else:
            print(f"  - No data was saved for {ticker}.")

        # Pause between different tickers as well
        if i < len(tickers) - 1:
            print(f"  - Pausing for {PAUSE_SEC} seconds before next ticker...")
            time.sleep(PAUSE_SEC)


    print("\n\nData download process complete.")

def main():
    """
    Main function to drive the script.
    """
    tickers = load_tickers(TICKERS_FILE)
    if not tickers:
        return

    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")

    try:
        datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    download_polygon_data(API_KEY, tickers, start_date_str, end_date_str, DATA_DIR)

if __name__ == "__main__":
    main()
