#!/usr/bin/env python3
"""
Downloads historical data from IBKR for tickers in tickers.txt,
saving all data for a ticker into a single file.
This version includes better pacing and a longer timeout to avoid API errors.
"""
import os
import time
import datetime
from typing import List
import logging
from ib_insync import *
import pandas as pd

# CONFIG
DATA_DIR = 'data/historical'
TICKERS_FILE = 'tickers.txt'
# --- MODIFICATION 1: Increase the pause between requests ---
PAUSE_SEC = 3 # Increased to 10 seconds to respect API pacing limits
CLIENT_ID = 1

def load_tickers(file_path: str) -> List[str]:
    # This function remains the same
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        logging.error(f"Tickers file not found at {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading tickers file: {e}")
        return []

def download_ibkr_data(symbols: List[str], start_date: datetime.date, end_date: datetime.date, data_dir: str = DATA_DIR):
    ib = IB()
    try:
        ib.connect('127.0.0.1', port=4001, clientId=CLIENT_ID)
        logging.info(f"Connected to IBKR at 127.0.0.1:{4001}")
        os.makedirs(data_dir, exist_ok=True)

        for i, sym in enumerate(symbols, 1):
            logging.info(f"\nProcessing symbol {sym} ({i}/{len(symbols)})")

            all_bars_df = pd.DataFrame()
            chunk_start_date = start_date

            while chunk_start_date <= end_date:
                chunk_end_date = chunk_start_date + datetime.timedelta(days=29)
                if chunk_end_date > end_date:
                    chunk_end_date = end_date

                duration_days = (chunk_end_date - chunk_start_date).days + 1
                duration_str = f'{duration_days} D'

                logging.info(f"  - Downloading data from {chunk_start_date} to {chunk_end_date}...")

                try:
                    contract = Stock(sym, 'SMART', 'USD')
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime=datetime.datetime.combine(chunk_end_date, datetime.time(23, 59, 59)),
                        durationStr=duration_str,
                        barSizeSetting='1 min',
                        whatToShow='TRADES',
                        useRTH=True,
                        # --- MODIFICATION 2: Add a longer timeout ---
                        timeout=120  # Wait up to 2 minutes for the request to complete
                    )

                    if bars:
                        chunk_df = util.df(bars)
                        all_bars_df = pd.concat([all_bars_df, chunk_df])
                    else:
                        logging.warning(f"  - No data for {sym} in this chunk.")

                    logging.info(f"  - Pausing for {PAUSE_SEC} seconds...")
                    time.sleep(PAUSE_SEC)

                except Exception as e:
                    logging.error(f"  - Error for {sym} in chunk {chunk_start_date}-{chunk_end_date}: {e}")
                    logging.info(f"  - Pausing for {PAUSE_SEC} seconds after error...")
                    time.sleep(PAUSE_SEC)

                chunk_start_date += datetime.timedelta(days=30)

            if not all_bars_df.empty:
                # Remove any duplicate rows that might occur at chunk boundaries, using the index
                all_bars_df = all_bars_df[~all_bars_df.index.duplicated(keep='first')] # <<< THIS IS THE NEW, CORRECTED LINE
                all_bars_df.sort_index(inplace=True) # Also sort by index now

                filename = f'{sym}.csv'
                filepath = os.path.join(data_dir, filename)
                all_bars_df.to_csv(filepath, index=False)
                logging.info(f"Successfully saved consolidated file for {sym} to {filepath}")

    finally:
        if ib.isConnected():
            ib.disconnect()
            logging.info("Disconnected from IBKR.")

# main() function remains the same
def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
    tickers = load_tickers(TICKERS_FILE)
    if not tickers: return

    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        return

    download_ibkr_data(tickers, start_date, end_date)
    logging.info("Data download complete.")

if __name__ == "__main__":
    main()