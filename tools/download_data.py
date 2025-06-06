#!/usr/bin/env python3
"""
Downloads 1-minute historical data from IBKR for tickers in tickers.txt,
prompts for a date range, and saves data into a flat directory structure
to be consumed by the backtesting engine.

Refined to remove lookahead bias (e.g., pre-calculated daily VWAP).
"""
import os
import time
import datetime
from typing import List, Tuple
import logging
from ib_insync import *
import pandas as pd

# CONFIG
# --- Refined: Data will be saved to a single, flat directory ---
DATA_DIR = 'data/historical'
TICKERS_FILE = 'tickers.txt'
PAUSE_SEC = 0.5
CLIENT_ID = 1

# (load_tickers function remains the same)
def load_tickers(file_path: str) -> List[str]:
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

# --- Refined: This function can be simplified as we only need the previous close ---
def fetch_previous_close(ib: IB, symbol: str, date: datetime.date) -> float | None:
    """Fetches the previous day's closing price."""
    contract = Stock(symbol, 'SMART', 'USD')
    # Request the bar from the day before
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.datetime.combine(date - datetime.timedelta(days=1), datetime.time(23, 59, 59)),
        durationStr='1 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
    )
    if bars:
        return bars[-1].close
    logging.warning(f"No previous day data for {symbol} on {date}")
    return None

def download_ibkr_data(symbols: List[str], start_date: datetime.date, end_date: datetime.date, data_dir: str = DATA_DIR):
    ib = IB()
    try:
        ib.connect('127.0.0.1', port=4001, clientId=CLIENT_ID)
        logging.info(f"Connected to IBKR at 127.0.0.1:{4001}")

        os.makedirs(data_dir, exist_ok=True)

        # Loop through each date in the specified range
        all_dates = [start_date + datetime.timedelta(n) for n in range(int((end_date - start_date).days) + 1)]

        for current_date in all_dates:
            date_str = current_date.strftime('%Y%m%d')
            logging.info(f"Downloading data for {current_date.strftime('%Y-%m-%d')} ({len(symbols)} symbols)...")

            for i, sym in enumerate(symbols, 1):
                try:
                    contract = Stock(sym, 'SMART', 'USD')
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime=datetime.datetime.combine(current_date, datetime.time(23, 59, 59)),
                        durationStr='1 D',
                        barSizeSetting='1 min',
                        whatToShow='TRADES',
                        useRTH=True,
                    )

                    if bars:
                        # --- Refined: Only fetch previous close. No daily open or VWAP to prevent lookahead bias. ---
                        prev_close = fetch_previous_close(ib, sym, current_date)

                        df = util.df(bars)
                        df['PrevClose'] = prev_close

                        # --- Refined: New file naming convention and path ---
                        filename = f'{sym}_{date_str}.csv'
                        filepath = os.path.join(data_dir, filename)
                        df.to_csv(filepath, index=False)
                        logging.info(f"({i}/{len(symbols)}) Saved data for {sym} to {filepath}")
                    else:
                        logging.warning(f"No 1-min data received for {sym} on {date_str}")

                except Exception as e:
                    logging.error(f"Error for {sym} on {date_str}: {e}")

                time.sleep(PAUSE_SEC)

    finally:
        if ib.isConnected():
            ib.disconnect()
            logging.info("Disconnected from IBKR.")

# (main function remains the same)
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