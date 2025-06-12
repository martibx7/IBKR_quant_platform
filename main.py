# main.py

import yaml
from datetime import datetime, timedelta
import os
import pandas as pd
from backtest.engine import BacktestEngine
from backtest.results import BacktestResults

def get_trade_dates(start_date_str, end_date_str) -> list[datetime]:
    """
    Generates a list of trading dates between the start and end date,
    excluding weekends.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    trade_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            trade_dates.append(current_date)
        current_date += timedelta(days=1)
    return trade_dates

def main():
    """
    Main execution function for the backtesting platform.
    """
    config_path = 'config.yaml'

    # --- FIX: Added robust input validation for dates ---
    while True:
        start_date_str = input("Enter backtest start date (YYYY-MM-DD): ")
        try:
            # Validate format
            datetime.strptime(start_date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

    while True:
        end_date_str = input("Enter backtest end date (YYYY-MM-DD): ")
        try:
            # Validate format
            datetime.strptime(end_date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")
    # --- END FIX ---

    trade_dates = get_trade_dates(start_date_str, end_date_str)

    if not trade_dates:
        print("No trading dates in the specified range.")
        return

    engine = BacktestEngine(config_path)

    for trade_date in trade_dates:
        try:
            engine.prepare_for_day(trade_date)
            engine.run_session()
        except Exception as e:
            print(f"An error occurred on {trade_date.strftime('%Y-%m-%d')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    final_results = BacktestResults(engine.ledger)

    if not final_results.trade_log.empty:
        print("\n--- Backtest Run Complete ---")
        final_results.print_summary()
        final_results.plot_equity_curve()
    else:
        print("\nBacktest run finished, but no trades were executed.")

if __name__ == '__main__':
    main()