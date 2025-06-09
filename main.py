# main.py
import yaml
from datetime import datetime, timedelta
import os
import pandas as pd
from backtest.engine import BacktestEngine

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

    start_date_str = input("Enter backtest start date (YYYY-MM-DD): ")
    end_date_str = input("Enter backtest end date (YYYY-MM-DD): ")
    trade_dates = get_trade_dates(start_date_str, end_date_str)

    if not trade_dates:
        print("No trading dates in the specified range.")
        return

    # --- Initialize and Run the Engine ---
    engine = BacktestEngine(config_path)

    # This part for aggregation is not fully implemented yet,
    # but the foundation is here. We'll track the last day's results.
    final_results = None
    for trade_date in trade_dates:
        try:
            # === MODIFIED: More realistic daily flow ===
            engine.prepare_for_day(trade_date)
            results = engine.run_session()

            if results:
                final_results = results # Keep track of the latest results

        except Exception as e:
            print(f"An error occurred on {trade_date.strftime('%Y-%m-%d')}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue

    # --- Aggregate and Display Final Results ---
    if final_results:
        print("\n--- Backtest Run Complete ---")
        final_results.print_summary()
        final_results.plot_equity_curve()
    else:
        print("\nBacktest run finished, but no results were generated.")

if __name__ == '__main__':
    main()