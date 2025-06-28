# main.py

import yaml
from datetime import datetime, timedelta
import os
import pandas as pd
from backtest.engine import BacktestEngine
from backtest.results import BacktestResults
import logging

## def get_trade_dates(start_date_str, end_date_str) -> list[datetime]:
##     start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
##     end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
##     trade_dates = []
##     current_date = start_date
##     while current_date <= end_date:
##         if current_date.weekday() < 5: # Monday to Friday
##             trade_dates.append(current_date)
##         current_date += timedelta(days=1)
##     return trade_dates

def main():
    """
    Main execution function for the backtesting platform.
    """
    config_path = 'config.yaml'

    while True:
        start_date_str = input("Enter backtest start date (YYYY-MM-DD): ")
        try:
            datetime.strptime(start_date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

    while True:
        end_date_str = input("Enter backtest end date (YYYY-MM-DD): ")
        try:
            datetime.strptime(end_date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

    engine = BacktestEngine(config_path, start_date=start_date_str, end_date=end_date_str)

    print("\nStarting backtest...")
    final_ledger = engine.run()

    # We need the engine object to pass to BacktestResults for Alpha/Beta calculation
    results = BacktestResults(final_ledger, engine)

    if results and not results.trade_log.empty:
        print("\n--- Backtest Run Complete ---")
        results.print_summary()

        summary_string = results.get_summary_string()
        engine.log_summary_to_strategy(summary_string)

        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = f"{log_dir}/trade_log_{engine.strategy_name}_{start_date_str}_to_{end_date_str}.xlsx"
        results.save_trade_log_to_excel(log_filename)
        print(f"Trade log saved to '{log_filename}'")

        # --- FIX: Call the plot_equity_curve method directly on the results object ---
        print("Displaying performance chart...")
        results.plot_equity_curve()

    else:
        print("Backtest complete. No trades were executed.")


if __name__ == '__main__':
    main()