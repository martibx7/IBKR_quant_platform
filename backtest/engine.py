# backtest/engine.py

import os
import pandas as pd
from strategies.base import BaseStrategy
from backtest.results import calculate_performance_metrics, print_performance_report, plot_equity_curve
from analytics.indicators import calculate_vwap # Import our new function

class BacktestEngine:
    # ... (__init__, _load_all_data, _create_master_timeline are unchanged) ...
    def __init__(self, data_dir: str, strategy: BaseStrategy):
        self.data_dir = data_dir
        self.strategy = strategy
        self.all_data = self._load_all_data()
        self.master_timeline = self._create_master_timeline()

    def _load_all_data(self) -> dict[str, pd.DataFrame]:
        """Loads all CSV files from the data directory into a dictionary of DataFrames."""
        data = {}
        print(f"Loading data from: {self.data_dir}")
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                symbol = os.path.splitext(filename)[0]
                filepath = os.path.join(self.data_dir, filename)
                try:
                    df = pd.read_csv(filepath, parse_dates=['Date'])
                    data[symbol] = df
                    print(f"  - Loaded {symbol}")
                except Exception as e:
                    print(f"  - Error loading {filename}: {e}")
        return data

    def _create_master_timeline(self) -> list:
        """Creates a master index of all unique trading days across all tickers."""
        all_dates = set()
        for df in self.all_data.values():
            all_dates.update(df['Date'].dt.date)
        return sorted(list(all_dates))


    def run(self):
        """
        Runs the backtest, processing raw data into profiles and VWAP.
        """
        if not self.all_data:
            print("No data loaded. Exiting backtest.")
            return

        print("\n--- Starting Portfolio Backtest ---")

        # Keep track of the previous day's data
        prev_trade_date = None

        for trade_date in self.master_timeline:
            date_str = trade_date.strftime('%Y-%m-%d')
            print(f"\n--- Simulating Day: {date_str} ---")

            if prev_trade_date is None:
                prev_trade_date = trade_date
                continue # Skip the first day as there's no "previous day" to analyze

            # 1. SCANNING STEP (using data from prev_trade_date)
            historical_data = {
                symbol: df[df['Date'].dt.date == prev_trade_date]
                for symbol, df in self.all_data.items()
            }
            # The strategy selects which tickers to trade today
            candidate_tickers = self.strategy.scan_for_candidates(trade_date, historical_data)

            if not candidate_tickers:
                print("No candidates found for today.")
                prev_trade_date = trade_date
                continue

            print(f"Today's Candidates: {candidate_tickers}")

            # 2. EXECUTION STEP
            todays_market_data = {
                symbol: df[df['Date'].dt.date == trade_date].copy().reset_index(drop=True)
                for symbol, df in self.all_data.items() if symbol in candidate_tickers
            }
            todays_market_data = {k: v for k, v in todays_market_data.items() if not v.empty}
            if not todays_market_data:
                print("No data available for candidate tickers on this day.")
                prev_trade_date = trade_date
                continue

            # **Calculate VWAP for each candidate ticker for the day**
            for symbol in todays_market_data:
                todays_market_data[symbol] = calculate_vwap(todays_market_data[symbol])

            # The strategy can run session start logic, now using pre-calculated profiles
            self.strategy.on_session_start(todays_market_data)

            num_bars = len(next(iter(todays_market_data.values())))
            for i in range(num_bars):
                current_bar_data = {
                    symbol: df.iloc[i] for symbol, df in todays_market_data.items()
                }
                session_bars = {
                    symbol: df.iloc[:i+1] for symbol, df in todays_market_data.items()
                }

                market_prices = {
                    pos: current_bar_data.get(pos, {}).get('Close')
                    for pos in self.strategy.ledger.open_positions
                }
                market_prices = {k: v for k, v in market_prices.items() if v is not None}

                self.strategy.on_bar(current_bar_data, session_bars, market_prices)

            # Update prev_trade_date for the next loop
            prev_trade_date = trade_date

        # --- Generate Final Performance Report ---
        self.strategy.on_session_end()
        print("\n--- Backtest Complete ---")
        metrics = calculate_performance_metrics(self.strategy.ledger)
        print_performance_report(metrics)
        plot_equity_curve(self.strategy.ledger)