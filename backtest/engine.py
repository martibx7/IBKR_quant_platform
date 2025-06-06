# backtest/engine.py

import pandas as pd
from strategies.base import BaseStrategy
# Import the new results functions
from backtest.results import calculate_performance_metrics, print_performance_report, plot_equity_curve

class BacktestEngine:
    def __init__(self, data_path: str, strategy: BaseStrategy):
        # ... (no changes to __init__ or _load_data)
        self.data_path = data_path
        self.strategy = strategy
        self._load_data()

    def _load_data(self):
        # ... (no changes)
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.data = {self.strategy.symbols[0]: self.df}
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            self.data = {}


    def run(self):
        """Runs the backtest and then generates the performance report."""
        if not self.data:
            return

        print("--- Starting Backtest ---")

        symbol = self.strategy.symbols[0]
        daily_data = self.data[symbol]

        self.strategy.on_session_start({symbol: daily_data})

        for i in range(1, len(daily_data)):
            current_session_bars = daily_data.iloc[:i]
            current_bar = daily_data.iloc[i] # Current bar is now the one we are acting on

            # For the ledger, create a dict of current market prices for open positions
            market_prices = {symbol: current_bar['Close']}

            self.strategy.on_bar(
                current_bar_data={symbol: current_bar},
                session_bars={symbol: current_session_bars},
                market_prices=market_prices
            )

        self.strategy.on_session_end()

        # --- NEW: Generate Performance Report ---
        metrics = calculate_performance_metrics(self.strategy.ledger)
        print_performance_report(metrics)
        plot_equity_curve(self.strategy.ledger)