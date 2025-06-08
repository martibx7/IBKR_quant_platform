# strategies/base.py

import pandas as pd
from core.ledger import BacktestLedger

class BaseStrategy:
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, symbols: list[str], ledger: BacktestLedger, **kwargs):
        self.symbols = symbols
        self.ledger = ledger
        pass

    def scan_for_candidates(self, current_date: pd.Timestamp, historical_data: dict) -> list[str]:
        """
        Scans all historical data to select tickers for the current day.
        This method should be overridden by specific strategies.
        """
        raise NotImplementedError("Strategy must implement scan_for_candidates method!")

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        pass

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict = None):
        """
        Process a new bar of data for all symbols.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Should implement 'on_bar' in a subclass.")

    def on_session_end(self):
        pass