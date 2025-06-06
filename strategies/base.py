# strategies/base.py

import pandas as pd
from core.ledger import BacktestLedger

class BaseStrategy:
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, symbols: list[str], ledger: BacktestLedger, **kwargs):
        """
        Initializes the base strategy.

        Args:
            symbols (list[str]): A list of symbols the strategy will trade.
            ledger (BacktestLedger): The ledger for recording trades.
            **kwargs: Catches any additional strategy-specific parameters from the config.
        """
        self.symbols = symbols
        self.ledger = ledger
        # Specific strategies will handle their own kwargs, the base class just accepts them.
        pass

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """Called once at the beginning of each trading session."""
        pass

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict):
        raise NotImplementedError("Strategy must implement on_bar method!")

    def on_session_end(self):
        """Called once at the end of each trading session."""
        pass