# strategies/base.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
import os

# --- CORRECTED IMPORT ---
from core.ledger import BacktestLedger

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, symbols: list[str], ledger: BacktestLedger, config: dict, **params):
        self.symbols = symbols
        self.ledger = ledger
        self.config = config
        self.params = params
        self.active_trades = {}
        self.current_prices = {s: 0 for s in symbols}
        self.data_for_day = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Sets up a strategy-specific logger."""
        logger = logging.getLogger(self.__class__.__name__)
        log_file = self.params.get('log_file')
        if log_file:
            # Ensure logs directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Remove old handler if it exists
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(log_file):
                    logger.removeHandler(handler)

            if not logger.handlers:
                handler = logging.FileHandler(log_file, mode='w')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        logger.setLevel(logging.INFO) # Default level
        # Prevent logging from propagating to the root logger
        logger.propagate = False
        return logger

    @abstractmethod
    def get_required_lookback(self) -> int:
        """
        Returns the number of days of historical data required by the strategy
        before the current trading day.
        """
        pass

    def on_new_day(self, trade_date: pd.Timestamp, data: dict[str, pd.DataFrame]):
        """
        Called at the start of each new trading day. Sets the data for the day.
        """
        self.data_for_day = data

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """
        Called once at the very start of the trading session simulation.
        """
        pass # Optional for strategies to implement

    def on_bar(self, symbol: str, bar: pd.Series):
        """
        Called for each new bar of data for a specific symbol.
        """
        self.current_prices[symbol] = bar['Close']

    def on_session_end(self):
        """
        Called once at the very end of the trading session simulation.
        """
        pass # Optional for strategies to implement