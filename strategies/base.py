# strategies/base.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
import os

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
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Sets up a strategy-specific logger."""
        logger = logging.getLogger(self.__class__.__name__)
        log_file = self.params.get('log_file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(log_file):
                    logger.removeHandler(handler)

            if not logger.handlers:
                # The fix is adding encoding='utf-8' to the line below
                handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger

    @abstractmethod
    def get_required_lookback(self) -> int:
        """Returns the number of trading days of historical data required."""
        pass

    @abstractmethod
    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        """
        NEW: Called once at the start of each day with historical data ONLY.
        Use this to pre-calculate indicators and find potential candidates.
        """
        pass

    @abstractmethod
    def on_bar(self, symbol: str, bar: pd.Series):
        """Called for each new bar of intraday data."""
        self.current_prices[symbol] = bar['close']

    def on_session_end(self):
        """Called once at the very end of the trading session simulation."""
        pass