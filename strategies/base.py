# strategies/base.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
import os
import math

from core.ledger import BacktestLedger

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, symbols: list[str], ledger: BacktestLedger, config: dict, timezone=None, **params):
        self.symbols = symbols
        self.ledger = ledger
        self.config = config
        self.params = params
        self.tz = timezone
        self.active_trades = {}
        self.current_prices = {s: 0 for s in symbols}
        self.logger = self._setup_logger()
        self.scale_down_on_cash = self.config.get('backtest', {}).get('scale_down_on_insufficient_cash', False)

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

    def _calculate_position_size(self, entry_price: float, stop_price: float, risk_per_trade_pct: float, liquidity_cap_pct: float = 0, bar_volume: float = 0) -> int:
        """
        Calculates position size based on risk, liquidity, and available cash,
        with an option to scale down if cash is insufficient.
        """
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0

        # 1. Calculate size based on risk
        equity = self.ledger.get_total_equity(self.current_prices)
        dollar_risk = equity * risk_per_trade_pct
        quantity = int(dollar_risk / risk_per_share)

        # 2. Apply liquidity cap if provided
        if liquidity_cap_pct > 0 and bar_volume > 0:
            quantity_by_liquidity = int(bar_volume * liquidity_cap_pct)
            quantity = min(quantity, quantity_by_liquidity)

        if quantity == 0:
            return 0

        # 3. Check against available cash
        cost = quantity * entry_price
        available_cash = self.ledger.get_cash() # Use settled cash for buying power

        if cost > available_cash:
            if self.scale_down_on_cash:
                new_quantity = math.floor(available_cash / entry_price)
                self.logger.debug(f"Insufficient cash for {quantity} shares. Scaling down to {new_quantity} shares.")
                return new_quantity
            else:
                # This will trigger the "Not enough buying power" warning in the ledger
                return quantity

        return quantity

    def log_summary(self, summary_string: str):
        """Writes the final backtest summary to the strategy's log file."""
        self.logger.info("\n" + summary_string)

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