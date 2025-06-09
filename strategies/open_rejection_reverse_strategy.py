# strategies/open_rejection_reverse_strategy.py

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from analytics.profiles import get_session
import pytz
import logging
import os

class OpenRejectionReverseStrategy(BaseStrategy):
    """
    A long-only strategy that buys failed breakdowns at the open.
    It identifies when the market tests lower, fails to find sellers,
    and then reverses back through the opening price.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters ---
        self.position_pct_limit = kwargs.get('position_pct_limit', 0.03) # Each trade uses up to 3% of available cash
        self.lookback_minutes = kwargs.get('lookback_minutes', 30)
        self.tz_str = kwargs.get('timezone', 'America/New_York')
        self.log_file = kwargs.get('log_file', 'logs/rejection_reverse.log')

        # --- Strategy State ---
        self.timezone = pytz.timezone(self.tz_str)
        self.session_state = {}

        self._setup_logger()

    def _setup_logger(self):
        """Initializes a file-based logger for the strategy."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(self.log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """Reset state at the start of each new day for all symbols."""
        self.logger.info(f"--- NEW SESSION ---")
        self.session_state = {}
        for symbol in self.symbols:
            # Check if the symbol exists in the provided session_data to avoid KeyError
            if symbol in session_data and not session_data[symbol].empty:
                self.session_state[symbol] = {
                    'opening_price': session_data[symbol].iloc[0]['Open'],
                    'initial_low': float('inf'),
                    'entry_check_done': False, # Flag to ensure we only check for entry once
                    'position_details': None
                }
                self.logger.info(f"  [START] {symbol} opened at {self.session_state[symbol]['opening_price']:.2f}")


    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        return [s for s, df in historical_data.items() if not df.empty]

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        for symbol, current_bar in current_bar_data.items():
            state = self.session_state.get(symbol)
            if not state: continue

            current_time = current_bar.name.tz_convert(self.timezone)

            # --- EXIT LOGIC ---
            if symbol in self.ledger.open_positions:
                position = self.ledger.open_positions[symbol]
                if current_time.time() >= pd.to_datetime('15:55:00').time():
                    self.logger.info(f"  [EXIT] EOD Exit for {symbol} at {current_bar['Close']:.2f}")
                    self.ledger.record_trade(
                        current_bar.name, symbol, position['quantity'],
                        current_bar['Close'], 'SELL', market_prices
                    )
                elif current_bar['Low'] <= position.get('stop_loss', -1):
                    self.logger.info(f"  [EXIT] Stop Loss for {symbol} at {position['stop_loss']:.2f}")
                    self.ledger.record_trade(
                        current_bar.name, symbol, position['quantity'],
                        position['stop_loss'], 'SELL', market_prices
                    )
                continue

            # --- ENTRY LOGIC ---
            if get_session(current_time) != 'Regular' or state['entry_check_done']:
                continue

            # Update the initial low during the lookback period
            if state['opening_price'] is not None:
                state['initial_low'] = min(state['initial_low'], current_bar['Low'])

            lookback_end_time = (current_time.normalize() + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=self.lookback_minutes)).tz_convert(self.timezone)

            # Check for entry trigger only once, right after the lookback period
            if current_time > lookback_end_time:
                state['entry_check_done'] = True # Mark check as done for the rest of the day

                opening_price = state['opening_price']
                initial_low = state['initial_low']

                trigger_condition = (initial_low < opening_price) and (current_bar['Close'] > opening_price)

                if trigger_condition:
                    self.logger.info(f">>> FAILED AUCTION TRIGGER for {symbol} at {current_time.time()} <<<")
                    self.logger.info(f"  [COND] Initial Low ({initial_low:.2f}) < Open ({opening_price:.2f}).")
                    self.logger.info(f"  [COND] Current Price ({current_bar['Close']:.2f}) > Open ({opening_price:.2f}).")

                    entry_price = current_bar['Close']
                    stop_loss_price = initial_low

                    if (entry_price - stop_loss_price) <= 0:
                        self.logger.warning(f"  [SKIP] {symbol}: Invalid risk (entry <= stop).")
                        continue

                    # --- NEW BET SIZING ---
                    available_cash = self.ledger.cash
                    position_value = available_cash * self.position_pct_limit
                    quantity = int(position_value / entry_price)

                    if quantity > 0:
                        self.logger.info(f"  [EXECUTE] Entering LONG {symbol} | Qty: {quantity} @ {entry_price:.2f} | Max Cost: ${position_value:,.2f}")
                        trade_successful = self.ledger.record_trade(
                            timestamp=current_bar.name, symbol=symbol, quantity=quantity,
                            price=entry_price, order_type='BUY', market_prices=market_prices
                        )
                        if trade_successful:
                            # Add stop loss to the official ledger position
                            self.ledger.open_positions[symbol]['stop_loss'] = stop_loss_price
                    else:
                        self.logger.info(f"  [SKIP] {symbol}: Calculated quantity is zero.")

