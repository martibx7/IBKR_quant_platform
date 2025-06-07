# strategies/volume_poc_strategy.py

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler

class SimplePocCrossStrategy(BaseStrategy):
    """
    A simple strategy that enters a trade when the price crosses above the
    Point of Control (POC) of a lookback period and holds for a fixed
    number of bars.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        self.open_position = None # Store details of the open trade
        self.trade_quantity = kwargs.get('trade_quantity', 100)
        self.lookback_period = kwargs.get('lookback_period', 15)
        # New parameter: hold for N bars then sell
        self.hold_period = kwargs.get('hold_period', 20)

    def on_bar(self, current_bar_data, session_bars, market_prices):
        symbol = self.symbols[0]
        current_bar = current_bar_data[symbol]

        # --- Exit Logic ---
        if self.open_position:
            bars_since_entry = len(session_bars[symbol]) - self.open_position['entry_bar_index']
            if bars_since_entry >= self.hold_period:
                # Use current_bar.name for the timestamp
                self.ledger.record_trade(
                    timestamp=current_bar.name, symbol=symbol,
                    quantity=self.trade_quantity, price=current_bar['Close'],
                    order_type='SELL', market_prices=market_prices
                )
                print(f"[{current_bar.name}] Exiting position in {symbol} due to hold period.")
                self.open_position = None
                return # Stop processing for this bar after selling

        # --- Entry Logic ---
        if len(session_bars[symbol]) < self.lookback_period or self.open_position:
            return

        # Use the most recent 'lookback_period' bars for the profile
        lookback_bars = session_bars[symbol].iloc[-self.lookback_period:]
        profiler = VolumeProfiler(lookback_bars, tick_size=0.01)
        poc = profiler.poc_price

        if poc is None:
            return

        price = current_bar['Close']
        if price > poc:
            # Use current_bar.name for the timestamp
            self.ledger.record_trade(
                timestamp=current_bar.name, symbol=symbol,
                quantity=self.trade_quantity, price=price,
                order_type='BUY', market_prices=market_prices
            )
            print(f"[{current_bar.name}] Entering position in {symbol} at {price:.2f}")
            # Store info about the open position
            self.open_position = {'entry_bar_index': len(session_bars[symbol])}