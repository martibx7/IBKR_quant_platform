# strategies/volume_poc_strategy.py

# ... (imports)
from strategies.base import BaseStrategy

# Update __init__ to accept kwargs for parameters
class SimplePocCrossStrategy(BaseStrategy):
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
                self.ledger.record_trade(
                    timestamp=current_bar['Date'], symbol=symbol,
                    quantity=self.trade_quantity, price=current_bar['Close'],
                    order_type='SELL', market_prices=market_prices
                )
                self.open_position = None
                return # Stop processing for this bar after selling

        # --- Entry Logic ---
        if len(session_bars[symbol]) < self.lookback_period or self.open_position:
            return

        profiler = VolumeProfiler(session_bars[symbol], tick_size=0.01)
        poc = profiler.poc_price
        if poc is None: return

        price = current_bar['Close']
        if price > poc:
            self.ledger.record_trade(
                timestamp=current_bar['Date'], symbol=symbol,
                quantity=self.trade_quantity, price=price,
                order_type='BUY', market_prices=market_prices
            )
            # Store info about the open position
            self.open_position = {'entry_bar_index': len(session_bars[symbol])}
