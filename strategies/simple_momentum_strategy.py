# strategies/simple_momentum_strategy.py

import pandas as pd
from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler # Import MarketProfiler

class SimpleMomentumStrategy(BaseStrategy):
    """
    A basic strategy that requires confirmation from Volume Profile, VWAP, and TPO Profile.
    - Scans for stocks that gained >= 10% on the previous day.
    - Enters a long position if the price is above the previous day's Volume POC,
      the current intraday VWAP, and the current intraday TPO POC.
    - Position size is 1% of the current cash balance.
    - Exits at a 1R profit target or a 1R stop-loss, where R is 2% of the entry price.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters from config.yaml ---
        self.daily_change_threshold = kwargs.get('daily_change_threshold', 0.10)
        self.position_size_pct = kwargs.get('position_size_pct', 0.01)
        self.r_unit_pct = kwargs.get('r_unit_pct', 0.02)

        # --- Strategy State ---
        self.prev_day_stats = {}
        self.active_trades = {}

    def scan_for_candidates(self, current_date: pd.Timestamp, historical_data: dict) -> list[str]:
        """
        Scans for stocks up >= 10% on the previous day and stores their POC.
        """
        print(f"Scanning {len(historical_data)} tickers for candidates...")
        candidates = []
        self.prev_day_stats = {}

        for symbol, df in historical_data.items():
            if df.empty or len(df) < 2:
                continue

            prev_close = df['Close'].iloc[0]
            last_close = df.iloc[-1]['Close']
            daily_change = (last_close - prev_close) / prev_close

            if daily_change >= self.daily_change_threshold:
                profiler = VolumeProfiler(df, tick_size=0.01)
                if profiler.poc_price:
                    self.prev_day_stats[symbol] = {'poc': profiler.poc_price}
                    candidates.append(symbol)

        return candidates

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict):
        """
        Manages entries and exits based on the triple-confirmation logic.
        """
        for symbol, current_bar in current_bar_data.items():
            # ---- EXIT LOGIC ----
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                if current_bar['Close'] <= trade['stop_loss']:
                    print(f"[{current_bar.name}] STOP LOSS for {symbol} at {trade['stop_loss']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], trade['stop_loss'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                elif current_bar['Close'] >= trade['take_profit']:
                    print(f"[{current_bar.name}] TAKE PROFIT for {symbol} at {trade['take_profit']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], trade['take_profit'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                continue

            # ---- ENTRY LOGIC ----
            if symbol not in self.prev_day_stats:
                continue

            current_price = current_bar['Close'] # The correct variable is defined here
            vwap = current_bar.get('VWAP')
            prev_day_poc = self.prev_day_stats[symbol]['poc']

            intraday_tpo_profiler = MarketProfiler(session_bars[symbol], tick_size=0.01)
            intraday_tpo_poc = intraday_tpo_profiler.poc_price

            if (vwap is not None and intraday_tpo_poc is not None and
                    current_price > vwap and
                    current_price > prev_day_poc and
                    current_price >= intraday_tpo_poc):

                # --- Calculate Position Size and Exits ---
                position_value = self.ledger.cash * self.position_size_pct
                quantity = int(position_value / current_price)
                if quantity == 0: continue

                # --- FIX IS HERE: Use 'current_price' instead of 'entry_price' ---
                r_amount = current_price * self.r_unit_pct
                stop_loss_price = current_price - r_amount
                take_profit_price = current_price + r_amount

                # --- Execute Trade ---
                print(f"[{current_bar.name}] ENTERING LONG for {symbol} at {current_price:.2f}")
                self.ledger.record_trade(current_bar.name, symbol, quantity, current_price, 'BUY', market_prices)

                self.active_trades[symbol] = {
                    'quantity': quantity,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price
                }