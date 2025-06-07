# strategies/open_drive_momentum_strategy.py

import pandas as pd
from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler
import numpy as np

class OpenDriveMomentumStrategy(BaseStrategy):
    """
    Trades on an Open-Drive signal, which indicates a potential Trend Day.
    - An Open-Drive is identified in the first 30 minutes of the session.
    - The open must be outside the previous day's value area.
    - The initial move must be strong and directional.
    - Position size is calculated to risk a fixed percentage of equity.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters ---
        self.open_drive_period = kwargs.get('open_drive_period', 30) # In minutes/bars
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.01) # 1% of equity

        # --- State Variables ---
        self.traded_today = False
        self.position_details = {}
        self.prev_day_stats = {}

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """
        Calculate previous day's stats.
        NOTE: This requires the backtest engine to provide previous day's data.
        For this example, we assume `session_data` contains data for both
        the previous and current trading day.
        """
        symbol = self.symbols[0]
        full_df = session_data[symbol]
        today_date = full_df['Date'].iloc[-1].date()

        # Filter for previous day's data
        prev_day_df = full_df[full_df['Date'].dt.date < today_date]

        if not prev_day_df.empty:
            # Calculate previous day's Value Area using the Volume Profiler
            prev_day_profiler = VolumeProfiler(prev_day_df, tick_size=0.01)
            if prev_day_profiler.poc_price is not None:
                self.prev_day_stats = {
                    'vah': prev_day_profiler.vah,
                    'val': prev_day_profiler.val
                }
        # Reset state for the new day
        self.traded_today = False
        self.position_details = {}


    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict):
        symbol = self.symbols[0]
        current_bar = current_bar_data[symbol]
        current_session = session_bars[symbol]
        bar_index = len(current_session) - 1

        # --- 1. EXIT LOGIC (For open LONG position) ---
        if self.position_details:
            # Simple end-of-day exit
            is_last_bar = bar_index == (len(self.ledger.df) - 1) # Check against the full dataframe length
            if is_last_bar:
                self.ledger.record_trade(
                    timestamp=current_bar['Date'], symbol=symbol,
                    quantity=self.position_details['quantity'], price=current_bar['Close'],
                    order_type='SELL', market_prices=market_prices
                )
                self.position_details = {}
                return

            # Check stop loss
            stop_loss = self.position_details['stop_loss']
            if current_bar['Close'] <= stop_loss:
                self.ledger.record_trade(
                    timestamp=current_bar['Date'], symbol=symbol,
                    quantity=self.position_details['quantity'], price=stop_loss, # Exit at stop price
                    order_type='SELL', market_prices=market_prices
                )
                self.position_details = {}
            return

        # --- 2. ENTRY LOGIC (Long-Only) ---
        # Only check for entry once per day, at the end of the open_drive_period
        if self.traded_today or bar_index != (self.open_drive_period - 1) or not self.prev_day_stats:
            return

        self.traded_today = True # Ensure we only try to enter once per day

        opening_drive_bars = current_session.iloc[:self.open_drive_period]
        open_price = opening_drive_bars.iloc[0]['Open']
        drive_low = opening_drive_bars['Low'].min()
        drive_high = opening_drive_bars['High'].max()
        current_price = current_bar['Close']

        # --- Check for Bullish Open-Drive ---
        # Condition: Open is above previous day's value area and the initial drive holds near its high.
        is_bullish_open_drive = (open_price > self.prev_day_stats.get('vah', np.inf) and
                                 current_price > drive_high * 0.995)

        if is_bullish_open_drive:
            # Condition met: Calculate position size and enter a LONG position

            # Risk Management
            stop_loss_price = drive_low
            risk_per_share = current_price - stop_loss_price
            if risk_per_share <= 0:
                return # Invalid risk, do not trade

            equity = self.ledger.cash # Using cash as a proxy for equity in a cash-only account
            risk_amount = equity * self.risk_per_trade
            quantity = int(risk_amount / risk_per_share)

            if quantity > 0:
                self.ledger.record_trade(
                    timestamp=current_bar['Date'], symbol=symbol,
                    quantity=quantity, price=current_price,
                    order_type='BUY', market_prices=market_prices
                )
                self.position_details = {
                    'direction': 'LONG',
                    'quantity': quantity,
                    'stop_loss': stop_loss_price
                }