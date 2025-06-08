import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from analytics.profiles import MarketProfiler, VolumeProfiler, get_session
from analytics.indicators import calculate_vwap
import pytz

class OpenDriveMomentumStrategy(BaseStrategy):
    """
    Implements the Session-Aware Open-Drive Momentum Strategy with robust risk management.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.01)
        self.max_allocation_pct = kwargs.get('max_allocation_pct', 0.25) # Max 25% of equity in one trade
        self.min_risk_per_share = kwargs.get('min_risk_per_share', 0.05) # Min 5 cents risk

        self.entry_time_str = kwargs.get('entry_time', '10:00:00')
        self.exit_time_str = kwargs.get('exit_time', '15:55:00')
        self.tz_str = kwargs.get('timezone', 'America/New_York')

        self.timezone = pytz.timezone(self.tz_str)
        self.entry_time = pd.to_datetime(self.entry_time_str).time()
        self.exit_time = pd.to_datetime(self.exit_time_str).time()

        self.prev_day_stats = {}
        self.position_details = {}


    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        """Calculates previous day's stats for each symbol."""
        candidates = []
        for symbol, df in historical_data.items():
            if df.empty: continue

            regular_session_df = df[df.index.to_series().apply(
                lambda dt: get_session(dt.tz_convert(self.timezone)) == 'Regular'
            )]
            if regular_session_df.empty: continue

            prev_day_m_profiler = MarketProfiler(regular_session_df, session='Regular')
            if prev_day_m_profiler.poc_price is not None:
                self.prev_day_stats[symbol] = {
                    'vah': prev_day_m_profiler.vah, 'val': prev_day_m_profiler.val, 'poc': prev_day_m_profiler.poc_price
                }
                candidates.append(symbol)
        return candidates


    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """Resets the state for the new day."""
        self.position_details = {}


    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        """Main event loop for the strategy."""
        for symbol in current_bar_data.keys():
            if symbol not in session_bars or symbol not in analytics: continue

            current_bar = current_bar_data[symbol]
            current_bar_time = current_bar.name.tz_convert(self.timezone).time()

            # --- EXIT LOGIC ---
            if symbol in self.position_details:
                details = self.position_details[symbol]

                if current_bar_time >= self.exit_time or current_bar['Low'] <= details['stop_loss']:
                    exit_price = details['stop_loss'] if current_bar['Low'] <= details['stop_loss'] else current_bar['Close']
                    self.ledger.record_trade(
                        timestamp=current_bar.name, symbol=symbol, quantity=details['quantity'],
                        price=exit_price, order_type='SELL', market_prices=market_prices
                    )
                    del self.position_details[symbol]
                    continue

                trade_duration = current_bar.name - details['entry_time']
                if trade_duration > pd.Timedelta(minutes=30):
                    vwap_df = analytics[symbol]['vwap']
                    if len(vwap_df) > 1 and current_bar['Low'] < vwap_df.iloc[-2]['vwap']:
                        self.ledger.record_trade(
                            timestamp=current_bar.name, symbol=symbol, quantity=details['quantity'],
                            price=current_bar['Close'], order_type='SELL', market_prices=market_prices
                        )
                        del self.position_details[symbol]
                        continue

            # --- ENTRY LOGIC ---
            if current_bar_time == self.entry_time and symbol not in self.position_details:
                if symbol not in self.prev_day_stats: continue

                current_session_df = session_bars[symbol]
                prev_day = self.prev_day_stats[symbol]
                opening_bar = current_session_df.iloc[0]
                vwap_df = analytics[symbol]['vwap']

                if not (opening_bar['Open'] > prev_day['vah'] and opening_bar['Open'] > prev_day['poc']): continue
                if vwap_df.empty or not (current_bar['Close'] > vwap_df.iloc[-1]['vwap']): continue

                opening_drive_bars = current_session_df.iloc[0:30]
                od_profiler = VolumeProfiler(opening_drive_bars, session='Regular')
                if od_profiler.poc_price is None: continue

                od_high, od_low = opening_drive_bars['High'].max(), opening_drive_bars['Low'].min()
                if not (od_profiler.poc_price <= (od_high + od_low) / 2): continue

                entry_price = current_bar['Close']
                stop_loss_price = od_low
                risk_per_share = entry_price - stop_loss_price

                # --- FIX 1: Ensure risk is meaningful ---
                if risk_per_share < self.min_risk_per_share:
                    print(f"[{current_bar.name}] SKIPPING {symbol}: Risk per share ({risk_per_share:.2f}) is below minimum ({self.min_risk_per_share}).")
                    continue

                equity = self.ledger.cash
                risk_amount = equity * self.risk_per_trade
                quantity = int(risk_amount / risk_per_share)

                # --- FIX 2: Cap total position allocation ---
                max_position_value = equity * self.max_allocation_pct
                proposed_position_value = quantity * entry_price

                if proposed_position_value > max_position_value:
                    quantity = int(max_position_value / entry_price)
                    print(f"[{current_bar.name}] SIZING {symbol}: Position size capped by max allocation. New quantity: {quantity}")

                if quantity > 0:
                    print(f"[{current_bar.name}] ENTRY CONFIRMED for {symbol} | Qty: {quantity}")
                    trade_successful = self.ledger.record_trade(
                        timestamp=current_bar.name, symbol=symbol, quantity=quantity,
                        price=entry_price, order_type='BUY', market_prices=market_prices
                    )
                    if trade_successful:
                        self.position_details[symbol] = {
                            'quantity': quantity, 'stop_loss': stop_loss_price, 'entry_time': current_bar.name
                        }