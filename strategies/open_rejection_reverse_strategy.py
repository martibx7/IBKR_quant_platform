# strategies/open_rejection_reverse_strategy.py

from datetime import datetime
import pandas as pd
from tqdm import tqdm
import pytz
import logging
import os

from .base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session

class OpenRejectionReverseStrategy(BaseStrategy):
    def __init__(self, symbols: list[str], ledger, **params):
        super().__init__(symbols, ledger, **params)
        self.risk_per_trade_pct = params.get('risk_per_trade_pct', 0.01)
        self.max_allocation_pct = params.get('max_allocation_pct', 0.25)
        self.take_profit_r = params.get('take_profit_r', 2.5)
        self.trailing_stop_r = params.get('trailing_stop_r', 1.0)
        self.params = params

        self.profile_type = params.get('profile_type', 'volume')
        self.tick_size = params.get('tick_size', 0.01)
        if self.profile_type == 'volume':
            self.profiler_class = VolumeProfiler
        elif self.profile_type == 'market':
            self.profiler_class = MarketProfiler
        else:
            raise ValueError(f"Invalid profile_type '{self.profile_type}'.")

        self.timezone = pytz.timezone(params.get('timezone', 'America/New_York'))
        self.prev_day_stats = {}
        self.active_trades = {}
        self.disqualified_today = set()
        self.opening_bar_info = {}
        self.session_data_today = {}
        self.current_prices = {}
        self.dip_occurred = set()

        self._setup_logger(params.get('log_file', 'logs/open_rejection_reverse.log'))
        self.logger.info(f"Strategy initialized to use '{self.profile_type}' profiles with tick size {self.tick_size}.")

    def _setup_logger(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info("Strategy logger initialized.")

    def scan_for_candidates(self, all_data: dict[str, pd.DataFrame], date: datetime.date) -> list[str]:
        self.logger.info(f"--- SCANNING CANDIDATES FOR {date.strftime('%Y-%m-%d')} using {self.profile_type} profiler ---")
        candidates = []
        min_volume = self.params.get('min_daily_volume', 500000)
        min_range_pct = self.params.get('min_prev_day_range_pct', 0.03)

        pre_candidates = []
        for symbol, df in all_data.items():
            prev_day_data = get_session(df, date, "Regular", tz_str=self.timezone.zone)
            if prev_day_data.empty or len(prev_day_data) < 2:
                continue

            daily_volume = prev_day_data['Volume'].sum()
            if daily_volume < min_volume: continue

            daily_high = prev_day_data['High'].max()
            daily_low = prev_day_data['Low'].min()
            day_range = (daily_high - daily_low) / daily_low if daily_low > 0 else 0
            if day_range < min_range_pct: continue

            pre_candidates.append({"symbol": symbol, "data": prev_day_data})

        self.logger.info(f"Pre-filtered to {len(pre_candidates)} symbols based on volume and range.")

        for item in tqdm(pre_candidates, desc=f"Profiling {len(pre_candidates)} pre-candidates", leave=False):
            symbol, prev_day_data = item['symbol'], item['data']
            profiler = self.profiler_class(self.tick_size)
            profile = profiler.calculate(prev_day_data)

            if not profile or profile.get('poc_price') is None: continue

            close = prev_day_data.iloc[-1]['Close']
            day_high = prev_day_data['High'].max()
            day_low = prev_day_data['Low'].min()

            if close > profile['poc_price'] and close > (day_high - 0.25 * (day_high - day_low)):
                self.logger.info(f"  [CANDIDATE] {symbol} | Close: {close:.2f} > POC: {profile['poc_price']:.2f} | VAH: {profile['value_area_high']:.2f}")
                candidates.append(symbol)
                self.prev_day_stats[symbol] = {'vah': profile['value_area_high'], 'poc': profile['poc_price']}

        self.logger.info(f"Found {len(candidates)} candidates after scanning.")
        return candidates

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        self.logger.info("--- NEW SESSION ---")
        self.active_trades.clear()
        self.disqualified_today.clear()
        self.opening_bar_info.clear()
        self.session_data_today = session_data
        # --- FIX: Use capitalized 'Open' to match DataFrame columns ---
        self.current_prices = {s: df.iloc[0]['Open'] for s, df in session_data.items()}
        self.dip_occurred.clear()

        for symbol, df in session_data.items():
            if symbol in self.prev_day_stats:
                # --- FIX: Use capitalized 'Open' ---
                open_price = df.iloc[0]['Open']
                prev_vah = self.prev_day_stats[symbol]['vah']
                if open_price <= prev_vah:
                    self.disqualified_today.add(symbol)
                    self.logger.info(f"  [DISQUALIFIED] {symbol}: Open ({open_price:.2f}) not above prior VAH ({prev_vah:.2f}).")
                else:
                    opening_bar = df.iloc[0]
                    # --- FIX: Use capitalized column names ---
                    self.opening_bar_info[symbol] = {'open': opening_bar['Open'], 'high': opening_bar['High'], 'low': opening_bar['Low']}
                    self.logger.info(f"  [SETUP OK] {symbol} opened at {open_price:.2f}, above prior VAH.")

    def on_bar(self, symbol: str, bar: pd.Series):
        # --- FIX: Use capitalized 'Close' ---
        self.current_prices[symbol] = bar['Close']

        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            trailing_trigger_price = trade['entry_price'] + (trade['risk_per_share'] * self.trailing_stop_r)
            if bar['High'] >= trailing_trigger_price:
                bars_since_entry = self.session_data_today[symbol].loc[lambda x: x.index >= trade['entry_timestamp']]
                if not bars_since_entry.empty:
                    trailing_profiler = self.profiler_class(self.tick_size)
                    profile = trailing_profiler.calculate(bars_since_entry)
                    if profile and profile.get('poc_price'):
                        new_stop_loss = profile['poc_price'] - self.tick_size
                        if new_stop_loss > trade['stop_loss']:
                            self.logger.info(f"  [ADJUST] Trailing stop for {symbol} to {new_stop_loss:.2f}")
                            trade['stop_loss'] = new_stop_loss

            exit_price, exit_reason = 0, None
            if bar.name.time() >= pd.to_datetime('15:55:00').time():
                exit_price, exit_reason = bar['Close'], "EOD"
            elif bar['Low'] <= trade['stop_loss']:
                exit_price, exit_reason = trade['stop_loss'], "Stop Loss"
            elif bar['High'] >= trade['take_profit']:
                exit_price, exit_reason = trade['take_profit'], "Take Profit"

            if exit_reason:
                self.logger.info(f"  [EXIT] {exit_reason} for {symbol} at {exit_price:.2f}")
                self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices)
                del self.active_trades[symbol]
            return

        if symbol in self.disqualified_today or symbol not in self.opening_bar_info:
            return

        opening_info = self.opening_bar_info[symbol]
        if bar['Low'] < opening_info['open']:
            self.dip_occurred.add(symbol)

        if symbol in self.dip_occurred and bar['Close'] > opening_info['high']:
            entry_price = opening_info['high']
            rejection_low = self.session_data_today[symbol].loc[:bar.name]['Low'].min()
            risk_per_share = entry_price - rejection_low
            if risk_per_share <= 0: return

            equity = self.ledger.get_total_equity(self.current_prices)
            risk_quantity = int((equity * self.risk_per_trade_pct) / risk_per_share)
            alloc_quantity = int((equity * self.max_allocation_pct) / entry_price)
            cash_quantity = int(self.ledger.cash / entry_price)
            quantity = min(risk_quantity, alloc_quantity, cash_quantity)
            if quantity == 0:
                self.disqualified_today.add(symbol)
                return

            self.logger.info(f">>> TRIGGER for {symbol} at {bar.name.time()} <<<")
            self.logger.info(f"  [EXECUTE] LONG {symbol} | Qty: {quantity} @ {entry_price:.2f}")
            if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                self.active_trades[symbol] = {
                    'entry_price': entry_price, 'quantity': quantity,
                    'stop_loss': rejection_low,
                    'take_profit': entry_price + (risk_per_share * self.take_profit_r),
                    'risk_per_share': risk_per_share, 'entry_timestamp': bar.name
                }
            self.disqualified_today.add(symbol)

    def on_session_end(self):
        pass