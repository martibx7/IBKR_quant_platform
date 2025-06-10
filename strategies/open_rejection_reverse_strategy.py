import pandas as pd
import pytz
import logging
import os

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session

class OpenRejectionReverseStrategy(BaseStrategy):
    """
    Implements a session-aware Open Rejection Reversal strategy that can
    be configured to use either a Volume Profile or a Market (TPO) Profile.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters ---
        self.risk_per_trade_pct = kwargs.get('risk_per_trade_pct', 0.01)
        self.max_allocation_pct = kwargs.get('max_allocation_pct', 0.25)
        self.take_profit_r = kwargs.get('take_profit_r', 2.5)
        self.trailing_stop_r = kwargs.get('trailing_stop_r', 1.0)
        self.min_daily_volume = kwargs.get('min_daily_volume', 500000)
        self.min_prev_day_range_pct = kwargs.get('min_prev_day_range_pct', 0.03)
        self.log_file = kwargs.get('log_file', 'logs/open_rejection_reverse.log')
        self.tz_str = kwargs.get('timezone', 'America/New_York')

        # --- Select Profiler and Tick Size based on config ---
        self.profile_type = kwargs.get('profile_type', 'volume')
        if self.profile_type == 'volume':
            self.ProfilerClass = VolumeProfiler
            self.tick_size = kwargs.get('tick_size_volume_profile', 0.01)
        elif self.profile_type == 'market':
            self.ProfilerClass = MarketProfiler
            self.tick_size = kwargs.get('tick_size_market_profile', 0.05)
        else:
            raise ValueError(f"Invalid profile_type '{self.profile_type}' in config. Choose 'volume' or 'market'.")

        # --- Strategy State ---
        self.timezone = pytz.timezone(self.tz_str)
        self.prev_day_stats = {}
        self.active_trades = {}
        self.disqualified_today = set()
        self.opening_bar_info = {}
        self._setup_logger()
        self.logger.info(f"Strategy initialized to use '{self.profile_type}' profiles with tick size {self.tick_size}.")

    def _setup_logger(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.propagate = False # Prevent logs from propagating to the root logger

        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info("Strategy logger initialized.")

    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        self.logger.info(f"--- SCANNING CANDIDATES FOR {trade_date.strftime('%Y-%m-%d')} using {self.profile_type} profiler ---")
        candidates = []
        for symbol, df in historical_data.items():
            if df.empty or len(df) < 2:
                continue

            regular_session_df = df[df.index.to_series().apply(
                lambda dt: get_session(dt.tz_convert(self.timezone)) == 'Regular'
            )]

            if regular_session_df.empty:
                continue

            vol = regular_session_df['Volume'].sum()
            day_high = regular_session_df['High'].max()
            day_low = regular_session_df['Low'].min()
            day_range = (day_high - day_low) / day_low if day_low > 0 else 0

            if vol < self.min_daily_volume:
                continue
            if day_range < self.min_prev_day_range_pct:
                continue

            profiler = self.ProfilerClass(regular_session_df, tick_size=self.tick_size)
            day_close = regular_session_df.iloc[-1]['Close']

            if not profiler.poc_price or not profiler.vah:
                continue
            if not day_close > profiler.poc_price:
                continue
            if not day_close > (day_high - 0.25 * (day_high - day_low)):
                continue

            self.prev_day_stats[symbol] = {'vah': profiler.vah, 'poc': profiler.poc_price}
            candidates.append(symbol)
            self.logger.info(f"  [CANDIDATE] {symbol} | Regular Session Close: {day_close:.2f} > POC: {profiler.poc_price:.2f} | VAH: {profiler.vah:.2f}")

        self.logger.info(f"Found {len(candidates)} candidates after scanning.")
        return candidates

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        self.logger.info("--- NEW SESSION ---")
        self.active_trades.clear()
        self.disqualified_today.clear()
        self.opening_bar_info.clear()

        for symbol, df in session_data.items():
            if symbol in self.prev_day_stats:
                regular_session_df = df[df.index.to_series().apply(
                    lambda dt: get_session(dt.tz_convert(self.timezone)) == 'Regular'
                )]

                if regular_session_df.empty:
                    self.disqualified_today.add(symbol)
                    self.logger.info(f"  [DISQUALIFIED] {symbol}: No regular session data found for today.")
                    continue

                open_price = regular_session_df.iloc[0]['Open']
                prev_vah = self.prev_day_stats[symbol]['vah']

                if open_price <= prev_vah:
                    self.disqualified_today.add(symbol)
                    self.logger.info(f"  [DISQUALIFIED] {symbol}: Regular Open ({open_price:.2f}) was not above prior VAH ({prev_vah:.2f}).")
                else:
                    opening_bar = regular_session_df.iloc[0]
                    self.opening_bar_info[symbol] = {'open': opening_bar['Open'], 'high': opening_bar['High'], 'low': opening_bar['Low']}
                    self.logger.info(f"  [SETUP OK] {symbol} opened regular session at {open_price:.2f}, above prior VAH.")

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        for symbol, current_bar in current_bar_data.items():
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]

                trailing_trigger_price = trade['entry_price'] + (trade['risk_per_share'] * self.trailing_stop_r)
                if current_bar['High'] >= trailing_trigger_price:
                    bars_since_entry = session_bars[symbol].loc[session_bars[symbol].index >= trade['entry_timestamp']]
                    if not bars_since_entry.empty:
                        trailing_profiler = self.ProfilerClass(bars_since_entry, tick_size=self.tick_size)
                        if trailing_profiler.poc_price:
                            new_stop_loss = trailing_profiler.poc_price - self.tick_size
                            if new_stop_loss > trade['stop_loss']:
                                self.logger.info(f"  [ADJUST] Trailing stop for {symbol} moved up to {new_stop_loss:.2f} (below POC of {trailing_profiler.poc_price:.2f})")
                                trade['stop_loss'] = new_stop_loss

                if current_bar.name.time() >= pd.to_datetime('15:55:00').time():
                    self.logger.info(f"  [EXIT] EOD for {symbol} at {current_bar['Close']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], current_bar['Close'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                elif current_bar['Low'] <= trade['stop_loss']:
                    self.logger.info(f"  [EXIT] Stop Loss for {symbol} at {trade['stop_loss']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], trade['stop_loss'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                elif current_bar['High'] >= trade['take_profit']:
                    self.logger.info(f"  [EXIT] Take Profit for {symbol} at {trade['take_profit']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], trade['take_profit'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                continue

            if symbol in self.disqualified_today or symbol not in self.opening_bar_info:
                continue

            opening_info = self.opening_bar_info[symbol]
            has_dipped = session_bars[symbol]['Low'].min() < opening_info['open']

            if has_dipped and current_bar['Close'] > opening_info['high']:
                entry_price = opening_info['high']
                rejection_low = session_bars[symbol]['Low'].min()
                risk_per_share = entry_price - rejection_low

                if risk_per_share <= 0:
                    self.disqualified_today.add(symbol)
                    continue

                equity = self.ledger.get_total_equity(market_prices)
                risk_quantity = int((equity * self.risk_per_trade_pct) / risk_per_share)
                if risk_quantity == 0:
                    self.disqualified_today.add(symbol)
                    continue

                max_value = equity * self.max_allocation_pct
                alloc_quantity = int(max_value / entry_price)
                if alloc_quantity == 0:
                    self.disqualified_today.add(symbol)
                    continue

                quantity = min(risk_quantity, alloc_quantity)
                available_cash = self.ledger.cash
                cash_quantity = int(available_cash / entry_price)

                if quantity > cash_quantity:
                    self.logger.info(f"  [SIZING] Insufficient cash. Scaling down from {quantity} to {cash_quantity} shares.")
                    quantity = cash_quantity

                if quantity == 0:
                    continue

                self.logger.info(f">>> TRIGGER for {symbol} at {current_bar.name.time()} <<<")
                self.logger.info(f"  [EXECUTE] Entering LONG {symbol} | Qty: {quantity} @ {entry_price:.2f}")
                trade_successful = self.ledger.record_trade(
                    current_bar.name, symbol, quantity, entry_price, 'BUY', market_prices
                )
                if trade_successful:
                    stop_loss_price = rejection_low
                    take_profit_price = entry_price + (risk_per_share * self.take_profit_r)
                    self.active_trades[symbol] = {
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'risk_per_share': risk_per_share,
                        'entry_timestamp': current_bar.name
                    }
                self.disqualified_today.add(symbol)