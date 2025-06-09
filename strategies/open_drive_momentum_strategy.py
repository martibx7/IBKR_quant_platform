import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from analytics.profiles import MarketProfiler, VolumeProfiler, get_session
from analytics.indicators import calculate_vwap
import pytz
import logging
import os

class OpenDriveMomentumStrategy(BaseStrategy):
    """
    Implements the Session-Aware Open-Drive Momentum Strategy with robust risk management
    and detailed logging.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # Strategy parameters
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.01)
        self.max_allocation_pct = kwargs.get('max_allocation_pct', 0.25)
        self.min_risk_per_share = kwargs.get('min_risk_per_share', 0.05)
        self.entry_time_str = kwargs.get('entry_time', '10:00:00')
        self.exit_time_str = kwargs.get('exit_time', '15:55:00')
        self.tz_str = kwargs.get('timezone', 'America/New_York')
        self.log_file = kwargs.get('log_file', 'logs/open_drive_momentum.log')

        # Timezone and time conversions
        self.timezone = pytz.timezone(self.tz_str)
        self.entry_time = pd.to_datetime(self.entry_time_str).time()
        self.exit_time = pd.to_datetime(self.exit_time_str).time()

        # Strategy state variables
        self.prev_day_stats = {}
        self.position_details = {}
        self.trade_attempted_today = set()

        # --- Set up a dedicated logger for this strategy ---
        self._setup_logger()

    def _setup_logger(self):
        """Initializes a file-based logger for the strategy."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger(f"OpenDriveMomentumStrategy.{id(self)}")
        self.logger.setLevel(logging.INFO)

        # Prevent logs from propagating to the root logger
        self.logger.propagate = False

        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(self.log_file, mode='w') # 'w' to clear file on each run
        fh.setLevel(logging.INFO)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(fh)

    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        """Calculates previous day's stats for each symbol."""
        candidates = []
        self.logger.info(f"--- SCANNING CANDIDATES FOR {trade_date.strftime('%Y-%m-%d')} ---")
        for symbol, df in historical_data.items():
            if df.empty: continue

            df.index = pd.to_datetime(df.index, utc=True)
            regular_session_df = df[df.index.to_series().apply(
                lambda dt: get_session(dt.tz_convert(self.timezone)) == 'Regular'
            )]
            if regular_session_df.empty: continue

            prev_day_m_profiler = MarketProfiler(regular_session_df)
            if prev_day_m_profiler.poc_price is not None:
                self.prev_day_stats[symbol] = {
                    'vah': prev_day_m_profiler.vah,
                    'val': prev_day_m_profiler.val,
                    'poc': prev_day_m_profiler.poc_price
                }
                candidates.append(symbol)
                self.logger.info(f"  [SCANNER] {symbol} Prev Day Stats | VAH: {prev_day_m_profiler.vah:.2f}, POC: {prev_day_m_profiler.poc_price:.2f}")
        return candidates

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """Resets the state for the new day."""
        self.position_details = {}
        self.trade_attempted_today = set()

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        """Main event loop for the strategy."""
        for symbol in current_bar_data.keys():
            if symbol not in session_bars or symbol not in analytics: continue

            current_bar = current_bar_data[symbol]
            current_bar_time = current_bar.name.tz_convert(self.timezone).time()

            # --- EXIT LOGIC ---
            if symbol in self.position_details:
                # ... (exit logic remains the same)
                pass

            # --- ENTRY LOGIC ---
            if current_bar_time == self.entry_time and symbol not in self.position_details and symbol not in self.trade_attempted_today:
                self.trade_attempted_today.add(symbol)
                self.logger.info(f">>> Checking Entry Conditions for {symbol} at {current_bar_time} <<<")

                if symbol not in self.prev_day_stats:
                    self.logger.warning(f"  [FAIL] {symbol}: No previous day stats available.")
                    continue

                # Get all necessary data
                current_session_df = session_bars[symbol]
                prev_day = self.prev_day_stats[symbol]
                opening_bar = current_session_df.iloc[0]
                vwap_df = analytics[symbol]['vwap']
                current_price = current_bar['Close']

                # --- MODIFIED: Condition 1 now only requires open above the previous day's POC ---
                cond1_open_above_poc = opening_bar['Open'] > prev_day['poc']
                self.logger.info(f"  [COND 1] Open > Prev POC?: {cond1_open_above_poc} (Open: {opening_bar['Open']:.2f}, POC: {prev_day['poc']:.2f})")
                if not cond1_open_above_poc:
                    self.logger.info(f"  [FAIL] {symbol}: Failed Condition 1 (Did not open above prior day's Point of Control).")
                    continue

                # Condition 2: Price Above VWAP
                if vwap_df.empty:
                    self.logger.warning(f"  [FAIL] {symbol}: VWAP data is empty.")
                    continue
                latest_vwap = vwap_df.iloc[-1]['vwap']
                cond2_price_above_vwap = current_price > latest_vwap
                self.logger.info(f"  [COND 2] Price > VWAP?: {cond2_price_above_vwap} (Price: {current_price:.2f}, VWAP: {latest_vwap:.2f})")
                if not cond2_price_above_vwap:
                    self.logger.info(f"  [FAIL] {symbol}: Failed Condition 2 (Price vs VWAP).")
                    continue

                # Condition 3: POC Confirms Accumulation
                opening_drive_bars = current_session_df.iloc[0:30]
                od_profiler = VolumeProfiler(opening_drive_bars)
                if od_profiler.poc_price is None:
                    self.logger.warning(f"  [FAIL] {symbol}: Could not calculate Opening Drive POC.")
                    continue

                od_high, od_low = opening_drive_bars['High'].max(), opening_drive_bars['Low'].min()
                od_midpoint = (od_high + od_low) / 2
                cond3_poc_in_lower_half = od_profiler.poc_price <= od_midpoint
                self.logger.info(f"  [COND 3] OD POC in Lower Half?: {cond3_poc_in_lower_half} (POC: {od_profiler.poc_price:.2f}, Mid: {od_midpoint:.2f})")
                if not cond3_poc_in_lower_half:
                    self.logger.info(f"  [FAIL] {symbol}: Failed Condition 3 (POC location).")
                    continue

                self.logger.info(f"  [SUCCESS] All conditions met for {symbol}. Attempting to execute trade.")
                # --- Position Sizing ---
                entry_price = current_price
                stop_loss_price = od_low
                risk_per_share = entry_price - stop_loss_price

                if risk_per_share < self.min_risk_per_share:
                    self.logger.warning(f"  [SKIP] {symbol}: Risk per share ({risk_per_share:.2f}) is below minimum ({self.min_risk_per_share}).")
                    continue

                # --- FIX: Use total equity for risk calculation, not just cash ---
                open_positions_value = 0
                for pos_symbol, position in self.ledger.open_positions.items():
                    price = market_prices.get(pos_symbol, position['entry_price'])
                    open_positions_value += position['quantity'] * price

                equity = self.ledger.cash + open_positions_value
                risk_amount = equity * self.risk_per_trade
                quantity = int(risk_amount / risk_per_share)

                max_position_value = equity * self.max_allocation_pct
                proposed_position_value = quantity * entry_price

                if proposed_position_value > max_position_value:
                    quantity = int(max_position_value / entry_price)
                    self.logger.info(f"  [SIZING] {symbol}: Position size capped by max allocation. New quantity: {quantity}")

                if quantity > 0:
                    self.logger.info(f"  [EXECUTE] ENTRY CONFIRMED for {symbol} | Qty: {quantity}")
                    trade_successful = self.ledger.record_trade(
                        timestamp=current_bar.name, symbol=symbol, quantity=quantity,
                        price=entry_price, order_type='BUY', market_prices=market_prices
                    )
                    if trade_successful:
                        self.position_details[symbol] = {
                            'quantity': quantity, 'stop_loss': stop_loss_price, 'entry_time': current_bar.name
                        }
