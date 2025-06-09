# strategies/open_rejection_reverse_strategy.py

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, get_session
import pytz
import logging
import os

class OpenRejectionReverseStrategy(BaseStrategy):
    """
    Implements a refined, multi-stage Open Rejection Reversal strategy based on
    market-generated information principles. This is a long-only strategy.

    Stage 1: The Scanner (End-of-Day)
        - Finds stocks with significant volume and range on the prior day.
        - Looks for a close in the upper quartile of the day's range and above the POC,
          indicating strong closing momentum.

    Stage 2: The Setup (At the Open)
        - Confirms the candidate stock has opened above the prior day's Value Area High (VAH),
          signaling acceptance of higher prices.

    Stage 3: The Trigger (Intraday)
        - Watches for an initial dip below the opening price.
        - The trigger is a rally back up through the high of the opening 1-minute bar.

    Stage 4: Trade Management
        - Enters on the trigger, with a stop-loss at the low of the rejection.
        - Position size is calculated based on a fixed percentage of equity risk.
        - Exits at a pre-defined R-multiple profit target or at the end of the day.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters from config.yaml ---
        self.risk_per_trade_pct = kwargs.get('risk_per_trade_pct', 0.01)
        self.take_profit_r = kwargs.get('take_profit_r', 2.5)
        self.min_daily_volume = kwargs.get('min_daily_volume', 500000)
        self.min_prev_day_range_pct = kwargs.get('min_prev_day_range_pct', 0.03)
        self.log_file = kwargs.get('log_file', 'logs/open_rejection_reverse.log')
        self.tz_str = kwargs.get('timezone', 'America/New_York')

        # --- Strategy State ---
        self.timezone = pytz.timezone(self.tz_str)
        self.prev_day_stats = {}
        self.active_trades = {}
        self.disqualified_today = set()
        self.opening_bar_info = {}

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
        self.logger.info("Strategy logger initialized.")

    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        """Stage 1: Finds stocks that closed strong on the prior day."""
        self.logger.info(f"--- SCANNING CANDIDATES FOR {trade_date.strftime('%Y-%m-%d')} ---")
        candidates = []
        for symbol, df in historical_data.items():
            if df.empty or len(df) < 2: continue

            # Filter by volume and range
            vol = df['Volume'].sum()
            day_high = df['High'].max()
            day_low = df['Low'].min()
            day_range = (day_high - day_low) / day_low if day_low > 0 else 0

            if vol < self.min_daily_volume or day_range < self.min_prev_day_range_pct:
                continue

            # Check for strong close
            profiler = VolumeProfiler(df, tick_size=0.01)
            day_close = df.iloc[-1]['Close']

            if profiler.poc_price and profiler.vah and day_close > profiler.poc_price:
                if day_close > (day_high - 0.25 * (day_high - day_low)):
                    self.prev_day_stats[symbol] = {'vah': profiler.vah, 'poc': profiler.poc_price}
                    candidates.append(symbol)
                    self.logger.info(f"  [CANDIDATE] {symbol} | Close: {day_close:.2f} > POC: {profiler.poc_price:.2f} | VAH: {profiler.vah:.2f}")

        self.logger.info(f"Found {len(candidates)} candidates.")
        return candidates

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        """Stage 2: Check the opening price against the prior day's value area."""
        self.logger.info(f"--- NEW SESSION ---")
        self.active_trades.clear()
        self.disqualified_today.clear()
        self.opening_bar_info.clear()

        for symbol, df in session_data.items():
            if symbol in self.prev_day_stats:
                open_price = df.iloc[0]['Open']
                prev_vah = self.prev_day_stats[symbol]['vah']

                if open_price <= prev_vah:
                    self.disqualified_today.add(symbol)
                    self.logger.info(f"  [DISQUALIFIED] {symbol}: Open ({open_price:.2f}) was not above prior VAH ({prev_vah:.2f}).")
                else:
                    self.opening_bar_info[symbol] = {
                        'open': df.iloc[0]['Open'],
                        'high': df.iloc[0]['High'],
                        'low': df.iloc[0]['Low']
                    }
                    self.logger.info(f"  [SETUP OK] {symbol} opened at {open_price:.2f}, above prior VAH.")

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        """Stages 3 & 4: Manages entry triggers, exits, and risk."""
        for symbol, current_bar in current_bar_data.items():
            # --- EXIT LOGIC ---
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
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

            # --- ENTRY LOGIC ---
            if symbol in self.disqualified_today or symbol not in self.opening_bar_info:
                continue

            # Stage 3: The Trigger
            opening_info = self.opening_bar_info[symbol]
            has_dipped = session_bars[symbol]['Low'].min() < opening_info['open']

            if has_dipped and current_bar['Close'] > opening_info['high']:
                entry_price = opening_info['high']
                rejection_low = session_bars[symbol]['Low'].min()

                # --- Stage 4: Execution & Risk Management ---
                risk_per_share = entry_price - rejection_low
                if risk_per_share <= 0:
                    self.disqualified_today.add(symbol) # Prevent further attempts
                    continue

                equity = self.ledger.cash
                risk_amount = equity * self.risk_per_trade_pct
                quantity = int(risk_amount / risk_per_share)

                if quantity == 0:
                    self.disqualified_today.add(symbol) # Prevent further attempts
                    continue

                self.logger.info(f">>> TRIGGER for {symbol} at {current_bar.name.time()} <<<")
                self.logger.info(f"  [COND] Dipped below open and rallied above opening bar high.")
                self.logger.info(f"  [EXECUTE] Entering LONG {symbol} | Qty: {quantity} @ {entry_price:.2f}")

                trade_successful = self.ledger.record_trade(
                    current_bar.name, symbol, quantity, entry_price, 'BUY', market_prices
                )

                if trade_successful:
                    stop_loss_price = rejection_low
                    take_profit_price = entry_price + (risk_per_share * self.take_profit_r)
                    self.active_trades[symbol] = {
                        'quantity': quantity,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price
                    }

                # Disqualify from further trades today to avoid re-entry
                self.disqualified_today.add(symbol)