# strategies/volume_accumulation_strategy.py

import pandas as pd
import pytz
import logging
import os
from datetime import time

from .base import BaseStrategy
from analytics.profiles import VolumeProfiler

class VolumeAccumulationStrategy(BaseStrategy):
    """
    Identifies multi-day consolidations with high-volume breakouts within a specific price range,
    and enters on a re-test of the consolidation's Point of Control (POC) after confirmation.
    """
    def __init__(self, symbols: list[str], ledger, **params):
        super().__init__(symbols, ledger, **params)

        # --- NEW: Added price range parameters ---
        self.min_price = params.get('min_price', 5.00)
        self.max_price = params.get('max_price', 100.00)

        self.consolidation_days = params.get('consolidation_days', 10)
        self.consolidation_range_pct = params.get('consolidation_range_pct', 0.07)
        self.breakout_volume_ratio = params.get('breakout_volume_ratio', 1.5)
        self.risk_per_trade_pct = params.get('risk_per_trade_pct', 0.01)
        self.profit_target_r = params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r = params.get('breakeven_trigger_r', 0.75)

        self.timezone = pytz.timezone(params.get('timezone', 'America/New_York'))
        self.tick_size = params.get('tick_size', 0.01)

        self.candidates = {}
        self.active_trades = {}
        self.current_prices = {}

        self._setup_logger(params.get('log_file', 'logs/volume_accumulation.log'))

    def _setup_logger(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.propagate = False
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def scan_for_candidates(self, all_data: dict[str, pd.DataFrame], trade_date: pd.Timestamp) -> list[str]:
        self.logger.info(f"--- SCANNING CANDIDATES FOR {trade_date.strftime('%Y-%m-%d')} ---")
        self.candidates.clear()

        scan_end_date = trade_date
        scan_start_date = scan_end_date - pd.Timedelta(days=self.consolidation_days + 5)

        for symbol, df in all_data.items():
            if df.empty: continue

            recent_data = df[(df.index.date >= scan_start_date) & (df.index.date <= scan_end_date)]
            unique_days = recent_data.index.normalize().unique()

            if len(unique_days) < self.consolidation_days: continue

            last_day_data = recent_data[recent_data.index.date == scan_end_date]
            if last_day_data.empty: continue

            # --- NEW: Price Filter ---
            last_close = last_day_data.iloc[-1]['Close']
            if not (self.min_price <= last_close <= self.max_price):
                continue
            # --- END NEW FILTER ---

            consolidation_days_in_data = unique_days[unique_days < pd.to_datetime(scan_end_date, utc=True)]
            if len(consolidation_days_in_data) < self.consolidation_days: continue

            consolidation_data = recent_data[recent_data.index.normalize().isin(consolidation_days_in_data)]
            if consolidation_data.empty: continue

            consolidation_high = consolidation_data['High'].max()
            consolidation_low = consolidation_data['Low'].min()

            if not consolidation_low > 0 or (consolidation_high - consolidation_low) / consolidation_low > self.consolidation_range_pct:
                continue

            if last_close <= consolidation_high:
                continue

            avg_consolidation_volume = consolidation_data['Volume'].sum() / len(consolidation_data.index.normalize().unique())
            breakout_volume = last_day_data['Volume'].sum()

            if avg_consolidation_volume > 0 and breakout_volume < (avg_consolidation_volume * self.breakout_volume_ratio):
                continue

            profiler = VolumeProfiler(self.tick_size)
            profile = profiler.calculate(consolidation_data)

            if not profile or profile.get('poc_price') is None: continue

            poc = profile['poc_price']
            self.candidates[symbol] = {
                'poc': poc,
                'stop_loss': consolidation_low,
                'status': 'watching',
                'profit_target_level': last_day_data['High'].max()
            }
            self.logger.info(f"  [CANDIDATE] {symbol}: Passed all filters. POC: {poc:.2f}, SL: {consolidation_low:.2f}, PT Level: {self.candidates[symbol]['profit_target_level']:.2f}")

        return list(self.candidates.keys())

    # --- on_session_start, on_bar, and other methods remain unchanged ---

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        self.logger.info("--- NEW SESSION STARTED ---")
        self.active_trades.clear()
        self.current_prices = {s: df.iloc[0]['Open'] for s, df in session_data.items() if not df.empty}

    def on_bar(self, symbol: str, bar: pd.Series):
        if symbol not in self.current_prices: return
        self.current_prices[symbol] = bar['Close']

        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
            return

        if symbol in self.candidates:
            self._handle_candidate(symbol, bar)

    def _manage_active_trade(self, symbol: str, bar: pd.Series):
        trade = self.active_trades[symbol]

        if not trade.get('is_breakeven', False):
            # Corrected breakeven logic to use profit_target_level
            profit_potential = trade['profit_target'] - trade['entry_price']
            breakeven_trigger_price = trade['entry_price'] + (profit_potential * self.breakeven_trigger_r)
            if bar['High'] >= breakeven_trigger_price:
                trade['stop_loss'] = trade['entry_price']
                trade['is_breakeven'] = True
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")

        exit_price, reason = None, None
        if bar['Low'] <= trade['stop_loss']:
            exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['High'] >= trade['profit_target']:
            exit_price, reason = trade['profit_target'], "Profit Target"
        elif bar.name.time() >= time(15, 50):
            exit_price, reason = bar['Close'], "End of Day"

        if exit_price:
            self.logger.info(f"  [EXIT] {symbol} @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices)
            del self.active_trades[symbol]

    def _handle_candidate(self, symbol: str, bar: pd.Series):
        candidate = self.candidates[symbol]

        if candidate['status'] == 'watching' and bar['Low'] <= candidate['poc']:
            candidate['status'] = 'triggered'
            candidate['confirmation_high'] = bar['High']
            self.logger.info(f"  [SETUP] {symbol} touched POC {candidate['poc']:.2f}. Waiting for confirmation above {bar['High']:.2f}.")

        elif candidate['status'] == 'triggered' and bar['High'] > candidate['confirmation_high']:
            entry_price = candidate['confirmation_high']
            stop_loss_price = candidate['stop_loss']
            profit_target_price = candidate['profit_target_level']
            risk_per_share = entry_price - stop_loss_price

            if risk_per_share <= 0 or entry_price >= profit_target_price:
                del self.candidates[symbol]
                return

            equity = self.ledger.get_total_equity(self.current_prices)
            dollar_risk = equity * self.risk_per_trade_pct
            quantity = int(dollar_risk / risk_per_share)

            cost = quantity * entry_price
            available_cash = self.ledger.cash
            if cost > available_cash:
                new_quantity = int(available_cash / entry_price)
                self.logger.info(f"  [SCALE DOWN] {symbol}: Not enough cash for {quantity} shares. Scaling down to {new_quantity}.")
                quantity = new_quantity

            if quantity == 0:
                del self.candidates[symbol]
                return

            self.logger.info(f"  [ENTRY] {symbol}: Confirmed above {entry_price:.2f}.")
            if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                self.active_trades[symbol] = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'profit_target': profit_target_price,
                    'quantity': quantity,
                    'risk_per_share': risk_per_share
                }
                self.logger.info(f"    -> Trade logged: Qty {quantity} @ {entry_price:.2f}, SL {stop_loss_price:.2f}, PT {profit_target_price:.2f}")

            del self.candidates[symbol]

    def on_session_end(self):
        pass