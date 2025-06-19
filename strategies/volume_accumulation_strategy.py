# strategies/volume_accumulation_strategy.py

import pandas as pd
import pytz
import logging
from datetime import time

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler

class VolumeAccumulationStrategy(BaseStrategy):
    """
    Identifies multi-day consolidations with high-volume breakouts, and enters on a
    re-test of the consolidation's Point of Control (POC) after confirmation.
    """
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        super().__init__(symbols, ledger, config, **params)
        self.min_price               = self.params.get('min_price', 5.00)
        self.max_price               = self.params.get('max_price', 100.00)
        self.consolidation_days      = self.params.get('consolidation_days', 10)
        self.consolidation_range_pct = self.params.get('consolidation_range_pct', 0.07)
        self.breakout_volume_ratio   = self.params.get('breakout_volume_ratio', 1.5)
        self.risk_per_trade_pct      = self.params.get('risk_per_trade_pct', 0.01)
        self.max_allocation_pct      = self.params.get('max_allocation_pct', 0.25)
        self.profit_target_r         = self.params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r     = self.params.get('breakeven_trigger_r', 0.75)
        self.timezone                = pytz.timezone(self.params.get('timezone', 'America/New_York'))
        self.tick_size               = self.config['backtest'].get('tick_size_volume_profile', 0.01)
        self.candidates              = {}

    def get_required_lookback(self) -> int:
        # If testing for 0 consolidation days, we still need at least 1 day for volume comparison.
        return max(1, self.consolidation_days)

    def on_new_day(self, trade_date: pd.Timestamp, data: dict[str, pd.DataFrame]):
        super().on_new_day(trade_date, data)
        self.logger.info(f"--- SCAN FOR {trade_date.date()} ---")
        self.scan_for_candidates(trade_date)

    def scan_for_candidates(self, scan_date: pd.Timestamp):
        if not self.data_for_day:
            return

        self.candidates.clear()

        breakout_date = scan_date.tz_convert('UTC').date()

        for symbol, df in self.data_for_day.items():
            day_data = df[df.index.date == breakout_date]
            # Use all data before the breakout day for the lookback period
            consol_data = df[df.index.date < breakout_date]

            if day_data.empty: continue

            # --- Filter by price first ---
            last_close = day_data['close'].iloc[-1]
            if not self.min_price <= last_close <= self.max_price: continue

            # --- NEW: Logic for handling both consolidation and no-consolidation tests ---
            is_candidate = False
            poc_data = pd.DataFrame()
            stop_loss_price = 0

            if self.consolidation_days == 0:
                # --- NO-CONSOLIDATION LOGIC: Look for a simple volume spike ---
                if consol_data.empty: continue # Need at least one prior day

                prev_day_volume = consol_data[consol_data.index.date == consol_data.index.date.max()]['volume'].sum()
                current_day_volume = day_data['volume'].sum()

                if prev_day_volume > 0 and current_day_volume > (prev_day_volume * self.breakout_volume_ratio):
                    is_candidate = True
                    poc_data = day_data # Use today's data for POC
                    stop_loss_price = day_data['low'].min()
            else:
                # --- STANDARD CONSOLIDATION LOGIC ---
                if consol_data.empty: continue

                # Check for sufficient historical days for a valid consolidation
                if len(consol_data.index.date.unique()) < self.consolidation_days: continue

                consol_high = consol_data['high'].max()
                consol_low = consol_data['low'].min()
                if consol_low == 0: continue

                consol_range_pct = (consol_high - consol_low) / consol_low
                if consol_range_pct > self.consolidation_range_pct: continue

                breakout_volume = day_data['volume'].sum()
                avg_daily_volume = consol_data.groupby(consol_data.index.date)['volume'].sum().mean()

                if breakout_volume <= (avg_daily_volume * self.breakout_volume_ratio): continue

                if last_close > consol_high:
                    is_candidate = True
                    poc_data = consol_data # Use consolidation data for POC
                    stop_loss_price = consol_low

            # If any logic path found a candidate, calculate its profile
            if is_candidate:
                profiler = VolumeProfiler(self.tick_size)
                profile = profiler.calculate(poc_data)
                if profile:
                    self.candidates[symbol] = {
                        'status': 'watching',
                        'poc': profile['poc_price'],
                        'stop_loss': stop_loss_price,
                    }

        self.logger.info(f"Scan found {len(self.candidates)} valid candidates.")

    def on_bar(self, symbol: str, bar: pd.Series):
        super().on_bar(symbol, bar)
        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
        elif symbol in self.candidates:
            self._handle_candidate(symbol, bar)

    def _handle_candidate(self, symbol: str, bar: pd.Series):
        candidate = self.candidates.get(symbol)
        if not candidate: return

        if candidate['status'] == 'watching' and bar['low'] <= candidate['poc']:
            candidate['status'] = 'triggered'
            candidate['confirmation_high'] = bar['high']
            self.logger.info(f"  [SETUP] {symbol} touched POC {candidate['poc']:.2f}. Waiting for confirmation above {bar['high']:.2f}.")

        elif candidate['status'] == 'triggered' and bar['high'] > candidate['confirmation_high']:
            entry_price = candidate['confirmation_high']
            stop_loss_price = candidate['stop_loss']
            risk_per_share = entry_price - stop_loss_price

            if risk_per_share <= 0:
                if symbol in self.candidates: del self.candidates[symbol]
                return

            equity = self.ledger.get_total_equity(self.current_prices)
            dollar_risk = equity * self.risk_per_trade_pct
            quantity = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

            if quantity > 0:
                self.logger.info(f"  [ENTRY] {symbol}: Confirmed above {entry_price:.2f}.")
                if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                    profit_target_price = entry_price + (risk_per_share * self.profit_target_r)
                    self.active_trades[symbol] = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss_price,
                        'profit_target': profit_target_price,
                        'quantity': quantity,
                        'risk_per_share': risk_per_share
                    }
            if symbol in self.candidates:
                del self.candidates[symbol]

    def _manage_active_trade(self, symbol: str, bar: pd.Series):
        trade = self.active_trades.get(symbol)
        if not trade: return

        if not trade.get('is_breakeven', False):
            breakeven_trigger_price = trade['entry_price'] + (trade['risk_per_share'] * self.breakeven_trigger_r)
            if bar['high'] >= breakeven_trigger_price:
                trade['stop_loss'] = trade['entry_price']
                trade['is_breakeven'] = True
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")

        exit_price, reason = None, None
        if bar['low'] <= trade['stop_loss']:
            exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['high'] >= trade.get('profit_target', float('inf')):
            exit_price, reason = trade['profit_target'], "Profit Target"
        elif bar.name.time() >= time(15, 50):
            exit_price, reason = bar['close'], "End of Day"

        if exit_price and reason:
            self.logger.info(f"  [EXIT] {symbol} @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices, exit_reason=reason)
            if symbol in self.active_trades:
                del self.active_trades[symbol]

    def on_session_end(self):
        pass