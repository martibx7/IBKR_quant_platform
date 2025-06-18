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
        self.profit_target_r         = self.params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r     = self.params.get('breakeven_trigger_r', 0.75)
        self.timezone                = pytz.timezone(self.params.get('timezone', 'America/New_York'))
        self.tick_size               = self.config['backtest'].get('tick_size_volume_profile', 0.01)
        self.candidates              = {}

    def get_required_lookback(self) -> int:
        return self.consolidation_days + 5

    def on_new_day(self, trade_date: pd.Timestamp, data: dict[str, pd.DataFrame]):
        super().on_new_day(trade_date, data)
        self.logger.info(f"--- VECTORIZED SCAN FOR {trade_date.date()} ---")
        self.scan_for_candidates(trade_date)

    def scan_for_candidates(self, scan_date: pd.Timestamp):
        if not self.data_for_day:
            return

        # concatenate all symbols' intraday bars
        all_symbols_df = pd.concat(self.data_for_day.values())
        if all_symbols_df.empty:
            return

        # assign date and split into consolidation vs breakout day
        all_symbols_df['date_obj'] = all_symbols_df.index.date
        breakout_date   = scan_date.date()
        day_data        = all_symbols_df[all_symbols_df['date_obj'] == breakout_date]
        consol_data     = all_symbols_df[all_symbols_df['date_obj'] < breakout_date]

        if day_data.empty or consol_data.empty:
            return

        # compute per-symbol metrics on uppercase columns
        last_closes       = day_data.groupby('symbol')['Close'].last()
        breakout_volumes  = day_data.groupby('symbol')['Volume'].sum()
        avg_volumes       = consol_data.groupby('symbol')['Volume'].mean()
        grp_consol        = consol_data.groupby('symbol')
        consol_high       = grp_consol['High'].max()
        consol_low        = grp_consol['Low'].min()
        consol_range_pct  = (consol_high - consol_low) / consol_low

        metrics = pd.DataFrame({
            'last_close':      last_closes,
            'breakout_volume': breakout_volumes,
            'avg_volume':      avg_volumes,
            'consol_high':     consol_high,
            'consol_low':      consol_low,
            'consol_range':    consol_range_pct
        }).dropna()

        if metrics.empty:
            return

        # apply your filters
        price_mask        = metrics['last_close'].between(self.min_price, self.max_price)
        consolidation_ok  = metrics['consol_range'] <= self.consolidation_range_pct
        volume_ok         = metrics['breakout_volume'] > metrics['avg_volume'] * self.breakout_volume_ratio
        breakout_above    = metrics['last_close'] > metrics['consol_high']
        final_mask        = price_mask & consolidation_ok & volume_ok & breakout_above

        candidates = metrics[final_mask].index.tolist()
        self.logger.info(f"Vectorized scan found {len(candidates)} candidates.")

        if not candidates:
            return

        # calculate POC/VAH/VAL for each candidate
        profiler = VolumeProfiler(self.tick_size)
        valid = 0
        for sym in candidates:
            sym_consol = consol_data[consol_data['symbol'] == sym]
            profile = profiler.calculate(sym_consol)
            if profile:
                self.candidates[sym] = {
                    'status':             'identified',
                    'poc':                profile['poc_price'],
                    'vah':                profile['value_area_high'],
                    'val':                profile['value_area_low'],
                    'confirmation_high':  0,
                    'stop_loss':          0,
                }
                valid += 1

        self.logger.info(f"Processed POC for {valid} valid candidates.")

    def on_bar(self, symbol: str, bar: pd.Series):
        super().on_bar(symbol, bar)
        if symbol in self.candidates and symbol not in self.active_trades:
            self._handle_candidate_entry(symbol, bar)
        if symbol in self.active_trades:
            self.on_exit_conditions(symbol, bar)

    def _handle_candidate_entry(self, symbol: str, bar: pd.Series):
        c = self.candidates[symbol]
        if c['status'] == 'identified' and bar['Low'] <= c['poc']:
            c['status'] = 'triggered'
            c['confirmation_high'] = bar['High']
            c['stop_loss'] = bar['Low']
            self.logger.info(f"  [SETUP] {symbol} touched POC {c['poc']:.2f}. Waiting above {bar['High']:.2f}.")
        elif c['status'] == 'triggered' and bar['High'] > c['confirmation_high']:
            entry  = c['confirmation_high']
            sl     = c['stop_loss']
            r_ps   = entry - sl
            if r_ps <= 0:
                del self.candidates[symbol]
                return
            target = entry + r_ps * self.profit_target_r
            equity = self.ledger.get_total_equity(self.current_prices)
            risk   = equity * self.risk_per_trade_pct
            qty    = int(risk / r_ps)
            if qty > 0:
                self.logger.info(f"  [ENTRY] {symbol} @ {entry:.2f}")
                if self.ledger.record_trade(bar.name, symbol, qty, entry, 'BUY', self.current_prices):
                    self.active_trades[symbol] = {
                        'entry_price':       entry,
                        'stop_loss':         sl,
                        'profit_target':     target,
                        'quantity':          qty,
                        'risk_per_share':    r_ps,
                        'breakeven_triggered': False
                    }
            del self.candidates[symbol]

    def on_exit_conditions(self, symbol: str, bar: pd.Series):
        t = self.active_trades.get(symbol)
        if not t:
            return

        # stop‐loss
        if bar['Low'] <= t['stop_loss']:
            self.logger.info(f"  [EXIT] {symbol} @ {t['stop_loss']:.2f} (Stop Loss)")
            self.ledger.close_trade(bar.name, symbol, t['stop_loss'], "Stop Loss", self.current_prices)
            del self.active_trades[symbol]
            return

        # profit‐target
        if bar['High'] >= t['profit_target']:
            self.logger.info(f"  [EXIT] {symbol} @ {t['profit_target']:.2f} (Profit Target)")
            self.ledger.close_trade(bar.name, symbol, t['profit_target'], "Profit Target", self.current_prices)
            del self.active_trades[symbol]
            return

        # move to breakeven
        if not t['breakeven_triggered']:
            be_price = t['entry_price'] + t['risk_per_share'] * self.breakeven_trigger_r
            if bar['High'] >= be_price:
                t['stop_loss'] = t['entry_price']
                t['breakeven_triggered'] = True
                self.logger.info(f"  [BREAKEVEN] {symbol} stop moved to {t['entry_price']:.2f}")

    def on_session_end(self):
        """Exit any open trades at the last bar's Close."""
        if not self.active_trades:
            return

        for sym in list(self.active_trades):
            if sym in self.data_for_day and not self.data_for_day[sym].empty:
                last = self.data_for_day[sym].iloc[-1]
                exit_price = last['Close']
                exit_time  = last.name
                self.logger.info(f"  [EXIT] {sym} @ {exit_price:.2f} (EOD)")
                self.ledger.close_trade(exit_time, sym, exit_price, "End of Day", self.current_prices)
                del self.active_trades[sym]
