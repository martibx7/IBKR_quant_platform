# strategies/volume_accumulation_strategy.py

import pandas as pd
import pytz
import logging

from .base import BaseStrategy
from analytics.profiles import VolumeProfiler

class VolumeAccumulationStrategy(BaseStrategy):
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        super().__init__(symbols, ledger, config, **params)
        self.min_price            = self.params.get('min_price', 5.00)
        self.max_price            = self.params.get('max_price', 100.00)
        self.consol_days          = self.params.get('consolidation_days', 10)
        self.consol_range_pct     = self.params.get('consolidation_range_pct', 0.07)
        self.breakout_vol_ratio   = self.params.get('breakout_volume_ratio', 1.5)
        self.risk_pct             = self.params.get('risk_per_trade_pct', 0.01)
        self.profit_target_r      = self.params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r  = self.params.get('breakeven_trigger_r', 0.75)
        self.timezone             = pytz.timezone(self.params.get('timezone', 'America/New_York'))
        self.tick_size            = self.config['backtest'].get('tick_size_volume_profile', 0.01)
        self.candidates           = {}

    def get_required_lookback(self) -> int:
        return self.consol_days + 5

    def on_new_day(self, trade_date: pd.Timestamp, data: dict[str, pd.DataFrame]):
        super().on_new_day(trade_date, data)
        self.logger.info(f"--- SCAN FOR {trade_date.date()} ---")
        self.scan_for_candidates(trade_date)

    def scan_for_candidates(self, scan_date: pd.Timestamp):
        if not self.data_for_day:
            return

        # merge all symbols
        df_all = pd.concat(self.data_for_day.values(), copy=False)
        if df_all.empty:
            return

        df_all['dt'] = df_all.index.date
        bd = scan_date.date()
        today   = df_all[df_all['dt'] == bd]
        prior   = df_all[df_all['dt'] <  bd]
        if today.empty or prior.empty:
            return

        # aggregate on lowercase cols
        last_close      = today.groupby('symbol')['close'].last()
        breakout_vol    = today.groupby('symbol')['volume'].sum()
        avg_vol         = prior.groupby('symbol')['volume'].mean()
        grp             = prior.groupby('symbol')
        chigh           = grp['high'].max()
        clow            = grp['low'].min()
        crange_pct      = (chigh - clow) / clow

        metrics = pd.DataFrame({
            'last_close':     last_close,
            'breakout_vol':   breakout_vol,
            'avg_vol':        avg_vol,
            'consol_high':    chigh,
            'consol_low':     clow,
            'consol_range':   crange_pct
        }).dropna()

        if metrics.empty:
            return

        mask_price = metrics['last_close'].between(self.min_price, self.max_price)
        mask_range = metrics['consol_range'] <= self.consol_range_pct
        mask_vol   = metrics['breakout_vol'] > metrics['avg_vol'] * self.breakout_vol_ratio
        mask_break = metrics['last_close'] > metrics['consol_high']
        picks      = metrics[mask_price & mask_range & mask_vol & mask_break].index.tolist()

        self.logger.info(f"Found {len(picks)} raw candidates.")
        if not picks:
            return

        prof = VolumeProfiler(self.tick_size)
        valid = 0
        for sym in picks:
            df_prior = prior[prior['symbol'] == sym]
            profile  = prof.calculate(df_prior)
            if profile:
                self.candidates[sym] = {
                    'status':            'identified',
                    'poc':               profile['poc_price'],
                    'vah':               profile['value_area_high'],
                    'val':               profile['value_area_low'],
                    'confirm_high':      0.0,
                    'stop_loss':         0.0,
                }
                valid += 1

        self.logger.info(f"Profiled {valid} candidates.")

    def on_bar(self, symbol: str, bar: pd.Series):
        super().on_bar(symbol, bar)
        if symbol in self.candidates and symbol not in self.active_trades:
            self._maybe_enter(symbol, bar)
        if symbol in self.active_trades:
            self._check_exit(symbol, bar)

    def _maybe_enter(self, sym: str, bar: pd.Series):
        c = self.candidates[sym]
        if c['status'] == 'identified' and bar['low'] <= c['poc']:
            c['status']       = 'triggered'
            c['confirm_high'] = bar['high']
            c['stop_loss']    = bar['low']
            self.logger.info(f"[SETUP] {sym} hit POC {c['poc']:.2f}")
        elif c['status'] == 'triggered' and bar['high'] > c['confirm_high']:
            entry = c['confirm_high']
            sl    = c['stop_loss']
            rps   = entry - sl
            if rps <= 0:
                del self.candidates[sym]
                return
            target = entry + rps * self.profit_target_r
            equity = self.ledger.get_total_equity(self.current_prices)
            risk   = equity * self.risk_pct
            qty    = int(risk / rps)
            if qty > 0:
                self.logger.info(f"[ENTRY] {sym} @ {entry:.2f}")
                if self.ledger.record_trade(bar.name, sym, qty, entry, 'BUY', self.current_prices):
                    self.active_trades[sym] = {
                        'entry_price': entry,
                        'stop_loss':   sl,
                        'profit_target': target,
                        'quantity':    qty,
                        'r_per_share': rps,
                        'breakeven':   False
                    }
            del self.candidates[sym]

    def _check_exit(self, sym: str, bar: pd.Series):
        t = self.active_trades[sym]
        # stop
        if bar['low'] <= t['stop_loss']:
            self.logger.info(f"[EXIT] {sym} SL @ {t['stop_loss']:.2f}")
            self.ledger.close_trade(bar.name, sym, t['stop_loss'], 'Stop Loss', self.current_prices)
            del self.active_trades[sym]
            return
        # profit target
        if bar['high'] >= t['profit_target']:
            self.logger.info(f"[EXIT] {sym} PT @ {t['profit_target']:.2f}")
            self.ledger.close_trade(bar.name, sym, t['profit_target'], 'Profit Target', self.current_prices)
            del self.active_trades[sym]
            return
        # breakeven
        if not t['breakeven']:
            be = t['entry_price'] + t['r_per_share'] * self.breakeven_trigger_r
            if bar['high'] >= be:
                t['stop_loss'] = t['entry_price']
                t['breakeven'] = True
                self.logger.info(f"[BREAKEVEN] {sym} SL -> {t['entry_price']:.2f}")

    def on_session_end(self):
        if not self.active_trades:
            return
        for sym in list(self.active_trades):
            df = self.data_for_day.get(sym)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                ep   = last['close']
                tm   = last.name
                self.logger.info(f"[EOD EXIT] {sym} @ {ep:.2f}")
                self.ledger.close_trade(tm, sym, ep, 'EOD', self.current_prices)
                del self.active_trades[sym]
