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
    Identifies multi-day consolidations with high-volume breakouts, and enters on a
    re-test of the consolidation's Point of Control (POC) after confirmation.
    This version uses a highly optimized and robust vectorized candidate scan.
    """
    def __init__(self, symbols: list[str], ledger, **params):
        super().__init__(symbols, ledger, **params)
        self.params = params
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
        self._setup_logger(params.get('log_file', 'logs/volume_accumulation_strategy.log'))

    def _setup_logger(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.propagate = False
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def scan_for_candidates(self, all_data: dict[str, pd.DataFrame], scan_date: pd.Timestamp) -> list[str]:
        self.logger.info(f"--- VECTORIZED SCAN FOR {scan_date.strftime('%Y-%m-%d')} ---")
        self.candidates.clear()

        if not all_data: return []

        all_symbols_df = pd.concat(all_data.values(), keys=all_data.keys(), names=['symbol', 'timestamp_utc'])
        self.logger.debug(f"Combined DataFrame created with {len(all_symbols_df)} rows for {len(all_symbols_df.index.get_level_values('symbol').unique())} symbols.")

        start_of_scan_day = pd.to_datetime(scan_date, utc=True)
        end_of_scan_day = start_of_scan_day + pd.Timedelta(days=1)
        ts_index = all_symbols_df.index.get_level_values('timestamp_utc')

        breakout_day_data = all_symbols_df[(ts_index >= start_of_scan_day) & (ts_index < end_of_scan_day)]
        if breakout_day_data.empty: return []

        last_closes = breakout_day_data.groupby('symbol')['Close'].last()
        breakout_volumes = breakout_day_data.groupby('symbol')['Volume'].sum()

        consolidation_df = all_symbols_df[ts_index < start_of_scan_day]
        if consolidation_df.empty: return []

        grouped_consol = consolidation_df.groupby('symbol')
        consol_high = grouped_consol['High'].max()
        consol_low = grouped_consol['Low'].min()
        consol_range_pct = (consol_high - consol_low) / consol_low

        # --- LOGICAL FIX: Use nunique() for a precise day count ---
        unique_days_count = grouped_consol['date'].nunique().clip(1)
        consol_avg_volume = grouped_consol['Volume'].sum() / unique_days_count

        metrics_df = pd.DataFrame({
            'last_close': last_closes,
            'breakout_volume': breakout_volumes,
            'consol_high': consol_high,
            'consol_low': consol_low,
            'consol_range_pct': consol_range_pct,
            'consol_avg_volume': consol_avg_volume
        })
        self.logger.debug(f"Initial metrics DataFrame created with shape: {metrics_df.shape}")

        metrics_df.dropna(inplace=True)
        self.logger.debug(f"Metrics DataFrame shape after dropna(): {metrics_df.shape}")

        if metrics_df.empty:
            self.logger.info("No symbols had complete data for all required metrics.")
            return []

        price_mask = (metrics_df['last_close'] >= self.min_price) & (metrics_df['last_close'] <= self.max_price)
        consolidation_mask = metrics_df['consol_range_pct'] < self.consolidation_range_pct
        volume_mask = metrics_df['breakout_volume'] > (metrics_df['consol_avg_volume'] * self.breakout_volume_ratio)
        breakout_mask = metrics_df['last_close'] > metrics_df['consol_high']

        self.logger.debug(f"Price mask hits: {price_mask.sum()}")
        self.logger.debug(f"Consolidation mask hits: {consolidation_mask.sum()}")
        self.logger.debug(f"Volume mask hits: {volume_mask.sum()}")
        self.logger.debug(f"Breakout mask hits: {breakout_mask.sum()}")

        final_mask = price_mask & consolidation_mask & volume_mask & breakout_mask
        final_candidate_symbols = metrics_df[final_mask].index.tolist()

        self.logger.info(f"Vectorized scan found {len(final_candidate_symbols)} candidates.")

        for symbol in final_candidate_symbols:
            symbol_consol_data = consolidation_df.loc[symbol]
            profiler = VolumeProfiler(self.tick_size)
            profile = profiler.calculate(symbol_consol_data)

            if profile and profile.get('poc_price') is not None:
                self.candidates[symbol] = {
                    'poc': profile['poc_price'],
                    'stop_loss': metrics_df.loc[symbol, 'consol_low'],
                    'status': 'watching'
                }

        self.logger.info(f"Processed POC for {len(self.candidates)} valid candidates.")
        return list(self.candidates.keys())

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
            breakeven_trigger_price = trade['entry_price'] + (trade['risk_per_share'] * self.breakeven_trigger_r)
            if bar['High'] >= breakeven_trigger_price:
                trade['stop_loss'] = trade['entry_price']
                trade['is_breakeven'] = True
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")

        exit_price, reason = None, None
        if bar['Low'] <= trade['stop_loss']:
            exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['High'] >= trade.get('profit_target', float('inf')):
            exit_price, reason = trade['profit_target'], "Profit Target"
        elif bar.name.time() >= time(15, 50):
            exit_price, reason = bar['Close'], "End of Day"

        if exit_price:
            self.logger.info(f"  [EXIT] {symbol} @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices, exit_reason=reason)
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
            risk_per_share = entry_price - stop_loss_price

            if risk_per_share <= 0:
                del self.candidates[symbol]
                return

            profit_target_price = entry_price + (risk_per_share * self.profit_target_r)

            equity = self.ledger.get_total_equity(self.current_prices)
            dollar_risk = equity * self.risk_per_trade_pct
            quantity = int(dollar_risk / risk_per_share)

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
            del self.candidates[symbol]

    def on_session_end(self):
        pass