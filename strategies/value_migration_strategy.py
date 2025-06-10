# strategies/value_migration_strategy.py

from datetime import datetime, time
import pandas as pd
from tqdm import tqdm
import pytz
import logging
import os

from .base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session
from analytics.indicators import calculate_vwap

class ValueMigrationStrategy(BaseStrategy):
    """
    Implements the Value Migration & VWAP Reclaim Strategy with corrected
    ranking, stricter filtering, and prioritized trade execution.
    """
    def __init__(self, symbols: list[str], ledger, **params):
        super().__init__(symbols, ledger, **params)
        self.risk_per_trade_pct = params.get('risk_per_trade_pct', 0.02)
        self.max_allocation_pct = params.get('max_allocation_pct', 0.25)
        self.min_price = params.get('min_price', 2.0)
        self.max_price = params.get('max_price', 30.0)
        self.max_daily_trades = params.get('max_daily_trades', 5)
        self.params = params

        self.profiler_class = VolumeProfiler
        self.tick_size = params.get('tick_size', 0.01)

        self.timezone = pytz.timezone(params.get('timezone', 'America/New_York'))
        self.prev_day_stats = {}
        self.active_trades = {}
        self.ranked_candidates = []
        self.session_data_today = {}
        self.current_prices = {}
        self.vwap_history = {}
        self.trades_today = 0

        self._setup_logger(params.get('log_file', 'logs/value_migration_details.log'))

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

    def scan_for_candidates(self, all_data: dict[str, pd.DataFrame], date: datetime.date) -> list[str]:
        self.logger.info(f"--- SCANNING CANDIDATES FOR {date.strftime('%Y-%m-%d')} ---")

        if self.ledger.cash < self.min_price:
            self.logger.warning("Cash balance below min_price. Halting scanning for the day.")
            return []

        candidates = []
        for symbol, df in all_data.items():
            prev_day_data = get_session(df, date, "Regular", tz_str=self.timezone.zone)
            if prev_day_data.empty: continue

            last_close = prev_day_data['Close'].iloc[-1]
            if not (self.min_price <= last_close <= self.max_price):
                continue

            profiler = self.profiler_class(self.tick_size)
            profile = profiler.calculate(prev_day_data)

            if not profile or not all(k in profile for k in ['value_area_high', 'value_area_low', 'poc_price']):
                continue

            day_high = prev_day_data['High'].max()
            day_low = prev_day_data['Low'].min()

            cond_poc = last_close > profile['poc_price']
            cond_range = (last_close - day_low) / (day_high - day_low) > 0.75 if (day_high - day_low) > 0 else False

            if cond_poc and cond_range:
                self.prev_day_stats[symbol] = {
                    'vah': profile['value_area_high'],
                    'val': profile['value_area_low'],
                    'poc': profile['poc_price'],
                    'high': day_high,
                    'low': day_low,
                    'close': last_close
                }
                candidates.append(symbol)

        self.logger.info(f"Found {len(candidates)} potential symbols after stricter filtering.")
        return candidates

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        self.logger.info("--- NEW SESSION ---")
        self.active_trades.clear()
        self.ranked_candidates = []
        self.session_data_today = session_data
        self.current_prices = {s: df.iloc[0]['Open'] for s, df in session_data.items()}
        self.vwap_history = {}
        self.trades_today = 0

        for symbol, df in session_data.items():
            if symbol in self.prev_day_stats:
                open_price = df.iloc[0]['Open']
                prev_stats = self.prev_day_stats[symbol]

                if open_price <= prev_stats['vah']:
                    continue

                initial_balance_end_time = (datetime.combine(datetime.now().date(), time(9, 30)) + pd.Timedelta(minutes=30)).time()
                initial_balance_df = df.between_time(time(9,30), initial_balance_end_time)
                if not initial_balance_df.empty and initial_balance_df['Low'].min() < prev_stats['vah']:
                    continue

                gap_strength = (open_price - prev_stats['vah']) / prev_stats['vah'] if prev_stats['vah'] > 0 else 0
                close_vs_range = (prev_stats['close'] - prev_stats['low']) / (prev_stats['high'] - prev_stats['low']) if (prev_stats['high'] - prev_stats['low']) > 0 else 0
                buying_tail_size = (prev_stats['val'] - prev_stats['low']) / prev_stats['val'] if prev_stats['val'] > 0 else 0

                # --- REVISED RANKING: Emphasize the gap strength ---
                score_gap = 0.6 * gap_strength
                score_range = 0.2 * close_vs_range
                score_tail = 0.2 * buying_tail_size

                total_score = score_gap + score_range + score_tail

                rank_details = {
                    'symbol': symbol,
                    'score': total_score,
                    'gap_pct': gap_strength * 100,
                    'gap_score': score_gap,
                    'range_pct': close_vs_range * 100,
                    'range_score': score_range,
                    'tail_pct': buying_tail_size * 100,
                    'tail_score': score_tail
                }
                self.ranked_candidates.append(rank_details)

        self.ranked_candidates = sorted(self.ranked_candidates, key=lambda x: x['score'], reverse=True)
        self.logger.info(f"Ranked {len(self.ranked_candidates)} candidates for today:")
        for rank in self.ranked_candidates:
            log_msg = (
                f"  [RANK] {rank['symbol']:<6} | Score: {rank['score']:.4f} | "
                f"Gap: {rank['gap_pct']:.2f}% (s:{rank['gap_score']:.4f}) | "
                f"Range: {rank['range_pct']:.2f}% (s:{rank['range_score']:.4f}) | "
                f"Tail: {rank['tail_pct']:.2f}% (s:{rank['tail_score']:.4f})"
            )
            self.logger.info(log_msg)

    def on_bar(self, symbol: str, bar: pd.Series):
        if symbol in self.active_trades:
            if bar.name.time() >= time(15, 55):
                trade = self.active_trades[symbol]
                exit_price = bar['Close']
                self.logger.info(f"  [EOD EXIT] {symbol} at {exit_price:.2f}")
                self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices)
                del self.active_trades[symbol]
            return

        self.current_prices[symbol] = bar['Close']

        if self.trades_today >= self.max_daily_trades:
            return

        # --- CORRECTED LOGIC: Only check for triggers in the top N ranked candidates ---
        top_candidates = [c['symbol'] for c in self.ranked_candidates[:self.max_daily_trades]]
        if symbol not in top_candidates:
            return

        session_df = self.session_data_today[symbol].loc[:bar.name]
        vwap = calculate_vwap(session_df)['vwap'].iloc[-1]
        self.vwap_history.setdefault(symbol, []).append(vwap)

        if bar.name.time() <= (datetime.combine(datetime.now().date(), time(9, 30)) + pd.Timedelta(minutes=30)).time():
            return

        is_above_vwap = bar['Close'] > vwap
        touched_vwap = bar['Low'] <= vwap
        is_vwap_rising = len(self.vwap_history.get(symbol, [])) > 1 and self.vwap_history[symbol][-1] > self.vwap_history[symbol][-2]

        if is_above_vwap and touched_vwap and is_vwap_rising:
            stop_loss_price = bar['Low']

            entry_price = bar['Close']
            risk_per_share = entry_price - stop_loss_price
            if risk_per_share <= 0: return

            equity = self.ledger.get_total_equity(self.current_prices)
            risk_quantity = int((equity * self.risk_per_trade_pct) / risk_per_share) if risk_per_share > 0 else 0
            alloc_quantity = int((equity * self.max_allocation_pct) / entry_price) if entry_price > 0 else 0
            cash_quantity = int(self.ledger.cash / entry_price) if entry_price > 0 else 0
            quantity = min(risk_quantity, alloc_quantity, cash_quantity)

            if quantity == 0: return

            self.logger.info(f">>> TRIGGER for {symbol} at {bar.name.time()} on VWAP reclaim <<<")
            self.logger.info(f"  [EXECUTE] LONG {symbol} | Qty: {quantity} @ {entry_price:.2f} | Stop: {stop_loss_price:.2f}")
            if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                self.active_trades[symbol] = {'stop_loss': stop_loss_price, 'quantity': quantity}
                self.trades_today += 1
                self.ranked_candidates = [c for c in self.ranked_candidates if c['symbol'] != symbol]

    def on_session_end(self):
        pass