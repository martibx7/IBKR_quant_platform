# strategies/consolidation_screener_strategy.py

import pandas as pd
import pytz
import logging
import os
from datetime import time

from .base import BaseStrategy

class ConsolidationScreenerStrategy(BaseStrategy):
    """
    A diagnostic strategy to audit the consolidation screening process.
    It does not trade. It only runs the vectorized scanner and logs every
    candidate that passes the initial screen to a CSV file for analysis.
    """
    def __init__(self, symbols: list[str], ledger, **params):
        super().__init__(symbols, ledger, **params)
        self.params = params
        self.min_price = params.get('min_price', 5.00)
        self.max_price = params.get('max_price', 100.00)
        self.consolidation_days = params.get('consolidation_days', 10)
        self.consolidation_range_pct = params.get('consolidation_range_pct', 0.07)
        self.breakout_volume_ratio = params.get('breakout_volume_ratio', 1.5)

        self.log_file = params.get('log_file', 'logs/consolidation_screener.log')
        self.output_csv_path = os.path.join(os.path.dirname(self.log_file), 'consolidation_candidates.csv')

        # Clear the CSV file at the start of a new backtest
        if os.path.exists(self.output_csv_path):
            os.remove(self.output_csv_path)

        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.propagate = False
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def scan_for_candidates(self, all_data: dict[str, pd.DataFrame], scan_date: pd.Timestamp) -> list[str]:
        self.logger.info(f"--- AUDIT SCAN FOR {scan_date.strftime('%Y-%m-%d')} ---")

        if not all_data: return []

        all_symbols_df = pd.concat(all_data.values(), keys=all_data.keys(), names=['symbol', 'timestamp_utc'])

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

        # --- Apply Filters ---
        price_mask = (metrics_df['last_close'] >= self.min_price) & (metrics_df['last_close'] <= self.max_price)
        consolidation_mask = metrics_df['consol_range_pct'] < self.consolidation_range_pct
        volume_mask = metrics_df['breakout_volume'] > (metrics_df['consol_avg_volume'] * self.breakout_volume_ratio)
        breakout_mask = metrics_df['last_close'] > metrics_df['consol_high']

        final_mask = price_mask & consolidation_mask & volume_mask & breakout_mask

        candidates_df = metrics_df[final_mask].copy()

        if not candidates_df.empty:
            candidates_df.reset_index(inplace=True)
            candidates_df.insert(0, 'scan_date', scan_date.strftime('%Y-%m-%d'))

            # --- Save to CSV ---
            # Append to the CSV file, writing headers only if the file is new
            header = not os.path.exists(self.output_csv_path)
            candidates_df.to_csv(self.output_csv_path, mode='a', header=header, index=False, float_format='%.2f')
            self.logger.info(f"Found {len(candidates_df)} candidates. Logged to {self.output_csv_path}")
        else:
            self.logger.info("Found 0 candidates.")

        return [] # Return an empty list because we are not trading

    # --- Dummy Methods: These do nothing ---
    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        pass

    def on_bar(self, symbol: str, bar: pd.Series):
        pass

    def on_session_end(self):
        pass
