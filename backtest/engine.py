# backtest/engine.py

import pandas as pd
from tqdm import tqdm
from datetime import time, datetime
import yaml
import logging
import os

from core.ledger import BacktestLedger
from analytics.indicators import calculate_vwap
from analytics.profiles import get_session
from strategies.base import BaseStrategy
from strategies.open_drive_momentum_strategy import OpenDriveMomentumStrategy
from strategies.simple_momentum_strategy import SimpleMomentumStrategy
from strategies.value_migration_strategy import ValueMigrationStrategy
from strategies.volume_poc_strategy import SimplePocCrossStrategy
from strategies.debug_visualization_strategy import DebugVisualizationStrategy
from strategies.open_rejection_reverse_strategy import OpenRejectionReverseStrategy
from core.fee_models import ZeroFeeModel, TieredIBFeeModel, FixedFeeModel
from .results import BacktestResults
import pytz

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestEngine:
    """
    Manages the setup and execution of a backtest for a given strategy.
    """
    STRATEGY_MAPPING = {
        'OpenDriveMomentumStrategy': OpenDriveMomentumStrategy,
        'SimpleMomentumStrategy': SimpleMomentumStrategy,
        'ValueMigrationStrategy': ValueMigrationStrategy,
        'SimplePocCrossStrategy': SimplePocCrossStrategy,
        'DebugVisualizationStrategy': DebugVisualizationStrategy,
        'OpenRejectionReverseStrategy': OpenRejectionReverseStrategy,
    }

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.strategy_params = self.config['strategy']['parameters'].get(self.strategy_name, {})
        self.backtest_params = self.config['backtest']
        self.fee_config = self.config.get('fees', {'model': 'zero'})
        self.symbols = []
        self.all_data = {}
        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model()
        )
        self.strategy = None
        self.tz_str = self.strategy_params.get('timezone', 'America/New_York')
        self.timezone = pytz.timezone(self.tz_str)
        self._load_all_data()

    def _load_config(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_fee_model(self):
        model_name = self.fee_config.get('model', 'zero').lower()
        if model_name == 'tiered':
            return TieredIBFeeModel(**self.fee_config.get('tiered', {}))
        elif model_name == 'fixed':
            return FixedFeeModel(**self.fee_config.get('fixed', {}))
        return ZeroFeeModel()

    def _load_all_data(self):
        data_dir = self.backtest_params['data_dir']
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for filename in tqdm(all_files, desc="Loading All Historical Data"):
            symbol = filename.split('.csv')[0]
            try:
                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path)
                df.columns = [col.strip().lower() for col in df.columns]
                df.set_index(pd.to_datetime(df['date'], utc=True), inplace=True)
                df.index.name = 'timestamp'
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                final_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df[final_cols].apply(pd.to_numeric)
                if symbol not in self.all_data: self.all_data[symbol] = []
                self.all_data[symbol].append(df)
            except Exception as e:
                print(f"ERROR processing {filename}: {e}")

        for symbol, df_list in self.all_data.items():
            if df_list:
                self.all_data[symbol] = pd.concat(df_list).sort_index()
            else:
                self.all_data[symbol] = pd.DataFrame()

    def _get_data_for_date(self, data, target_date):
        if data.empty: return pd.DataFrame()
        return data.loc[data.index.date == target_date]

    def run(self, trade_date: datetime):
        self.trade_date = trade_date

        # --- FIX #1: Settle cash from the previous day's trades ---
        self.ledger.settle_funds()

        # --- FIX #2: Safely check for empty dataframes before processing ---
        all_available_dates = sorted(list(set(
            d.date() for s_data in self.all_data.values()
            if isinstance(s_data, pd.DataFrame) and not s_data.empty
            for d in s_data.index
        )))

        prev_day_date = next((d for d in sorted(all_available_dates, reverse=True) if d < trade_date.date()), None)

        if not prev_day_date:
            print(f"No prior day data found for {self.trade_date.strftime('%Y-%m-%d')}. Skipping scan.")
            return None

        historical_data_for_scan = {s: self._get_data_for_date(df, prev_day_date) for s, df in self.all_data.items()}

        strategy_class = self.STRATEGY_MAPPING[self.strategy_name]
        self.strategy = strategy_class(list(self.all_data.keys()), self.ledger, **self.strategy_params)

        self.symbols = self.strategy.scan_for_candidates(self.trade_date, historical_data_for_scan)

        if not self.symbols:
            print(f"No candidates found for {self.trade_date.strftime('%Y-%m-%d')}. Skipping.")
            return None

        session_data = {s: self._get_data_for_date(self.all_data[s], self.trade_date.date()) for s in self.symbols if s in self.all_data}
        session_data = {s: df for s, df in session_data.items() if not df.empty}

        if not session_data:
            print(f"No market data available for any candidates on {self.trade_date.strftime('%Y-%m-%d')}. Skipping.")
            return None

        self.strategy.on_session_start(session_data)

        all_timestamps = sorted(list(set.union(*(set(df.index) for df in session_data.values()))))

        for timestamp in all_timestamps:
            current_bar_data = {}
            session_bars = {}
            market_prices = {}
            analytics = {}

            for symbol, df in session_data.items():
                if timestamp in df.index:
                    current_bar_data[symbol] = df.loc[timestamp]
                    market_prices[symbol] = df.loc[timestamp]['Close']
                    session_bars[symbol] = df.loc[df.index <= timestamp]
                    analytics[symbol] = {'vwap': calculate_vwap(session_bars[symbol].copy())}

            if current_bar_data:
                self.strategy.on_bar(current_bar_data, session_bars, market_prices, analytics)
                self.ledger._update_equity(timestamp, market_prices)

        self.strategy.on_session_end()
        return BacktestResults(self.ledger)