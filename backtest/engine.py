# backtest/engine.py

import yaml
import pandas as pd
import os
import logging
from tqdm import tqdm
from datetime import datetime
from typing import Union

from core.ledger import BacktestLedger
from backtest.results import BacktestResults
from core.fee_models import ZeroFeeModel, TieredIBFeeModel, FixedFeeModel
from analytics.profiles import get_session
from importlib import import_module
import re


engine_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class BacktestEngine:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.strategy_params = self.config['strategy']['parameters'].get(self.strategy_name, {})
        self.backtest_params = self.config['backtest']
        self.fee_config = self.config.get('fees', {'model': 'zero'})
        self.all_data = {}
        self.session_data = {}

        self._load_all_data()

        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model()
        )

        self.strategy = self._initialize_strategy()

        engine_logger.info("Building trading calendar...")
        all_dates = set()
        for symbol_data in self.all_data.values():
            if not symbol_data.empty:
                all_dates.update(symbol_data.index.normalize().date)

        self.trading_calendar = sorted([d for d in all_dates if d.weekday() < 5])
        engine_logger.info(f"Calendar built with {len(self.trading_calendar)} unique trading days.")

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

    def _initialize_strategy(self):
        strategy_module_name = camel_to_snake(self.strategy_name)
        strategy_module_path = f'strategies.{strategy_module_name}'
        try:
            strategy_module = import_module(strategy_module_path)
            strategy_class = getattr(strategy_module, self.strategy_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find strategy '{self.strategy_name}' in '{strategy_module_path}.py'. Error: {e}")

        self.strategy_params['tick_size'] = self.backtest_params.get('tick_size_volume_profile', 0.01)

        symbols_to_use = self.strategy_params.get('symbols') or list(self.all_data.keys())
        log_message = (f"Using specific symbols from config for {self.strategy_name}: {symbols_to_use}"
                       if self.strategy_params.get('symbols')
                       else f"No specific symbols in config for {self.strategy_name}. Defaulting to all {len(symbols_to_use)} loaded symbols.")
        engine_logger.info(log_message)

        return strategy_class(symbols=symbols_to_use, ledger=self.ledger, **self.strategy_params)

    def _load_all_data(self):
        data_dir = self.backtest_params['data_dir']
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for filename in tqdm(all_files, desc="Loading All Historical Data"):
            symbol = os.path.splitext(filename)[0]
            try:
                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path, parse_dates=['date'])
                df.columns = [col.strip().lower() for col in df.columns]
                df = df.rename(columns={'date': 'timestamp'})
                df.set_index(pd.to_datetime(df['timestamp'], utc=True), inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                self.all_data[symbol] = df.apply(pd.to_numeric)
            except Exception as e:
                engine_logger.error(f"ERROR processing {filename}: {e}")

    def _get_previous_trading_day(self, current_date_obj: datetime.date) -> Union[datetime.date, None]:
        try:
            idx = self.trading_calendar.index(current_date_obj)
            return self.trading_calendar[idx - 1] if idx > 0 else None
        except ValueError:
            return next((d for d in reversed(self.trading_calendar) if d < current_date_obj), None)

    def prepare_for_day(self, trade_date: datetime):
        self.current_trade_date = trade_date
        engine_logger.info(f"\n--- Preparing for trade date: {trade_date.strftime('%Y-%m-%d')} ---")
        self.ledger.settle_funds()

        if trade_date.date() not in self.trading_calendar:
            engine_logger.warning(f"Date {trade_date.strftime('%Y-%m-%d')} not in trading calendar. Skipping day.")
            self.session_data = {}
            return

        self.prev_trade_date = self._get_previous_trading_day(trade_date.date())
        if not self.prev_trade_date:
            engine_logger.warning(f"No previous trading day found for {trade_date.strftime('%Y-%m-%d')}. Cannot scan for candidates.")
            self.session_data = {}
            return

        candidate_symbols = self.strategy.scan_for_candidates(self.all_data, self.prev_trade_date)
        engine_logger.info(f"Preparation complete. Found {len(candidate_symbols)} candidates for today.")

        self.session_data = {}
        for symbol in candidate_symbols:
            symbol_data = get_session(self.all_data.get(symbol, pd.DataFrame()), trade_date.date(), "Regular")
            if not symbol_data.empty:
                self.session_data[symbol] = symbol_data

    def run_session(self):
        if not self.session_data:
            engine_logger.info(f"No session data for {self.current_trade_date.strftime('%Y-%m-%d')}. Nothing to process.")
            return

        all_timestamps = sorted(list(set.union(*(set(df.index) for df in self.session_data.values()))))
        self.strategy.on_session_start(self.session_data)

        for timestamp in tqdm(all_timestamps, desc=f"Simulating {self.current_trade_date.strftime('%Y-%m-%d')}", leave=False):
            # --- RESTORED: This block is crucial for equity calculation and was missing. ---
            market_prices = {s: df.loc[timestamp, 'Close'] for s, df in self.session_data.items() if timestamp in df.index}
            if market_prices:
                self.ledger._update_equity(timestamp, market_prices)
            # --- END RESTORED BLOCK ---

            for sym, data in self.session_data.items():
                if timestamp in data.index:
                    self.strategy.current_prices[sym] = data.loc[timestamp, 'Close']
            for sym, data in self.session_data.items():
                if timestamp in data.index:
                    bar = data.loc[timestamp]
                    self.strategy.on_bar(sym, bar)

        self.strategy.on_session_end()