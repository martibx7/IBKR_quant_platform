# backtest/engine.py

import yaml
import pandas as pd
import os
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Union
from sqlalchemy import create_engine, text

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
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.strategy_params = self.config['strategy']['parameters'].get(self.strategy_name, {})
        self.backtest_params = self.config['backtest']
        self.data_source_config = self.config.get('data_source', {'type': 'csv'})
        self.fee_config = self.config.get('fees', {'model': 'zero'})

        self.start_date = start_date
        self.end_date = end_date

        self.all_data = {}
        self.session_data = {}

        self.db_engine = None
        if self.data_source_config.get('type') == 'sqlite':
            db_path = self.data_source_config.get('db_path')
            if not db_path: raise ValueError("db_path not specified in config for sqlite source.")
            if not os.path.exists(db_path): raise FileNotFoundError(f"DB not found at {db_path}.")
            self.db_engine = create_engine(f'sqlite:///{db_path}')

        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model()
        )

        self.available_symbols = self._get_available_symbols()
        self.strategy = self._initialize_strategy()

        # --- FIX: Call the method directly without assigning its return value ---
        self._build_trading_calendar()

    def _load_config(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    # --- NEW: Helper method to get all available symbols from the DB ---
    def _get_available_symbols(self):
        """Gets a list of all unique symbols available in the data source."""
        if self.data_source_config.get('type') == 'sqlite':
            engine_logger.info("Fetching available symbols from database...")
            with self.db_engine.connect() as connection:
                result = connection.execute(text("SELECT DISTINCT symbol FROM price_data"))
                return [row[0] for row in result]
        else: # Fallback for CSV
            data_dir = self.backtest_params['data_dir']
            return [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.csv')]


    def _initialize_fee_model(self):
        model_name = self.fee_config.get('model', 'zero').lower()
        if model_name == 'tiered':
            return TieredIBFeeModel(**self.fee_config.get('tiered', {}))
        elif model_name == 'fixed':
            return FixedFeeModel(**self.fee_config.get('fixed', {}))
        return ZeroFeeModel()

    # --- UPDATED: Now includes logic to read symbols from a file ---
    def _initialize_strategy(self):
        strategy_module_name = camel_to_snake(self.strategy_name)
        strategy_module_path = f'strategies.{strategy_module_name}'
        try:
            strategy_module = import_module(strategy_module_path)
            strategy_class = getattr(strategy_module, self.strategy_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find strategy '{self.strategy_name}' in '{strategy_module_path}.py'. Error: {e}")

        self.strategy_params['tick_size'] = self.backtest_params.get('tick_size_volume_profile', 0.01)

        symbols_to_use = []

        # --- UPDATED: More robust logic for handling symbol sources ---

        # .get() is safer as it returns None if the key doesn't exist
        strategy_symbols = self.strategy_params.get('symbols')
        symbol_file = self.backtest_params.get('symbol_file')

        # Priority: 1. Strategy-specific 'symbols' list, 2. Global 'symbol_file', 3. All available
        if strategy_symbols:
            symbols_to_use = strategy_symbols
            engine_logger.info(f"Using specific symbols from strategy config list: {len(symbols_to_use)} symbols.")
        elif symbol_file: # This will be false if symbol_file is None or an empty string ""
            engine_logger.info(f"Loading symbols from global file: {symbol_file}")
            try:
                # Assume the file path is relative to the project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_path = os.path.join(project_root, symbol_file)
                with open(full_path, 'r') as f:
                    symbols_to_use = [line.strip().upper() for line in f if line.strip()]
                engine_logger.info(f"Loaded {len(symbols_to_use)} symbols from {symbol_file}.")
            except FileNotFoundError:
                engine_logger.error(f"ERROR: Symbol file not found at '{full_path}'. Using all available symbols.")
                symbols_to_use = self.available_symbols
        else:
            symbols_to_use = self.available_symbols
            engine_logger.info(f"No specific symbols in config. Defaulting to all {len(symbols_to_use)} available symbols.")

        return strategy_class(symbols=symbols_to_use, ledger=self.ledger, **self.strategy_params)

    # --- NEW: Builds the calendar from the DB without loading all price data ---
    def _build_trading_calendar(self):
        engine_logger.info("Building trading calendar...")
        if self.data_source_config.get('type') == 'sqlite':
            query = "SELECT DISTINCT date(timestamp) as trade_date FROM price_data WHERE date(timestamp) BETWEEN ? AND ?"
            dates_df = pd.read_sql(query, self.db_engine, params=(self.start_date, self.end_date))
            self.trading_calendar = sorted([pd.to_datetime(d).date() for d in dates_df['trade_date']])
        else: # Fallback for CSV
            # This part remains memory-intensive for CSVs, but the primary path is now sqlite
            temp_all_data = {}
            data_dir = self.backtest_params['data_dir']
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            for filename in tqdm(all_files, desc="Building Calendar from CSVs"):
                symbol = os.path.splitext(filename)[0]
                df = pd.read_csv(os.path.join(data_dir, filename), usecols=['date'], parse_dates=['date'])
                all_dates = set(df['date'].dt.date)
            self.trading_calendar = sorted([d for d in all_dates if d.weekday() < 5])

        engine_logger.info(f"Calendar built with {len(self.trading_calendar)} unique trading days.")

    # --- NEW: This method loads data for a small, rolling window each day ---
    def _load_data_for_day(self, trade_date: datetime):
        """
        Loads data from SQLite for the specific trade_date plus a lookback period,
        pre-filtering for liquidity.
        """
        lookback_days = self.strategy_params.get('consolidation_days', 10) + 20
        lookback_start_date = trade_date - timedelta(days=lookback_days)

        min_avg_volume = self.backtest_params.get('min_avg_daily_volume', 0)

        # Step 1: Find symbols that meet the liquidity criteria (this part is unchanged and correct)
        liquid_symbols_query = """
            SELECT symbol FROM (
                SELECT symbol, date(timestamp) as day, SUM(volume) as daily_volume
                FROM price_data
                WHERE date(timestamp) BETWEEN ? AND ?
                GROUP BY symbol, day
            )
            GROUP BY symbol
            HAVING AVG(daily_volume) >= ?
        """
        liquid_symbols_df = pd.read_sql(
            liquid_symbols_query,
            self.db_engine,
            params=(lookback_start_date.strftime('%Y-%m-%d'), trade_date.strftime('%Y-%m-%d'), min_avg_volume)
        )
        liquid_symbols = liquid_symbols_df['symbol'].tolist()

        if not liquid_symbols:
            self.all_data = {}
            return

        # --- FIX: Simplify the main query and filter in pandas ---

        # Step 2: Fetch all price data in the date range. This is fast on an indexed DB.
        price_data_query = "SELECT * FROM price_data WHERE date(timestamp) BETWEEN ? AND ?"

        daily_slice_df = pd.read_sql(
            price_data_query,
            self.db_engine,
            index_col='timestamp',
            parse_dates=['timestamp'],
            params=(lookback_start_date.strftime('%Y-%m-%d'), trade_date.strftime('%Y-%m-%d'))
        )

        if daily_slice_df.empty:
            self.all_data = {}
            return

        # Step 3: Use pandas' highly optimized 'isin' to filter for our liquid symbols.
        daily_slice_df = daily_slice_df[daily_slice_df['symbol'].isin(liquid_symbols)]
        # --- END FIX ---

        daily_slice_df.index = pd.to_datetime(daily_slice_df.index, utc=True)
        daily_slice_df = daily_slice_df[['open', 'high', 'low', 'close', 'volume', 'symbol']]
        daily_slice_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']

        self.all_data = {symbol: group.drop(columns=['symbol']) for symbol, group in daily_slice_df.groupby('symbol')}
        self.strategy.symbols = list(self.all_data.keys())

    def _get_previous_trading_day(self, current_date_obj: datetime.date) -> Union[datetime.date, None]:
        # ... (this method is unchanged) ...
        try:
            idx = self.trading_calendar.index(current_date_obj)
            return self.trading_calendar[idx - 1] if idx > 0 else None
        except ValueError:
            return next((d for d in reversed(self.trading_calendar) if d < current_date_obj), None)

    # --- UPDATED: prepare_for_day now drives the daily data loading ---
    def prepare_for_day(self, trade_date: datetime):
        self.current_trade_date = trade_date
        engine_logger.info(f"\n--- Preparing for trade date: {trade_date.strftime('%Y-%m-%d')} ---")
        self.ledger.settle_funds()

        if trade_date.date() not in self.trading_calendar:
            engine_logger.warning(f"Date {trade_date.strftime('%Y-%m-%d')} not in trading calendar. Skipping day.")
            self.session_data = {}
            return

        # Load data for today + lookback period
        self._load_data_for_day(trade_date.date())

        self.prev_trade_date = self._get_previous_trading_day(trade_date.date())
        if not self.prev_trade_date:
            engine_logger.warning(f"No previous trading day found for {trade_date.strftime('%Y-%m-%d')}. Cannot scan for candidates.")
            self.session_data = {}
            return

        candidate_symbols = self.strategy.scan_for_candidates(self.all_data, self.prev_trade_date)
        engine_logger.info(f"Preparation complete. Found {len(candidate_symbols)} candidates for today.")

        self.session_data = {}
        for symbol in candidate_symbols:
            # We already have the data in self.all_data, just get today's session from it
            if symbol in self.all_data:
                symbol_data_full = self.all_data[symbol]
                symbol_session_data = symbol_data_full[symbol_data_full.index.date == trade_date.date()]
                if not symbol_session_data.empty:
                    self.session_data[symbol] = symbol_session_data

    def run_session(self):
        # ... (this method is unchanged) ...
        if not self.session_data:
            engine_logger.info(f"No session data for {self.current_trade_date.strftime('%Y-%m-%d')}. Nothing to process.")
            return

        all_timestamps = sorted(list(set.union(*(set(df.index) for df in self.session_data.values()))))
        self.strategy.on_session_start(self.session_data)

        for timestamp in tqdm(all_timestamps, desc=f"Simulating {self.current_trade_date.strftime('%Y-%m-%d')}", leave=False):
            market_prices = {s: df.loc[timestamp, 'Close'] for s, df in self.session_data.items() if timestamp in df.index}
            if market_prices:
                self.ledger._update_equity(timestamp, market_prices)

            for sym, data in self.session_data.items():
                if timestamp in data.index:
                    self.strategy.current_prices[sym] = data.loc[timestamp, 'Close']
            for sym, data in self.session_data.items():
                if timestamp in data.index:
                    bar = data.loc[timestamp]
                    self.strategy.on_bar(sym, bar)

        self.strategy.on_session_end()