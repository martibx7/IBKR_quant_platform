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
    # ... (init and other methods are unchanged) ...
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.strategy_params = self.config['strategy']['parameters'].get(self.strategy_name, {})
        self.backtest_params = self.config['backtest']
        self.data_source_config = self.config.get('data_source', {'type': 'csv'})
        self.fee_config = self.config.get('fees', {'model': 'zero'})

        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.all_data = {}
        self.session_data = {}
        self.master_data_df = None

        self.db_engine = None
        if self.data_source_config.get('type') == 'sqlite':
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'price_data.db')
            if not os.path.exists(db_path): raise FileNotFoundError(f"Database not found at {db_path}. Please run tools/populate_db.py")
            self.db_engine = create_engine(f'sqlite:///{db_path}')

        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model()
        )

        self.available_symbols = self._get_available_symbols()
        self.strategy = self._initialize_strategy()

        if self.data_source_config.get('type') == 'sqlite':
            self._preload_all_data()

        self._build_trading_calendar()

    def _load_config(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _get_available_symbols(self):
        if self.data_source_config.get('type') == 'sqlite':
            engine_logger.info("Fetching available symbols from database...")
            with self.db_engine.connect() as connection:
                result = connection.execute(text("SELECT DISTINCT symbol FROM price_data"))
                return [row[0] for row in result]
        else: # Fallback for CSV
            data_dir = self.backtest_params.get('data_dir', 'data/historical')
            return [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.csv')]

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
        symbols_to_use = self.strategy_params.get('symbols') or self.available_symbols
        return strategy_class(symbols=symbols_to_use, ledger=self.ledger, **self.strategy_params)

    def _preload_all_data(self):
        engine_logger.info("Preloading all data for the backtest period... (This may take a moment)")
        lookback_days = self.strategy_params.get('consolidation_days', 10) + 20
        full_range_start_date = self.start_date - timedelta(days=lookback_days)

        liquid_symbols_query = """
            SELECT symbol FROM (
                SELECT symbol, date, SUM(volume) as daily_volume FROM price_data
                WHERE date BETWEEN ? AND ? GROUP BY symbol, date
            ) GROUP BY symbol HAVING AVG(daily_volume) >= ? """

        liquid_symbols_df = pd.read_sql(
            liquid_symbols_query, self.db_engine,
            params=(full_range_start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'), self.backtest_params.get('min_avg_daily_volume', 0))
        )
        liquid_symbols = liquid_symbols_df['symbol'].tolist()
        if not liquid_symbols:
            self.master_data_df = pd.DataFrame()
            return

        placeholders = ','.join('?' for _ in liquid_symbols)
        price_data_query = f"SELECT * FROM price_data WHERE date BETWEEN ? AND ? AND symbol IN ({placeholders})"

        params = tuple([full_range_start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')] + liquid_symbols)

        self.master_data_df = pd.read_sql(
            price_data_query, self.db_engine,
            params=params,
            parse_dates=['timestamp']
        )

        if not self.master_data_df.empty:
            self.master_data_df.set_index('timestamp', inplace=True)
            self.master_data_df = self.master_data_df.tz_localize('UTC')
            self.master_data_df['date_obj'] = self.master_data_df.index.date

        engine_logger.info(f"Successfully preloaded {len(self.master_data_df)} rows into memory.")

    def _build_trading_calendar(self):
        engine_logger.info("Building trading calendar...")
        if self.data_source_config.get('type') == 'sqlite':
            if self.master_data_df is not None and not self.master_data_df.empty:
                all_dates = self.master_data_df['date_obj'].unique()
                self.trading_calendar = sorted([d for d in all_dates if self.start_date <= d <= self.end_date])
            else:
                self.trading_calendar = []
        else: # Fallback for CSV
            all_dates = set()
            data_dir = self.backtest_params['data_dir']
            for symbol in self.available_symbols:
                file_path = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, usecols=['date'], parse_dates=['date'])
                    all_dates.update(df['date'].dt.date)
            self.trading_calendar = sorted([d for d in all_dates if self.start_date <= d <= self.end_date])

        engine_logger.info(f"Calendar built with {len(self.trading_calendar)} unique trading days.")

    def _load_data_for_day(self, trade_date: datetime.date):
        if self.data_source_config.get('type') == 'sqlite':
            lookback_days = self.strategy_params.get('consolidation_days', 10) + 20
            lookback_start_date = trade_date - timedelta(days=lookback_days)

            daily_slice_df = self.master_data_df[
                (self.master_data_df['date_obj'] >= lookback_start_date) &
                (self.master_data_df['date_obj'] < trade_date + timedelta(days=1))
                ]
            if daily_slice_df.empty:
                self.all_data = {}
                return

            daily_slice_df_renamed = daily_slice_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'symbol': 'Symbol'})

            # --- FIX: Keep the 'date' column for accurate day counting in the strategy ---
            self.all_data = {symbol: group.drop(columns=['Symbol', 'date_obj']) for symbol, group in daily_slice_df_renamed.groupby('Symbol')}

        else: # Logic for CSV
            self.all_data = {}
            data_dir = self.backtest_params['data_dir']
            for symbol in self.available_symbols:
                file_path = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                    df.columns = [col.strip().lower() for col in df.columns]
                    # Reset index to get 'date' as a column, then rename
                    df.reset_index(inplace=True)
                    df = df.rename(columns={'date': 'timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df.set_index('timestamp', inplace=True)
                    # Ensure timezone awareness for CSV to match strategy expectations
                    if df.index.tz is None:
                        df = df.tz_localize('UTC')
                    # Add the text 'date' column for the strategy
                    df['date'] = df.index.strftime('%Y-%m-%d')
                    self.all_data[symbol] = df

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
            if symbol in self.all_data:
                symbol_data_full = self.all_data[symbol]
                symbol_session_data = symbol_data_full[symbol_data_full.index.date == trade_date.date()]
                if not symbol_session_data.empty:
                    self.session_data[symbol] = symbol_session_data

    def run_session(self):
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