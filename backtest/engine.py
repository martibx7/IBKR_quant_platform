# backtest/engine.py

import yaml
import pandas as pd
import os
import logging
from tqdm import tqdm
from datetime import datetime
import pytz
from importlib import import_module
import re

from core.ledger import BacktestLedger
from backtest.results import BacktestResults
from core.fee_models import ZeroFeeModel, TieredIBFeeModel, FixedFeeModel

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

        # --- MODIFIED: Initialization Order ---
        # 1. Load all data first to know the symbol universe
        self._load_all_data()

        # 2. Initialize the ledger with the fee model
        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model()
        )

        # 3. Initialize the strategy, now with knowledge of all symbols
        self.strategy = self._initialize_strategy()

        # 4. Build the trading calendar
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
        """
        Dynamically imports and initializes the strategy, providing either a specific
        list of symbols from the config or the entire loaded universe.
        """
        strategy_module_name = camel_to_snake(self.strategy_name)
        strategy_module_path = f'strategies.{strategy_module_name}'

        try:
            strategy_module = import_module(strategy_module_path)
            strategy_class = getattr(strategy_module, self.strategy_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find strategy '{self.strategy_name}' in '{strategy_module_path}.py'. Error: {e}")

        # Inject global tick size settings
        self.strategy_params['tick_size_market_profile'] = self.backtest_params.get('tick_size_market_profile', 0.05)
        self.strategy_params['tick_size_volume_profile'] = self.backtest_params.get('tick_size_volume_profile', 0.01)

        # --- MODIFIED: Symbol Universe Logic ---
        # Check if a specific list of symbols is provided in the strategy's parameters
        if 'symbols' in self.strategy_params and self.strategy_params['symbols']:
            symbols_to_use = self.strategy_params['symbols']
            engine_logger.info(f"Using specific symbols from config for {self.strategy_name}: {symbols_to_use}")
        else:
            # Otherwise, default to the entire universe of loaded symbols
            symbols_to_use = list(self.all_data.keys())
            engine_logger.info(f"No specific symbols in config for {self.strategy_name}. Defaulting to all {len(symbols_to_use)} loaded symbols.")

        return strategy_class(symbols=symbols_to_use, ledger=self.ledger, **self.strategy_params)

    # ... (the rest of the file remains the same) ...
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
                engine_logger.error(f"ERROR processing {filename}: {e}")

        for symbol, df_list in self.all_data.items():
            if df_list:
                self.all_data[symbol] = pd.concat(df_list).sort_index()
            else:
                self.all_data[symbol] = pd.DataFrame()

    def _get_data_for_date(self, data, target_date):
        if data.empty: return pd.DataFrame()
        return data.loc[data.index.date == target_date]

    def run_backtest(self):
        engine_logger.info(f"Starting backtest for strategy '{self.strategy_name}'...")
        for trade_date in tqdm(self.trading_calendar, desc="Running Backtest"):
            trade_date_dt = datetime.combine(trade_date, datetime.min.time())
            self.prepare_for_day(trade_date_dt)
            if self.session_data:
                self.run_session()
        engine_logger.info("Backtest finished.")
        return BacktestResults(self.ledger)

    def prepare_for_day(self, trade_date: datetime):
        self.trade_date = trade_date
        engine_logger.info(f"\n--- Preparing for trade date: {self.trade_date.strftime('%Y-%m-%d')} ---")
        self.ledger.settle_funds()

        prev_day_date = None
        try:
            current_date_index = self.trading_calendar.index(trade_date.date())
            if current_date_index > 0:
                prev_day_date = self.trading_calendar[current_date_index - 1]
        except ValueError:
            engine_logger.warning(f"Date {trade_date.date()} not in trading calendar. Skipping day.")
            self.session_data = {}
            return

        if not prev_day_date:
            engine_logger.warning("This is the first day in the calendar, no previous day available for scanning.")
            self.symbols = []
            self.session_data = {}
            return

        historical_data_for_scan = {s: self._get_data_for_date(df, prev_day_date) for s, df in self.all_data.items()}
        self.symbols = self.strategy.scan_for_candidates(self.trade_date, historical_data_for_scan)

        if not self.symbols:
            engine_logger.info(f"No candidates found by strategy for {self.trade_date.strftime('%Y-%m-%d')}.")
            self.session_data = {}
            return

        session_data = {s: self._get_data_for_date(self.all_data[s], self.trade_date.date()) for s in self.symbols if s in self.all_data}
        self.session_data = {s: df for s, df in session_data.items() if not df.empty}

        if not self.session_data:
            engine_logger.warning(f"No market data available for any candidates on {self.trade_date.strftime('%Y-%m-%d')}.")
            return

        self.strategy.on_session_start(self.session_data)
        engine_logger.info(f"Preparation complete. Found {len(self.session_data)} candidates with data for today.")

    def run_session(self):
        engine_logger.info(f"--- Running trade session for {self.trade_date.strftime('%Y-%m-%d')} ---")
        all_timestamps = sorted(list(set.union(*(set(df.index) for df in self.session_data.values()))))

        for timestamp in all_timestamps:
            current_bar_data = {s: df.loc[timestamp] for s, df in self.session_data.items() if timestamp in df.index}
            session_bars = {s: df.loc[df.index <= timestamp] for s, df in self.session_data.items()}
            market_prices = {s: df.loc[timestamp]['Close'] for s, df in self.session_data.items() if timestamp in df.index}

            analytics = {}

            if current_bar_data:
                self.strategy.on_bar(current_bar_data, session_bars, market_prices, analytics)
                self.ledger._update_equity(timestamp, market_prices)

        self.strategy.on_session_end()