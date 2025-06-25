# backtest/engine.py

import pandas as pd
import logging
from tqdm import tqdm
from importlib import import_module
from zoneinfo import ZoneInfo
from datetime import date
import yaml
import os
import re


from core.ledger import BacktestLedger
from core.fee_models import ZeroFeeModel, TieredIBFeeModel, FixedFeeModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
engine_logger = logging.getLogger(__name__)

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class BacktestEngine:
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.backtest_params = self.config['backtest']
        tz_name = self.backtest_params.get("timezone", "UTC")
        self.tz = ZoneInfo(tz_name)
        self.fee_config = self.config.get('fees', {'model': 'zero'})

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feather_path = self.backtest_params.get('feather_dir', 'data/feather')
        self.feather_dir = os.path.join(self.project_root, feather_path)

        # --- NEW: Store the user's desired TEST period ---
        self.test_start_dt = pd.Timestamp(start_date, tz=self.tz)
        self.test_end_dt   = pd.Timestamp(end_date,   tz=self.tz)

        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model(),
            slippage_model=self.backtest_params.get('slippage_model', 'none'),
            slippage_pct=self.backtest_params.get('slippage_pct', 0.0)
        )

        # --- REVISED ORDER OF OPERATIONS ---
        # 1. Get the full universe of symbols available in the data directory.
        engine_logger.info("Scanning for all available symbols...")
        available_symbols = self._get_liquid_symbols()

        # 2. Initialize the strategy to determine which symbols it will use and its lookback period.
        self.strategy, self.symbols = self._initialize_strategy(available_symbols)

        # 3. Load data ONLY for the symbols the strategy requires.
        self.data_store = self._load_all_data_from_feather()

        # 4. Generate the FULL trading calendar from all available data.
        # This will be used for slicing lookback periods.
        self.full_trading_calendar = self._get_all_trading_dates()

        # 5. Generate the TEST calendar, which the backtest will loop over.
        # This ensures the backtest only runs on the user-specified dates.
        self.test_trading_calendar = [
            dt for dt in self.full_trading_calendar
            if self.test_start_dt.date() <= dt <= self.test_end_dt.date()
        ]

        if not self.test_trading_calendar:
            raise ValueError("The specified date range contains no trading days found in the data.")

        engine_logger.info(f"Test period contains {len(self.test_trading_calendar)} trading days.")

        self.current_prices = {s: 0 for s in self.symbols}


    def _get_liquid_symbols(self) -> list[str]:
        engine_logger.info(f"Scanning for all available symbols in {self.feather_dir}...")
        if not os.path.exists(self.feather_dir):
            raise FileNotFoundError(f"Feather data directory not found: {self.feather_dir}")

        symbols = [os.path.splitext(f)[0] for f in os.listdir(self.feather_dir) if f.endswith('.feather')]
        engine_logger.info(f"Found {len(symbols)} total available symbols.")
        return symbols

    def _load_all_data_from_feather(self) -> dict[str, pd.DataFrame]:
        engine_logger.info(f"Pre-loading data for {len(self.symbols)} strategy-specific symbols...")
        data_store = {}
        # This now iterates over the CORRECT, larger list of symbols (if SPY was added).
        for symbol in tqdm(self.symbols, desc="Loading Feather Files"):
            file_path = os.path.join(self.feather_dir, f"{symbol}.feather")
            if os.path.exists(file_path):
                df = pd.read_feather(file_path)
                df.set_index("timestamp", inplace=True)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(self.tz)

                df["date"] = df.index.date

                df.columns = [col.lower() for col in df.columns]
                data_store[symbol] = df
            else:
                engine_logger.warning(f"Feather file for symbol {symbol} not found at {file_path}. Skipping.")
        return data_store

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

    def _initialize_strategy(self, available_symbols: list) -> tuple:
        """
        MODIFIED: Initializes the strategy and returns the instance and the final list
        of symbols the strategy will operate on, ensuring SPY is included.
        """
        strategy_module_name = camel_to_snake(self.strategy_name)
        strategy_module_path = f'strategies.{strategy_module_name}'
        try:
            strategy_module = import_module(strategy_module_path)
            strategy_class = getattr(strategy_module, self.strategy_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find strategy '{self.strategy_name}' in '{strategy_module_path}.py'. Error: {e}")

        all_strategy_params = self.config['strategy'].get('parameters', {})
        active_strategy_params = all_strategy_params.get(self.strategy_name, {})

        final_symbols = []
        use_file = active_strategy_params.get('use_symbol_file', False)
        symbol_file_path = active_strategy_params.get('symbol_file')

        if use_file and symbol_file_path:
            full_path = os.path.join(self.project_root, symbol_file_path)
            try:
                with open(full_path, 'r') as f:
                    symbols_from_file = {line.strip().upper() for line in f if line.strip()}
                final_symbols = list(set(available_symbols) & symbols_from_file)
                engine_logger.info(f"Strategy will use {len(final_symbols)} symbols found in both the file and data directory.")
            except FileNotFoundError:
                engine_logger.error(f"Symbol file not found: {full_path}. Defaulting to all available symbols.")
                final_symbols = available_symbols
        else:
            final_symbols = available_symbols
            engine_logger.info(f"Strategy will use all {len(final_symbols)} available symbols.")

        # --- THIS IS THE FIX ---
        # Ensure SPY is always included for benchmark calculations.
        # We use a set for an efficient check.
        if 'SPY' not in set(final_symbols):
            engine_logger.info("Adding 'SPY' to symbol list for benchmark calculations.")
            final_symbols.append('SPY')

        # Pass the original symbol list (without SPY unless specified) to the strategy
        strategy_symbols = [s for s in final_symbols if s != 'SPY'] if 'SPY' not in (set(available_symbols) if use_file else set()) else final_symbols

        strategy_instance = strategy_class(
            symbols=strategy_symbols, # Pass the original list to the strategy
            ledger=self.ledger,
            config=self.config,
            timezone=self.tz,
            **active_strategy_params
        )
        # The engine itself will use the list that includes SPY
        return strategy_instance, final_symbols



    def _get_all_trading_dates(self) -> list[date]:
        """Generates a sorted list of all unique trading dates from the data store."""
        all_dates = set()
        for df in self.data_store.values():
            all_dates.update(df['date'].unique())

        return sorted(list(all_dates))


    def _fetch_data(self, dates_to_fetch: list) -> dict[str, pd.DataFrame]:
        if not self.symbols or not dates_to_fetch: return {}

        results = {}
        for symbol in self.symbols:
            if symbol in self.data_store:
                df = self.data_store[symbol]
                filtered_df = df[df['date'].isin(dates_to_fetch)]
                if not filtered_df.empty:
                    results[symbol] = filtered_df
        return results

    def run(self):
        engine_logger.info(f"Starting backtest for period: {self.test_start_dt.date()} to {self.test_end_dt.date()}...")
        lookback_period = self.strategy.get_required_lookback()

        for trade_date in tqdm(self.test_trading_calendar, desc="Running Backtest"):
            self.ledger.settle_funds(trade_date)

            try:
                current_date_index = self.full_trading_calendar.index(trade_date)
            except ValueError:
                engine_logger.warning(f"Trade date {trade_date} not found in full calendar. Skipping.")
                continue

            if current_date_index < lookback_period:
                engine_logger.warning(f"Not enough historical data for {trade_date}. Need {lookback_period} days, have {current_date_index}. Skipping day.")
                continue

            lookback_start_index = current_date_index - lookback_period
            lookback_dates = self.full_trading_calendar[lookback_start_index:current_date_index]
            historical_data = self._fetch_data(lookback_dates)
            intraday_data = self._fetch_data([trade_date])

            if not historical_data or not any(v is not None and not v.empty for v in intraday_data.values()):
                engine_logger.warning(f"Missing historical or intraday data for {trade_date}. Skipping.")
                continue

            self.strategy.on_market_open(historical_data, intraday_data)

            if not intraday_data:
                self.strategy.on_session_end()
                continue

            last_timestamp_today = None
            # Create a combined DataFrame to iterate through bars chronologically
            all_intraday_bars = pd.concat(intraday_data.values()).sort_index()

            for timestamp, bar in all_intraday_bars.iterrows():
                symbol = bar['symbol']
                # The engine provides all bars, but the strategy's on_bar will filter
                # for symbols it is explicitly managing.
                self.strategy.on_bar(symbol, bar)
                last_timestamp_today = timestamp

            self.strategy.on_session_end()
            if last_timestamp_today:
                self.ledger._update_equity(last_timestamp_today, self.strategy.current_prices)

        if hasattr(self.strategy, 'generate_report'):
            self.strategy.generate_report()

        engine_logger.info("Backtest finished.")
        return self.ledger