# backtest/engine.py

import pandas as pd
from sqlalchemy import create_engine, text
import logging
import pytz
from tqdm import tqdm
from importlib import import_module
import yaml
import os
import re

from core.ledger import BacktestLedger
from core.fee_models import ZeroFeeModel, TieredIBFeeModel, FixedFeeModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
engine_logger = logging.getLogger(__name__)

def get_db_engine(config: dict):
    """Creates a SQLAlchemy engine from the config file."""
    db_config = config['database']
    db_type = db_config['type']

    if db_type == 'postgresql':
        conn_str = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        return create_engine(conn_str)
    elif db_type == 'sqlite':
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, db_config.get('db_path', 'data/price_data.db'))
        return create_engine(f"sqlite:///{db_path}")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class BacktestEngine:
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config = self._load_config(config_path)
        self.strategy_name = self.config['strategy']['name']
        self.backtest_params = self.config['backtest']
        self.fee_config = self.config.get('fees', {'model': 'zero'})

        self.start_dt = pd.Timestamp(start_date, tz='America/New_York')
        self.end_dt = pd.Timestamp(end_date, tz='America/New_York')

        self.db_engine = get_db_engine(self.config)

        # --- UPDATE: Pass slippage settings to the Ledger ---
        self.ledger = BacktestLedger(
            initial_cash=self.backtest_params['initial_cash'],
            fee_model=self._initialize_fee_model(),
            slippage_model=self.backtest_params.get('slippage_model', 'none'),
            slippage_pct=self.backtest_params.get('slippage_pct', 0.0)
        )

        self.symbols = self._get_liquid_symbols()
        self.strategy = self._initialize_strategy()
        self.trading_calendar = self._get_trade_dates()
        self.current_prices = {}

    def _get_liquid_symbols(self) -> list[str]:
        """
        Performs a one-time query to get a universe of symbols that meet
        the minimum average daily volume requirement from the config.
        """
        min_vol = self.backtest_params.get('min_avg_daily_volume', 0)
        if not min_vol: # If no minimum is set, get all symbols
            with self.db_engine.connect() as connection:
                result = connection.execute(text("SELECT DISTINCT symbol FROM price_data"))
                return [row[0] for row in result]

        engine_logger.info(f"Filtering for symbols with avg daily volume >= {min_vol}...")

        # Use a lookback period before the start date for a realistic liquidity assessment
        lookback_start = (self.start_dt - pd.Timedelta(days=60)).date().isoformat()
        lookback_end = (self.start_dt - pd.Timedelta(days=1)).date().isoformat()

        query = text("""
            SELECT symbol
            FROM (
                SELECT symbol, date, SUM(volume) as daily_volume
                FROM price_data
                WHERE date BETWEEN :start AND :end
                GROUP BY symbol, date
            ) AS daily_volumes
            GROUP BY symbol
            HAVING AVG(daily_volume) >= :min_vol
        """)

        with self.db_engine.connect() as connection:
            result = connection.execute(query, {'start': lookback_start, 'end': lookback_end, 'min_vol': min_vol})
            liquid_symbols = [row[0] for row in result]

        engine_logger.info(f"Found {len(liquid_symbols)} liquid symbols to backtest.")
        return liquid_symbols

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

        all_strategy_params = self.config['strategy'].get('parameters', {})
        active_strategy_params = all_strategy_params.get(self.strategy_name, {})

        # Determine which symbols to use: the list from the config OR the full liquid list.
        symbols_to_use = active_strategy_params.get('symbols') or self.symbols
        self.symbols = symbols_to_use # Update the engine's list of symbols

        # --- FIX APPLIED HERE ---
        # Remove 'symbols' from the params dict if it exists to prevent passing it twice.
        active_strategy_params.pop('symbols', None)

        # Now, `**active_strategy_params` will not contain a conflicting 'symbols' key.
        return strategy_class(
            symbols=symbols_to_use,
            ledger=self.ledger,
            config=self.config,
            **active_strategy_params
        )

    def _get_trade_dates(self) -> list[pd.Timestamp]:
        query = text("SELECT DISTINCT date FROM price_data WHERE date BETWEEN :start AND :end ORDER BY date;")
        with self.db_engine.connect() as conn:
            dates_df = pd.read_sql(query, conn, params={'start': self.start_dt.date().isoformat(), 'end': self.end_dt.date().isoformat()})

        trade_dates = pd.to_datetime(dates_df['date']).dt.tz_localize('America/New_York')
        return trade_dates.tolist()

    def _load_data_for_day(self, trade_date: pd.Timestamp, lookback_trading_days: int) -> dict[str, pd.DataFrame]:
        """
        Smartly loads data for the current trade_date plus the N previous *trading* days.
        """
        if not self.symbols:
            return {}

        with self.db_engine.connect() as conn:
            # Step 1: Find the exact dates of the last N trading days before the current trade_date.
            trading_dates_query = text("""
                SELECT DISTINCT date
                FROM price_data
                WHERE date < :trade_date
                ORDER BY date DESC
                LIMIT :n_days
            """)

            lookback_dates_df = pd.read_sql(
                trading_dates_query,
                conn,
                params={'trade_date': trade_date.date().isoformat(), 'n_days': lookback_trading_days}
            )

            if lookback_dates_df.empty:
                return {} # Not enough historical data to proceed

            # Create the final list of dates to fetch (lookback dates + current trade date)
            dates_to_fetch = lookback_dates_df['date'].tolist() + [trade_date.date().isoformat()]

            # Step 2: Fetch the market data for only those specific dates.
            in_params_symbols = {f"sym_{i}": sym for i, sym in enumerate(self.symbols)}
            in_params_dates = {f"d_{i}": dt for i, dt in enumerate(dates_to_fetch)}

            placeholders_symbols = ", ".join(f":{key}" for key in in_params_symbols)
            placeholders_dates = ", ".join(f":{key}" for key in in_params_dates)

            query = text(
                f"SELECT timestamp, symbol, open, high, low, close, volume FROM price_data "
                f"WHERE symbol IN ({placeholders_symbols}) AND date IN ({placeholders_dates})"
            )

            params = {**in_params_symbols, **in_params_dates}
            df = pd.read_sql(query, conn, params=params, parse_dates=['timestamp'])

        if df.empty: return {}

        # Localize or convert timestamps to UTC to match the strategy's expectation
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        df.set_index('timestamp', inplace=True)
        df.columns = [col.lower() for col in df.columns]

        return {sym: sym_df for sym, sym_df in df.groupby('symbol')}

    def run(self):
        engine_logger.info(f"Starting backtest from {self.start_dt.date()} to {self.end_dt.date()}...")
        lookback = self.strategy.get_required_lookback()

        for trade_date in tqdm(self.trading_calendar, desc="Running Backtest"):
            self.ledger.settle_funds()

            daily_data_with_lookback = self._load_data_for_day(trade_date, lookback)
            if not daily_data_with_lookback: continue

            self.strategy.on_new_day(trade_date, daily_data_with_lookback)

            bars_for_today = {}
            for sym, df in daily_data_with_lookback.items():
                today_df = df[df.index.date == trade_date.date()]
                if not today_df.empty:
                    bars_for_today[sym] = today_df

            if not bars_for_today: continue

            all_bars_today = pd.concat(bars_for_today.values()).sort_index()

            for timestamp, bar in all_bars_today.iterrows():
                symbol = bar['symbol']
                self.current_prices[symbol] = bar['close']
                self.strategy.on_bar(symbol, bar)

            self.strategy.on_session_end()
            if not all_bars_today.empty:
                self.ledger._update_equity(all_bars_today.index[-1], self.current_prices)

        engine_logger.info("Backtest finished.")
        return self.ledger