import yaml
import importlib
import pandas as pd
from backtest.engine import BacktestEngine
from core.ledger import BacktestLedger
from core.fee_models import TieredIBFeeModel
from analytics.profiles import VolumeProfiler, MarketProfiler
from visualization.plotter import plot_price_and_profiles

def get_strategy_class(strategy_name: str):
    """Dynamically imports a strategy class from the strategies module."""
    try:
        # Assumes the file is named after the strategy in snake_case
        module_name = f"simple_poc_cross_strategy" # Example, could be made dynamic
        module_path = f"strategies.{module_name}"
        strategy_module = importlib.import_module(module_path)
        return getattr(strategy_module, strategy_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find strategy '{strategy_name}'. Please check the class and file name.")
        print(e)
        return None

def main():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        return

    backtest_config = config['backtest']
    strategy_config = config['strategy']

    fee_model = TieredIBFeeModel()
    ledger = BacktestLedger(backtest_config['initial_cash'], fee_model)
    StrategyClass = get_strategy_class(strategy_config['name'])

    if not StrategyClass: return

    strategy = StrategyClass(symbols=backtest_config['symbols'], ledger=ledger, **strategy_config['parameters'])
    engine = BacktestEngine(backtest_config['data_path'], strategy)
    engine.run()

    print("\nGenerating full session analysis plots...")
    full_day_df = pd.read_csv(backtest_config['data_path'])
    full_day_df['Date'] = pd.to_datetime(full_day_df['Date'])

    volume_profiler = VolumeProfiler(full_day_df, tick_size=0.01)
    market_profiler = MarketProfiler(full_day_df, tick_size=0.01)

    plot_price_and_profiles(
        bars_df=full_day_df,
        volume_profiler=volume_profiler,
        market_profiler=market_profiler,
        symbol=backtest_config['symbols'][0]
    )

if __name__ == '__main__':
    main()