# main.py

import yaml
import importlib
from backtest.engine import BacktestEngine
from core.ledger import BacktestLedger
from core.fee_models import TieredIBFeeModel
# Remove profile-specific plotting for now, as it was single-ticker focused
# from analytics.profiles import VolumeProfiler, MarketProfiler
# from visualization.plotter import plot_price_and_profiles

def get_strategy_class(strategy_name: str):
    """Dynamically imports a strategy class from the strategies module."""
    try:
        # Convert CamelCase strategy name to snake_case for the filename
        module_name = ''.join(['_' + i.lower() if i.isupper() else i for i in strategy_name]).lstrip('_')
        module_path = f"strategies.{module_name}"
        strategy_module = importlib.import_module(module_path)
        return getattr(strategy_module, strategy_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find strategy '{strategy_name}'. Ensure the file is named '{module_name}.py' and the class name matches.")
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

    # The 'symbols' list is now managed by the engine/scanner, but can be passed for context
    strategy = StrategyClass(symbols=[], ledger=ledger, **strategy_config['parameters'])

    # The engine now takes the directory path
    engine = BacktestEngine(backtest_config['data_dir'], strategy)
    engine.run()

    # The post-run analysis will need to be updated for multi-asset results,
    # so we comment out the old single-ticker plots for now.
    # print("\nGenerating full session analysis plots...")
    # ...

if __name__ == '__main__':
    main()