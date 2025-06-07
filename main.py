# main.py

import yaml
import importlib
import datetime # Import datetime
from backtest.engine import BacktestEngine
from core.ledger import BacktestLedger
from core.fee_models import TieredIBFeeModel

def get_strategy_class(strategy_name: str):
    # ... (this function remains the same) ...
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

    # --- Prompt for backtest date range ---
    start_date_str = input("Enter backtest start date (YYYY-MM-DD): ")
    end_date_str = input("Enter backtest end date (YYYY-MM-DD): ")
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    fee_model = TieredIBFeeModel()
    ledger = BacktestLedger(backtest_config['initial_cash'], fee_model)
    StrategyClass = get_strategy_class(strategy_config['name'])

    if not StrategyClass: return

    strategy = StrategyClass(symbols=[], ledger=ledger, **strategy_config['parameters'])

    engine = BacktestEngine(backtest_config['data_dir'], strategy)

    # Pass the dates to the run method
    engine.run(start_date, end_date)

if __name__ == '__main__':
    main()