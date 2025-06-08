import os
import re # Import the regular expression module
import yaml
import importlib
from datetime import datetime
from backtest.engine import BacktestEngine
from core.ledger import BacktestLedger
from core.fee_models import TieredIBFeeModel, ZeroFeeModel # Import fee models directly

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports a strategy class from the strategies directory.
    """
    # --- FIX: Convert CamelCase to snake_case for the filename ---
    module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy_name).lower()
    module_path = f"strategies.{module_name}"

    try:
        StrategyModule = importlib.import_module(module_path)
        StrategyClass = getattr(StrategyModule, strategy_name)
        return StrategyClass
    except ImportError as e:
        print(f"Error importing strategy: Could not find module at {module_path}.py")
        raise e

def get_fee_model_instance(fee_model_name: str):
    """
    Returns an instance of the specified fee model.
    """
    if fee_model_name.lower() == 'ibkr_tiered':
        return TieredIBFeeModel()
    elif fee_model_name.lower() == 'zero':
        return ZeroFeeModel()
    else:
        raise ValueError(f"Fee model '{fee_model_name}' not found.")

def main():
    # --- Load Configuration ---
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Please ensure the configuration file exists.")
        return
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse config.yaml. Please check its formatting. Details: {e}")
        return

    backtest_config = config['backtest']
    strategy_config = config['strategy']
    fee_config = config['fees']

    # --- User Input for Dates ---
    start_date_str = input(f"Enter backtest start date (YYYY-MM-DD) [default: {backtest_config['start_date']}]: ") or backtest_config['start_date']
    end_date_str = input(f"Enter backtest end date (YYYY-MM-DD) [default: {backtest_config['end_date']}]: ") or backtest_config['end_date']

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    # --- Setup ---
    fee_model = get_fee_model_instance(fee_config['model'])
    ledger = BacktestLedger(backtest_config['initial_cash'], fee_model)
    StrategyClass = get_strategy_class(strategy_config['name'])

    symbols = backtest_config.get('symbols')
    if not symbols:
        try:
            symbols = [f.split('.')[0] for f in os.listdir(backtest_config['data_dir']) if f.endswith('.csv')]
        except FileNotFoundError:
            print(f"ERROR: Data directory not found at '{backtest_config['data_dir']}'")
            return

    strategy = StrategyClass(symbols=symbols, **strategy_config.get('parameters', {}), ledger=ledger)

    # --- Run Backtest ---
    engine = BacktestEngine(backtest_config['data_dir'], strategy)
    engine.run(start_date, end_date)


if __name__ == "__main__":
    main()