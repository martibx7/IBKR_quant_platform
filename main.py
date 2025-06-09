import os
import re
import yaml
import importlib
from datetime import datetime
from backtest.engine import BacktestEngine
from core.ledger import BacktestLedger
from core.fee_models import TieredIBFeeModel, ZeroFeeModel

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports a strategy class from the strategies directory.
    """
    module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy_name).lower()
    module_path = f"strategies.{module_name}"
    try:
        StrategyModule = importlib.import_module(module_path)
        StrategyClass = getattr(StrategyModule, strategy_name)
        return StrategyClass
    except ImportError as e:
        print(f"Error importing strategy: Could not find module at {module_path}.py")
        raise e
    except AttributeError:
        print(f"Error: Class '{strategy_name}' not found in module '{module_path}'.")
        raise

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
    start_date_str = input("Enter backtest start date (YYYY-MM-DD): ")
    end_date_str = input("Enter backtest end date (YYYY-MM-DD): ")

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.")
        return

    # --- Setup ---
    fee_model = get_fee_model_instance(fee_config['model'])
    ledger = BacktestLedger(backtest_config['initial_cash'], fee_model)

    # --- Load Strategy based on config ---
    active_strategy_name = strategy_config['name']
    print(f"\nActive Strategy: {active_strategy_name}")

    try:
        strategy_params = strategy_config['parameters'][active_strategy_name]
    except KeyError:
        print(f"ERROR: Parameters for strategy '{active_strategy_name}' not found in config.yaml under strategy.parameters.")
        return

    StrategyClass = get_strategy_class(active_strategy_name)

    # --- Discover Symbols from Data Directory ---
    try:
        data_dir = backtest_config['data_dir']
        symbols = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not symbols:
            print(f"Warning: No CSV files found in the data directory: '{data_dir}'")
    except FileNotFoundError:
        print(f"ERROR: Data directory not found at '{data_dir}'")
        return

    strategy = StrategyClass(symbols=symbols, **strategy_params, ledger=ledger)

    # --- Run Backtest ---
    engine = BacktestEngine(data_dir, strategy)
    engine.run(start_date, end_date)


if __name__ == "__main__":
    main()