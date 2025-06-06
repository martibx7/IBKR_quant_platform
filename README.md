# IBKR Quant Platform

This project is a modular framework for developing, backtesting, and deploying quantitative trading strategies using Python and data from Interactive Brokers.

## Project Goal

The primary goal is to create a flexible and robust platform that supports both rigorous historical backtesting and a seamless transition to live paper/real trading.

## Current Status

The project is in the initial development phase. The core backtesting framework is complete and functional. Current capabilities include:
- Minute-by-minute, event-driven backtesting to prevent lookahead bias.
- A configurable, tiered commission model to simulate realistic trading costs.
- A dynamic results module that calculates key performance metrics (Sharpe Ratio, Max Drawdown, Win Rate, etc.).
- Pluggable strategy architecture.
- Integrated analytics for Volume Profile and Market Profile (TPO) analysis.

The immediate focus is on enhancing the backtesting engine and developing more sophisticated strategies.

## Architecture Overview

The platform is built on a modular design to separate concerns:

- **/analytics**: Contains reusable tools for market analysis (`VolumeProfiler`, `MarketProfiler`).
- **/backtest**: Houses the core backtesting engine and results analysis tools.
- **/core**: Contains essential components like the `BacktestLedger` and `TieredIBFeeModel`.
- **/strategies**: Location for all trading strategy logic. Each strategy is a class that inherits from a common `BaseStrategy`.
- **/tools**: Contains helper scripts for data acquisition.
- **/visualization**: Tools for plotting results, such as equity curves and analytical profiles.
- **`config.yaml`**: A central configuration file for managing all backtest and strategy parameters.

## Data Pipeline

Historical data is acquired via a two-step process external to the main backtesting application:

1.  **Ticker Discovery**: A custom algorithm is run on the QuantConnect platform to scan the market and identify a list of tickers that met specific criteria on given historical dates.
2.  **Data Download**: The `tools/download_data.py` script uses the list from step 1 to download 1-minute OHLCV bar data from Interactive Brokers.

The final data is stored in the `/data/historical/` directory (which is git-ignored) using the naming convention `TICKER_YYYYMMDD.csv`.

## How to Run a Backtest

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Data**: Place historical 1-minute data files into the `/data/historical/` directory.

3.  **Configure Backtest**: Edit `config.yaml` to set the initial cash, data path, symbols, and strategy parameters.

4.  **Run**: Execute the main script from the root directory.
    ```bash
    python main.py
    ```

## Next Steps

- Enhance the backtest engine to run over multiple days and symbols.
- Develop and test more complex, event-driven strategies.
- Build out the live trading components for connecting to the IBKR API using IB_async and IBC.