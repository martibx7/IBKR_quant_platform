# IBKR Quant Platform

## Project Overview

This project is a modular framework for developing, backtesting, and deploying quantitative trading strategies using Python. It is designed to simulate trading logic as realistically as possible, with a focus on market-generated information and advanced risk management techniques. The platform has evolved from a basic concept to a robust engine capable of running complex, multi-day backtests.

## Current Status

The core backtesting engine is fully functional and stable. The primary focus has been the development and iterative refinement of the `OpenRejectionReverseStrategy`, a sophisticated day-trading strategy inspired by the principles in James Dalton's "Mind Over Markets".

## Key Features

-   **Event-Driven Backtesting Engine**: The engine operates on a bar-by-bar basis, preventing lookahead bias and closely simulating a live trading environment. It now uses a pre-calculated trading calendar to reliably handle weekends and holidays.
-   **Realistic Cash Settlement**: The `BacktestLedger` fully simulates a T+1 cash settlement environment, where proceeds from sales are not available for trading until the next session. This is critical for accurately testing strategies in a cash account.
-   **Configurable Strategies**: All strategy parameters are externalized to `config.yaml`, allowing for easy tuning and experimentation without altering the source code.
-   **Advanced Analytics**: The platform integrates `VolumeProfiler` and `MarketProfiler` to provide strategies with market-generated context, such as Point of Control (POC) and Value Area (VA).
-   **Sophisticated Risk Management**: The active strategy employs a multi-layered position sizing model that considers:
    1.  A percentage of total equity to define the maximum risk per trade.
    2.  A maximum allocation cap to prevent over-concentration in a single position.
    3.  A final check against available settled cash, scaling the position down if necessary.
-   **Dynamic Trade Management**: The strategy includes advanced logic for managing open positions, such as a dynamic trailing stop-loss that uses the migrating Point of Control (POC) of the intraday trend.
-   **Detailed Performance Reporting**: At the end of each run, the `BacktestResults` module calculates and displays a full suite of performance metrics (PnL, Win Rate, Profit Factor, Max Drawdown, etc.) and plots the equity curve.
-   **Extensive Logging**: The engine and strategies produce detailed logs, which have been instrumental in diagnosing and fixing complex logical issues.

## Architecture Overview

The platform maintains a modular design to separate concerns:

-   **/analytics**: Contains reusable tools for market analysis (`VolumeProfiler`, `MarketProfiler`, `VWAP`).
-   **/backtest**: Houses the core backtesting engine (`engine.py`) and results analysis tools (`results.py`).
-   **/core**: Contains essential components like the `BacktestLedger` and `BaseFeeModel`.
-   **/strategies**: Location for all trading strategy logic. Each strategy inherits from `BaseStrategy`.
-   `/logs`: Output directory for the detailed strategy and engine logs.
-   **`config.yaml`**: A central configuration file for managing all backtest, strategy, and fee parameters.

## Strategy Spotlight: Open Rejection Reversal

The primary strategy developed on this platform is a long-only, day-trading strategy that operates in three phases:
1.  **The Scan (Pre-Market)**: Analyzes the previous day's data to find stocks that closed with strong volume and momentum, creating a daily watchlist.
2.  **The Setup (Market Open)**: Confirms that a candidate has opened above the prior day's value area, indicating a continuation of bullish sentiment.
3.  **The Trigger (Intraday)**: Waits for a specific pattern where the price dips below the open and then rallies back through the opening bar's high. This pattern serves as the entry trigger, with a stop-loss placed at the low of the dip.

## How to Run a Backtest

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Add Data**: Ensure your 1-minute historical data files are in the directory specified in `config.yaml` (default: `/data/historical/`).
3.  **Configure the Test**: Edit `config.yaml` to select the active strategy (`name`) and set its corresponding parameters.
4.  **Run the Engine**: Execute the main script from the project root. You will be prompted for a start and end date for the backtest.
    ```bash
    python main.py
    ```

## Future Development

-   **Strategy Refinement**: Continue to test and refine the `OpenRejectionReverseStrategy` based on backtest results.
-   **New Strategies**: Implement other strategies based on "Mind Over Markets" or different trading concepts.
-   **Live Trading Integration**: Build out the components necessary to connect the strategy logic to a live paper or real money account via the Interactive Brokers API.
-   **Portfolio-Level Analysis**: Enhance the results module to aggregate statistics across multiple backtest runs and provide portfolio-level insights.
-   **User Interface**: Consider developing a simple UI for easier configuration and monitoring of backtests.  Streamlit, maybe?