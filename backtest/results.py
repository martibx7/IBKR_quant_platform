# backtest/results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_performance_metrics(ledger, risk_free_rate: float = 0.0):
    """
    Calculates and returns a dictionary of performance metrics.
    """
    metrics = {}
    trades_df = pd.DataFrame(ledger.closed_trades)
    equity_df = pd.DataFrame(ledger.equity_curve).set_index('timestamp')

    if trades_df.empty:
        print("No closed trades to analyze.")
        return {}

    # Profit and Loss
    total_pnl = trades_df['pnl'].sum()
    metrics['Total PnL'] = total_pnl
    metrics['Total Return (%)'] = (total_pnl / ledger.initial_cash) * 100

    # Trade Stats
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    metrics['Win Rate (%)'] = (len(winning_trades) / len(trades_df)) * 100 if not trades_df.empty else 0
    metrics['Profit Factor'] = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if not losing_trades.empty and winning_trades['pnl'].sum() > 0 else 0
    metrics['Avg Win ($)'] = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    metrics['Avg Loss ($)'] = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    metrics['Risk/Reward Ratio'] = abs(metrics['Avg Win ($)'] / metrics['Avg Loss ($)']) if metrics['Avg Loss ($)'] != 0 else np.inf

    # Sharpe Ratio (assuming daily returns for simplicity)
    if not equity_df.empty:
        daily_returns = equity_df['value'].resample('D').last().pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252) # Annualized
            metrics['Sharpe Ratio'] = sharpe_ratio

    # Max Drawdown
    if not equity_df.empty:
        cumulative_max = equity_df['value'].cummax()
        drawdown = (equity_df['value'] - cumulative_max) / cumulative_max
        metrics['Max Drawdown (%)'] = drawdown.min() * 100

    return metrics

def print_performance_report(metrics: dict):
    """Prints a formatted report of performance metrics."""
    print("\n--- Performance Report ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<20} {value:.2f}")
        else:
            print(f"{key:<20} {value}")
    print("--------------------------")

def plot_equity_curve(ledger):
    """Plots the portfolio value over time."""
    equity_df = pd.DataFrame(ledger.equity_curve)
    if equity_df.empty:
        print("No equity data to plot.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(equity_df['timestamp'], equity_df['value'], label='Equity Curve')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()