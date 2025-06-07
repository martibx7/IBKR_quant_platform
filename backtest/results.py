# backtest/results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.ledger import BacktestLedger

def calculate_performance_metrics(ledger: BacktestLedger) -> dict:
    """
    Calculates a dictionary of performance metrics from the backtest ledger.
    """
    metrics = {}
    initial_cash = ledger.initial_cash
    equity_df = pd.DataFrame(ledger.equity_curve)

    # --- Return and PnL ---
    final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_cash
    metrics['total_pnl'] = final_equity - initial_cash
    metrics['total_return_pct'] = (metrics['total_pnl'] / initial_cash) * 100

    # --- Broker Fees ---
    metrics['total_fees'] = ledger.total_fees

    # --- Trade Stats ---
    sells = [t for t in ledger.trade_history if t.get('order_type') == 'SELL']
    wins = [t for t in sells if t['pnl'] > 0]
    losses = [t for t in sells if t['pnl'] <= 0]

    metrics['num_trades'] = len(sells)
    metrics['num_wins'] = len(wins)
    metrics['num_losses'] = len(losses)

    metrics['win_rate_pct'] = (metrics['num_wins'] / metrics['num_trades']) * 100 if metrics['num_trades'] > 0 else 0

    total_profit = sum(t['pnl'] for t in wins)
    total_loss = abs(sum(t['pnl'] for t in losses))

    metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else np.inf
    metrics['avg_win_dollars'] = total_profit / metrics['num_wins'] if metrics['num_wins'] > 0 else 0
    metrics['avg_loss_dollars'] = total_loss / metrics['num_losses'] if metrics['num_losses'] > 0 else 0
    metrics['risk_reward_ratio'] = metrics['avg_win_dollars'] / metrics['avg_loss_dollars'] if metrics['avg_loss_dollars'] > 0 else np.inf

    # --- Drawdown Statistics ---
    if not equity_df.empty:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown_dollars'] = equity_df['equity'] - equity_df['peak']
        equity_df['drawdown_pct'] = (equity_df['drawdown_dollars'] / equity_df['peak']) * 100

        metrics['max_drawdown_dollars'] = equity_df['drawdown_dollars'].min()
        metrics['max_drawdown_pct'] = equity_df['drawdown_pct'].min()
    else:
        metrics['max_drawdown_dollars'] = 0
        metrics['max_drawdown_pct'] = 0

    return metrics

def print_performance_report(metrics: dict):
    """
    Prints a formatted performance report from a metrics dictionary.
    """
    print("\n--- Performance Report ---")
    print(f"Total PnL:              ${metrics.get('total_pnl', 0):.2f}")
    print(f"Total Return (%):       {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Total Broker Fees:      ${metrics.get('total_fees', 0):.2f}")
    print("-" * 30)
    print(f"Total Trades:           {metrics.get('num_trades', 0)}")
    print(f"Win Rate (%):           {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"Profit Factor:          {metrics.get('profit_factor', 0):.2f}")
    print(f"Avg Win ($):            ${metrics.get('avg_win_dollars', 0):.2f}")
    print(f"Avg Loss ($):           ${metrics.get('avg_loss_dollars', 0):.2f}")
    print(f"Risk/Reward Ratio:      {metrics.get('risk_reward_ratio', 0):.2f}")
    print("-" * 30)
    print(f"Max Drawdown ($):       ${metrics.get('max_drawdown_dollars', 0):.2f}")
    print(f"Max Drawdown (%):       {metrics.get('max_drawdown_pct', 0):.2f}%")
    print("-" * 30)

def plot_equity_curve(ledger: BacktestLedger):
    """
    Plots the equity curve over time.
    """
    if not ledger.equity_curve:
        print("No equity data to plot.")
        return

    equity_df = pd.DataFrame(ledger.equity_curve)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

    plt.figure(figsize=(12, 8))
    plt.plot(equity_df['timestamp'], equity_df['equity'], label='Equity Curve')
    plt.title('Portfolio Equity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.show()