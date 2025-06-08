import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_performance_metrics(ledger) -> dict:
    """
    Calculates key performance metrics from the ledger's trade history.
    """
    trade_log = ledger.get_trade_log()
    if not trade_log:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate (%)": 0,
            "Total P&L ($)": 0,
            "Average Win ($)": 0,
            "Average Loss ($)": 0,
            "Profit Factor": "N/A",
            "Sharpe Ratio": "N/A"
        }

    df = pd.DataFrame(trade_log)
    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['quantity'] - df['fees']

    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]

    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = df['pnl'].sum()

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(losing_trades['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else "inf"

    # --- Calculate Sharpe Ratio ---
    equity_df = pd.DataFrame(ledger.history)
    equity_df.dropna(subset=['timestamp'], inplace=True) # Ensure no NaT values
    equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

    sharpe_ratio = "N/A"
    if equity_df['returns'].std() > 0:
        # Assuming 252 trading days in a year for annualization
        sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()

    return {
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": f"{win_rate:.2f}",
        "Total P&L ($)": f"{total_pnl:,.2f}",
        "Average Win ($)": f"{avg_win:,.2f}",
        "Average Loss ($)": f"{avg_loss:,.2f}",
        "Profit Factor": f"{profit_factor:.2f}" if isinstance(profit_factor, float) else profit_factor,
        "Sharpe Ratio": f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, float) else sharpe_ratio
    }


def print_performance_report(metrics: dict):
    """
    Prints a formatted performance report.
    """
    print("\n--- Performance Report ---")
    for key, value in metrics.items():
        print(f"{key:<20} {value}")
    print("--------------------------\n")


def plot_equity_curve(ledger):
    """
    Plots the equity curve from the ledger's history.
    """
    if not ledger.history:
        print("No history to plot for equity curve.")
        return

    equity_df = pd.DataFrame(ledger.history)

    # --- FIX: Drop rows with no timestamp before plotting ---
    equity_df.dropna(subset=['timestamp'], inplace=True)

    if equity_df.empty:
        print("No valid history with timestamps to plot for equity curve.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df['timestamp'], equity_df['equity'], label='Equity Curve')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity (USD)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()