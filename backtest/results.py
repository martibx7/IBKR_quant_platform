# backtest/results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BacktestResults:
    """
    Analyzes, displays, and plots the results of a backtest from a BacktestLedger.
    """
    def __init__(self, ledger):
        self.ledger = ledger
        self.trade_log = ledger.get_trade_log()
        self.equity_curve = ledger.get_equity_curve()
        self.metrics = self.calculate_metrics()

    def calculate_metrics(self):
        """Calculates all key performance metrics."""
        if self.trade_log.empty:
            print("No trades were executed. Cannot calculate metrics.")
            return None

        # --- Profit & Loss --
        total_pnl = self.trade_log['pnl'].sum()
        total_fees = self.trade_log['fees'].sum()
        net_pnl = total_pnl - total_fees

        initial_equity = self.ledger.initial_cash
        final_equity = self.equity_curve['equity'].iloc[-1]

        # --- Win/Loss Analysis ---
        winning_trades = self.trade_log[self.trade_log['pnl'] > 0]
        losing_trades = self.trade_log[self.trade_log['pnl'] <= 0]

        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        total_trades = num_winning + num_losing
        win_rate = (num_winning / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = winning_trades['pnl'].mean()
        avg_loss = losing_trades['pnl'].mean()

        # --- NEW: Calculate Risk/Reward Ratio ---
        if pd.notna(avg_loss) and avg_loss != 0:
            risk_reward_ratio = abs(avg_win / avg_loss)
        else:
            # If there are no losses, R:R is infinite
            risk_reward_ratio = float('inf')

        # --- Drawdown Calculation (uses high-frequency data) ---
        self.equity_curve['peak'] = self.equity_curve['equity'].cummax()
        self.equity_curve['drawdown'] = self.equity_curve['equity'] - self.equity_curve['peak']
        max_drawdown = self.equity_curve['drawdown'].min() if not self.equity_curve['drawdown'].empty else 0
        max_drawdown_pct = (self.equity_curve['drawdown'] / self.equity_curve['peak']).min() if not self.equity_curve['drawdown'].empty else 0

        # --- Sharpe Ratio Calculation (uses resampled daily data) ---
        daily_equity = self.equity_curve['equity'].resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)

        return {
            'Initial Equity': f"${initial_equity:,.2f}",
            'Final Equity': f"${final_equity:,.2f}",
            'Total PnL': f"${total_pnl:,.2f}",
            'Total Fees': f"${total_fees:,.2f}",
            'Net PnL': f"${net_pnl:,.2f}",
            'Total Trades': total_trades,
            'Win Rate': f"{win_rate:.2f}%",
            'Profit Factor': f"{profit_factor:.2f}",
            'Avg Win': f"${avg_win:,.2f}",
            'Avg Loss': f"${avg_loss:,.2f}",
            'Reward/Risk Ratio': f"{risk_reward_ratio:.2f}:1" if risk_reward_ratio != float('inf') else 'inf',
            'Max Drawdown': f"${max_drawdown:,.2f} ({max_drawdown_pct:.2%})",
            'Sharpe Ratio (annualized)': f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else 'N/A',
        }

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        """Calculates the annualized Sharpe ratio from a series of daily returns."""
        if returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        return sharpe_ratio

    def print_summary(self):
        """Prints a formatted summary of the backtest results."""
        if not self.metrics:
            return

        print("\n--- Backtest Results ---")
        for key, value in self.metrics.items():
            print(f"{key:<28} {str(value):>15}")
        print("--------------------------------------------\n")

    def plot_equity_curve(self):
        """
        Plots the equity curve and drawdown from the ledger's history.
        """
        if self.equity_curve.empty:
            print("No history to plot for equity curve.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('Portfolio Performance', fontsize=16)

        ax1.plot(self.equity_curve.index, self.equity_curve['equity'], label='Equity', color='blue')
        ax1.plot(self.equity_curve.index, self.equity_curve['peak'], label='Peak Equity', color='green', linestyle='--')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        ax2.fill_between(self.equity_curve.index, self.equity_curve['drawdown'] * 100, 0, color='red', alpha=0.5, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown as % of Peak Equity')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()