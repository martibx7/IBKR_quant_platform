# backtest/results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine, text

class BacktestResults:
    """
    Analyzes, displays, and plots the results of a backtest from a BacktestLedger.
    """
    def __init__(self, ledger, engine=None):
        self.ledger = ledger
        self.engine = engine
        self.trade_log = ledger.get_trade_log()
        self.equity_curve = ledger.get_equity_curve()
        self.metrics = self.calculate_metrics()

    def _get_benchmark_returns(self, start_date, end_date, ticker='SPY') -> pd.Series:
        """Fetches daily returns for a benchmark ticker from the database."""
        if self.engine is None:
            print("Warning: Database engine not provided. Cannot calculate Alpha and Beta.")
            return pd.Series(dtype=float)

        query = text(f"""
            SELECT date, close FROM price_data
            WHERE symbol = :ticker AND date BETWEEN :start AND :end
            GROUP BY date
            ORDER BY date;
        """)
        try:
            with self.engine.db_engine.connect() as conn:
                benchmark_df = pd.read_sql(query, conn, params={'ticker': ticker, 'start': start_date, 'end': end_date})

            if benchmark_df.empty:
                print(f"Warning: No benchmark data found for {ticker} in the specified date range.")
                return pd.Series(dtype=float)

            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])

            # --- FIX APPLIED HERE ---
            # Localize the benchmark's naive DatetimeIndex to UTC to match the portfolio's index.
            benchmark_df['date'] = benchmark_df['date'].dt.tz_localize('UTC')

            benchmark_df.set_index('date', inplace=True)
            return benchmark_df['close'].pct_change().dropna()
        except Exception as e:
            print(f"Warning: Could not load benchmark data for {ticker}. Alpha and Beta will not be calculated. Error: {e}")
            return pd.Series(dtype=float)

    def calculate_metrics(self):
        """Calculates all key performance metrics."""
        if self.trade_log.empty:
            print("No trades were executed. Cannot calculate metrics.")
            return None

        total_pnl = self.trade_log['pnl'].sum()
        total_fees = self.trade_log['fees'].sum()
        net_pnl = total_pnl
        initial_equity = self.ledger.initial_cash
        final_equity = self.equity_curve['equity'].iloc[-1] if not self.equity_curve.empty else initial_equity

        winning_trades = self.trade_log[self.trade_log['pnl'] > 0]
        losing_trades = self.trade_log[self.trade_log['pnl'] <= 0]

        total_trades = len(self.trade_log)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        if not self.equity_curve.empty:
            self.equity_curve['peak'] = self.equity_curve['equity'].cummax()
            self.equity_curve['drawdown'] = self.equity_curve['equity'] - self.equity_curve['peak']
            max_drawdown_dollars = self.equity_curve['drawdown'].min()
            max_drawdown_pct = (self.equity_curve['drawdown'] / self.equity_curve['peak']).min()
        else:
            max_drawdown_dollars, max_drawdown_pct = 0, 0

        daily_equity = self.equity_curve['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)

        start_date = self.equity_curve.index.min().date()
        end_date = self.equity_curve.index.max().date()
        benchmark_returns = self._get_benchmark_returns(start_date, end_date)

        alpha, beta = None, None
        if not benchmark_returns.empty:
            aligned_df = pd.DataFrame({'portfolio': daily_returns, 'benchmark': benchmark_returns}).dropna()
            covariance = aligned_df['portfolio'].cov(aligned_df['benchmark'])
            market_variance = aligned_df['benchmark'].var()
            beta = covariance / market_variance if market_variance != 0 else 0
            risk_free_rate = 0.0
            expected_return = risk_free_rate + beta * (aligned_df['benchmark'].mean() * 252)
            actual_return = aligned_df['portfolio'].mean() * 252
            alpha = actual_return - expected_return

        return {
            "Initial Equity": initial_equity, "Final Equity": final_equity,
            "Total PnL": total_pnl, "Total Fees": total_fees, "Net PnL": net_pnl,
            "Total Trades": total_trades, "Win Rate": win_rate, "Profit Factor": profit_factor,
            "Avg Win": avg_win, "Avg Loss": avg_loss, "Reward/Risk Ratio": reward_risk_ratio,
            "Max Drawdown": max_drawdown_pct, "Max Drawdown ($)": max_drawdown_dollars,
            "Sharpe Ratio (annualized)": sharpe_ratio,
            "Alpha": alpha, "Beta": beta
        }

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        if returns.empty or returns.std() == 0: return 0.0
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    def print_summary(self):
        if not self.metrics: return
        print("\n--- Backtest Results ---")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if key in ['Win Rate', 'Max Drawdown']:
                    print(f"{key:<28} {value:.2f}%")
                elif key in ['Max Drawdown ($)']:
                    print(f"{key:<28} ${value:,.2f}")
                else:
                    print(f"{key:<28} {value:.2f}")
            else:
                print(f"{key:<28} {str(value)}")
        print("--------------------------------------------\n")

    def plot_equity_curve(self):
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
        ax2.fill_between(self.equity_curve.index, self.equity_curve['drawdown'], 0, color='red', alpha=0.5, label='Drawdown')
        ax2.set_ylabel('Drawdown ($)')
        ax2.set_title('Drawdown')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def save_trade_log_to_excel(self, file_path: str):
        """Saves the detailed trade log to an Excel file."""
        if self.trade_log.empty:
            print("No trades to export.")
            return
        try:
            log_to_save = self.trade_log.copy()
            if 'entry_time' in log_to_save.columns:
                log_to_save['entry_time'] = log_to_save['entry_time'].dt.tz_localize(None)
            if 'exit_time' in log_to_save.columns:
                log_to_save['exit_time'] = log_to_save['exit_time'].dt.tz_localize(None)

            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            log_to_save.to_excel(file_path, index=False, sheet_name='TradeLog')
            print(f"Trade log successfully exported to {file_path}")
        except ImportError:
            print("\nTo export to Excel, you need 'openpyxl'. Please run: pip install openpyxl")
        except Exception as e:
            print(f"An error occurred while exporting to Excel: {e}")