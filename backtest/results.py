"""
backtest/results.py

Post-trade analytics & visualisation.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# The sqlalchemy import is no longer needed for this function
# from sqlalchemy import create_engine, text

if TYPE_CHECKING:  # avoids an import cycle
    from .ledger  import BacktestLedger
    from .engine  import BacktestEngine

log = logging.getLogger(__name__)


class BacktestResults:
    """
    Aggregates a finished BacktestLedger and produces analytics / charts.
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, ledger: "BacktestLedger", engine: "BacktestEngine" | None = None) -> None:
        self.ledger      = ledger
        self.engine      = engine

        # copies so we can munge them freely
        self.trade_log     = ledger.get_trade_log().copy()
        self.equity_curve  = ledger.get_equity_curve().copy()

        self._prepare_equity_curve()

        if not self.equity_curve.empty:
            self.equity_curve["peak"]      = self.equity_curve["equity"].cummax()
            self.equity_curve["drawdown"]  = self.equity_curve["equity"] - self.equity_curve["peak"]

        self.metrics = self._calculate_metrics()

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _prepare_equity_curve(self) -> None:
        if self.equity_curve.empty:
            log.warning("Equity curve is empty – nothing to prepare.")
            return

        self.equity_curve.sort_index(inplace=True)
        if self.equity_curve.index.has_duplicates:
            dup_ct = self.equity_curve.index.duplicated(keep="last").sum()
            log.debug(f"Removing {dup_ct:,} duplicate equity timestamps.")
            self.equity_curve = self.equity_curve[~self.equity_curve.index.duplicated(keep="last")]

        start = self.equity_curve.index.min().floor("min")
        end   = self.equity_curve.index.max().ceil("min")
        full_index = pd.date_range(start=start, end=end, freq="min", tz=self.equity_curve.index.tz)

        self.equity_curve = self.equity_curve.reindex(full_index).ffill()

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def _get_benchmark_returns(self, start_date, end_date, ticker: str = "SPY") -> pd.Series:
        """
        MODIFIED: Fetches daily % returns for a benchmark from its .feather file.
        """
        if self.engine is None:
            log.warning("BacktestEngine not provided – Alpha/Beta will be skipped.")
            return pd.Series(dtype=float)

        spy_path = os.path.join(self.engine.feather_dir, f"{ticker}.feather")
        if not os.path.exists(spy_path):
            log.warning(f"Benchmark file not found: {spy_path}. Alpha/Beta will be skipped.")
            return pd.Series(dtype=float)

        try:
            # Read the benchmark data from the feather file
            df = pd.read_feather(spy_path)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(self.engine.tz)

            # Filter for the backtest date range
            df = df.loc[start_date:end_date]
            if df.empty:
                log.warning(f"No benchmark data for {ticker} in the date range {start_date} to {end_date}.")
                return pd.Series(dtype=float)

            # Resample to get daily closing prices and calculate returns
            daily_closes = df['close'].resample('D').last().dropna()
            return daily_closes.pct_change().dropna()

        except Exception as e:
            log.warning(f"Could not load or process benchmark file {spy_path} – {e}")
            return pd.Series(dtype=float)


    def _calculate_metrics(self) -> dict | None:
        if self.trade_log.empty:
            log.warning("No trades were executed – metrics unavailable.")
            return None

        total_pnl   = self.trade_log["pnl"].sum()
        total_fees  = self.trade_log["fees"].sum()
        net_pnl     = total_pnl

        initial_eq  = self.ledger.initial_cash
        final_eq    = self.equity_curve["equity"].iloc[-1]

        wins    = self.trade_log[self.trade_log["pnl"]  > 0]
        losses  = self.trade_log[self.trade_log["pnl"] <= 0]
        n_trades = len(self.trade_log)
        win_rate = (len(wins) / n_trades) * 100 if n_trades else 0

        gross_profit = wins["pnl"].sum()
        gross_loss   = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else np.inf

        avg_win  = wins["pnl"].mean()   if not wins.empty   else 0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss else np.inf

        max_dd  = self.equity_curve["drawdown"].min()
        peak_at_dd = self.equity_curve.loc[self.equity_curve['drawdown'] == max_dd, 'peak'].iloc[0]
        max_dd_pct = max_dd / peak_at_dd if peak_at_dd else 0

        daily_eq  = self.equity_curve["equity"].resample("D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe    = self._sharpe(daily_ret)

        alpha, beta = None, None
        if not daily_ret.empty:
            start_dt = daily_ret.index.min()
            end_dt = daily_ret.index.max()
            bench = self._get_benchmark_returns(start_dt, end_dt)
            if not bench.empty:
                # Align the portfolio returns and benchmark returns
                aligned = pd.DataFrame({"portfolio": daily_ret, "benchmark": bench}).dropna()
                if len(aligned) > 1:
                    # Calculate Beta
                    covariance = aligned['portfolio'].cov(aligned['benchmark'])
                    market_variance = aligned['benchmark'].var()
                    beta = covariance / market_variance if market_variance != 0 else 0

                    # Calculate Alpha
                    risk_free_rate = 0.0 # Assuming a risk-free rate of 0 for simplicity
                    portfolio_cagr = (1 + aligned['portfolio'].mean())**252 - 1
                    benchmark_cagr = (1 + aligned['benchmark'].mean())**252 - 1
                    alpha = portfolio_cagr - (risk_free_rate + beta * (benchmark_cagr - risk_free_rate))

        return {
            "Initial Equity": initial_eq,   "Final Equity": final_eq,
            "Total PnL": total_pnl,        "Total Fees": total_fees,
            "Net PnL": net_pnl,            "Total Trades": n_trades,
            "Win Rate": win_rate,          "Profit Factor": profit_factor,
            "Avg Win": avg_win,            "Avg Loss": avg_loss,
            "Reward/Risk Ratio": rr_ratio,
            "Max Drawdown": max_dd_pct,    "Max Drawdown ($)": max_dd,
            "Sharpe Ratio (annualized)": sharpe,
            "Alpha": alpha,                "Beta": beta,
        }

    @staticmethod
    def _sharpe(ret: pd.Series, rfr: float = 0.0, periods: int = 252) -> float:
        if ret.empty or ret.std() == 0:
            return 0.0
        excess = ret - (rfr / periods)
        return (excess.mean() / excess.std()) * np.sqrt(periods)

    def get_summary_string(self) -> str:
        """Returns the formatted backtest results as a multi-line string."""
        if not self.metrics:
            return "No metrics to display."

        m = self.metrics
        dd_str = f"${m['Max Drawdown ($)']:,.2f} ({m['Max Drawdown']:.2%})"

        summary = []
        summary.append("--- Backtest Results ---")
        summary.append(f"{'Initial Equity':<28} ${m['Initial Equity']:>15,.2f}")
        summary.append(f"{'Final Equity':<28} ${m['Final Equity']:>15,.2f}")
        summary.append(f"{'Total PnL':<28} ${m['Total PnL']:>15,.2f}")
        summary.append(f"{'Total Fees':<28} ${m['Total Fees']:>15,.2f}")
        summary.append(f"{'Net PnL':<28} ${m['Net PnL']:>15,.2f}")
        summary.append(f"{'Total Trades':<28} {m['Total Trades']:>15}")
        summary.append(f"{'Win Rate':<28} {m['Win Rate']:>14.2f}%")
        summary.append(f"{'Profit Factor':<28} {m['Profit Factor']:>15.2f}")
        summary.append(f"{'Avg Win':<28} ${m['Avg Win']:>15,.2f}")
        summary.append(f"{'Avg Loss':<28} ${m['Avg Loss']:>15,.2f}")
        summary.append(f"{'Reward/Risk Ratio':<28} {m['Reward/Risk Ratio']:>14.2f}:1")
        summary.append(f"{'Max Drawdown':<28} {dd_str:>15}")
        summary.append(f"{'Sharpe Ratio (annualized)':<28} {m['Sharpe Ratio (annualized)']:>15.2f}")
        if m['Alpha'] is not None:
            summary.append(f"{'Alpha':<28} {m['Alpha']:>15.4f}")
            summary.append(f"{'Beta':<28} {m['Beta']:>15.4f}")
        summary.append("--------------------------------------------")

        return "\n".join(summary)

    # --- MODIFIED: This method now uses the string generator ---
    def print_summary(self):
        """Prints the formatted backtest results to the console."""
        print("\n" + self.get_summary_string())

    def plot_equity_curve(self) -> None:
        if self.equity_curve.empty or "drawdown" not in self.equity_curve.columns:
            log.warning("Cannot plot equity curve – missing data.")
            return

        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle("Portfolio Performance", fontsize=16)

        # equity
        ax1.plot(self.equity_curve.index, self.equity_curve["equity"],
                 label="Equity", color="blue")
        ax1.plot(self.equity_curve.index, self.equity_curve["peak"],
                 label="Peak Equity", color="green", linestyle="--")
        ax1.set_ylabel("Equity ($)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # drawdown
        ax2.fill_between(self.equity_curve.index,
                         self.equity_curve["drawdown"], 0,
                         color="red", alpha=0.5, label="Drawdown")
        ax2.set_ylabel("Drawdown ($)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def save_trade_log_to_excel(self, file_path: str):
        if self.trade_log.empty:
            print("No trades to export.")
            return
        try:
            log_to_save = self.trade_log.copy()
            time_cols = [c for c in ("entry_time", "exit_time") if c in log_to_save.columns]
            for col in time_cols:
                if pd.api.types.is_datetime64_any_dtype(log_to_save[col]) and log_to_save[col].dt.tz is not None:
                    log_to_save[col] = log_to_save[col].dt.tz_localize(None)

            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            log_to_save.to_excel(file_path, index=False, sheet_name="TradeLog")
        except ImportError:
            print("\nTo export to Excel, you need 'openpyxl'. Please run: pip install openpyxl")
        except Exception as e:
            print(f"An error occurred while exporting to Excel: {e}")