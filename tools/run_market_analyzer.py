# tools/run_market_analyzer.py

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import yaml
from datetime import datetime, time
import pytz

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from analytics.profiles import MarketProfiler, VolumeProfiler, get_session
from analytics.indicators import calculate_vwap

class MarketAnalyzer:
    """
    An interactive tool to generate two detailed analysis plots for a specific
    ticker and date: one for Volume Profile and one for Market Profile.
    """
    def __init__(self, config_path='config.yaml'):
        config_path = os.path.join(project_root, 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"ERROR: Could not find config.yaml at {config_path}")
            sys.exit(1)

        self.data_dir = os.path.join(project_root, config.get('backtest', {}).get('data_dir', 'data/historical'))
        self.tick_size_market = config.get('backtest', {}).get('tick_size_market_profile', 0.05)
        self.tick_size_volume = config.get('backtest', {}).get('tick_size_volume_profile', 0.01)
        self.timezone = pytz.timezone('America/New_York')

    def _load_data_for_ticker(self, ticker: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{ticker.upper()}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        df.columns = [col.strip().lower() for col in df.columns]
        if df.index.tz is None:
            df = df.tz_localize('UTC')
        else:
            df = df.tz_convert('UTC')
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df[~df.index.duplicated(keep='first')]

    def run(self):
        ticker = input("Enter the ticker to analyze (e.g., SPY): ").upper()
        date_str = input("Enter the date to analyze (YYYY-MM-DD): ")
        try:
            analysis_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return

        print(f"\nLoading data for {ticker} on {date_str}...")
        full_df = self._load_data_for_ticker(ticker)
        day_df = full_df[full_df.index.date == analysis_date]

        if day_df.empty:
            print(f"No data available for {ticker} on {date_str}.")
            return

        print("Analyzing sessions...")
        sessions = ['Pre-Market', 'Regular', 'After-Hours']
        for session_name in sessions:
            session_df = day_df[day_df.index.to_series().apply(
                lambda dt: get_session(dt.tz_convert(self.timezone)) == session_name
            )]

            if session_df.empty:
                print(f"\n--- No data for {session_name} Session ---")
                continue

            title = f"{ticker} | {date_str} | {session_name} Session"
            print(f"\n--- Analysis for {title} ---")

            market_profiler = MarketProfiler(session_df.copy(), tick_size=self.tick_size_market)
            volume_profiler = VolumeProfiler(session_df.copy(), tick_size=self.tick_size_volume)
            vwap_df = calculate_vwap(session_df.copy())

            print(market_profiler.get_profile_str())
            self.plot_volume_analysis(session_df, volume_profiler, vwap_df, title)
            self.plot_market_analysis(session_df, market_profiler, vwap_df, title)

    def _plot_candlesticks(self, ax, df, title):
        """Helper to plot candlesticks and format axes."""
        ax.set_title(title, fontsize=14)
        df_reset = df.reset_index()
        df_reset['date'] = df_reset['date'].map(mdates.date2num)
        candlestick_ohlc(ax, df_reset[['date', 'Open', 'High', 'Low', 'Close']].values, width=0.0005, colorup='g', colordown='r', alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=self.timezone))
        ax.set_ylabel("Price ($)")
        ax.grid(True, linestyle='--', alpha=0.6)

    def plot_volume_analysis(self, df: pd.DataFrame, vol_profile: VolumeProfiler, vwap_df: pd.DataFrame, title: str):
        """Generates Plot 1: Price Action + VWAP + Volume Profile."""
        fig, (ax_price, ax_vol) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
        fig.suptitle(f"{title}\n(Volume Profile Analysis)", fontsize=16)

        self._plot_candlesticks(ax_price, df, "Price Action")
        ax_price.plot(vwap_df.index, vwap_df['vwap'], color='blue', linestyle='--', label='VWAP', zorder=10)
        if vol_profile.poc_price is not None:
            ax_price.axhline(vol_profile.poc_price, color='orange', linestyle='-', lw=1.5, label=f'POC: {vol_profile.poc_price:.2f}')
        ax_price.legend()

        ax_vol.set_title("Volume Profile", fontsize=14)
        if not vol_profile.profile.empty:
            ax_vol.barh(vol_profile.profile.index, vol_profile.profile.values, height=self.tick_size_volume * 0.9, align='center', color='gray', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

    def plot_market_analysis(self, df: pd.DataFrame, market_profile: MarketProfiler, vwap_df: pd.DataFrame, title: str):
        """Generates Plot 2: Price Action + VWAP + Market Profile (TPO)."""
        fig, (ax_tpo, ax_price) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 3]}, sharey=True)
        fig.suptitle(f"{title}\n(Market Profile Analysis)", fontsize=16)

        self._plot_candlesticks(ax_price, df, "Price Action")
        ax_price.plot(vwap_df.index, vwap_df['vwap'], color='blue', linestyle='--', label='VWAP')
        if market_profile.poc_price is not None:
            ax_price.axhline(market_profile.poc_price, color='purple', linestyle='-', lw=1.5, label=f'TPO POC: {market_profile.poc_price:.2f}')
        ax_price.legend()

        # --- MODIFIED: Plot TPO letters on the left subplot ---
        ax_tpo.set_title("Market Profile (TPO)", fontsize=14)
        ax_tpo.invert_xaxis() # Makes text align neatly from the left
        ax_tpo.set_xticks([]) # Hide x-axis numbers

        # Use a transform to place text relative to the data on the y-axis
        # This makes it robust to zooming and panning
        for price, tpos in market_profile.tpo_profile.items():
            ax_tpo.text(0.95, price, "".join(sorted(tpos)),
                        fontsize=8,
                        ha='right',
                        va='center',
                        color='blue',
                        alpha=0.7,
                        transform=ax_tpo.get_yaxis_transform())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    analyzer = MarketAnalyzer()
    analyzer.run()