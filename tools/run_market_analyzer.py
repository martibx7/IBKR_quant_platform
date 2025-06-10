# tools/run_market_analyzer.py

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import yaml
from datetime import datetime
import pytz

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary components from the updated framework
from analytics.profiles import MarketProfiler, VolumeProfiler, get_session
from analytics.indicators import calculate_vwap

class MarketAnalyzer:
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

        df = pd.read_csv(file_path, parse_dates=['date'])
        df.columns = [col.strip().lower() for col in df.columns]
        df = df.rename(columns={'date': 'timestamp'})
        df.set_index(pd.to_datetime(df['timestamp'], utc=True), inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df[~df.index.duplicated(keep='first')].apply(pd.to_numeric)

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
        sessions = ['Pre-Market', 'Regular', 'Post-Market']

        day_df_ny = day_df.tz_convert(self.timezone)

        for session_name in sessions:
            title = f"{ticker} | {date_str} | {session_name} Session"
            print(f"\n--- Analysis for {session_name} Session ---")

            session_df = get_session(day_df_ny, analysis_date, session_name, tz_str=self.timezone.zone)

            if session_df.empty:
                print("     No data for this session.")
                continue

            market_profiler = MarketProfiler(tick_size=self.tick_size_market)
            market_profile = market_profiler.calculate(session_df)

            volume_profiler = VolumeProfiler(tick_size=self.tick_size_volume)
            volume_profile = volume_profiler.calculate(session_df)

            vwap_df = calculate_vwap(session_df.copy())
            final_vwap = vwap_df['vwap'].iloc[-1] if not vwap_df.empty else 'N/A'

            # --- ADDED: Console summary output ---
            print(f"     Time Range : {session_df.index[0].time()} - {session_df.index[-1].time()} (ET)")
            print(f"     Ending VWAP: {final_vwap:.2f}" if isinstance(final_vwap, float) else f"     Ending VWAP: {final_vwap}")

            print("     Volume Profile:")
            if volume_profile:
                print(f"       - POC: {volume_profile['poc_price']:.2f}, VAH: {volume_profile['value_area_high']:.2f}, VAL: {volume_profile['value_area_low']:.2f}")
            else:
                print("       - Not enough data to calculate.")

            print("     Market Profile (TPO):")
            if market_profile:
                print(f"       - POC: {market_profile['poc_price']:.2f}, VAH: {market_profile['value_area_high']:.2f}, VAL: {market_profile['value_area_low']:.2f}")
            else:
                print("       - Not enough data to calculate.")

            self.plot_volume_analysis(session_df, volume_profile, vwap_df, title)
            self.plot_market_analysis(session_df, market_profile, vwap_df, title)

    def _plot_candlesticks(self, ax, df, title):
        ax.set_title(title, fontsize=14)
        df_reset = df.reset_index()
        df_reset['timestamp'] = df_reset['timestamp'].map(mdates.date2num)
        candlestick_ohlc(ax, df_reset[['timestamp', 'Open', 'High', 'Low', 'Close']].values, width=0.0005, colorup='g', colordown='r', alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=self.timezone))
        ax.set_ylabel("Price ($)")
        ax.grid(True, linestyle='--', alpha=0.6)

    def plot_volume_analysis(self, df: pd.DataFrame, vol_profile_data: dict, vwap_df: pd.DataFrame, title: str):
        fig, (ax_price, ax_vol) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
        fig.suptitle(f"{title}\n(Volume Profile Analysis)", fontsize=16)

        self._plot_candlesticks(ax_price, df, "Price Action")
        ax_price.plot(vwap_df.index, vwap_df['vwap'], color='blue', linestyle='--', label='VWAP', zorder=10)

        if vol_profile_data and vol_profile_data.get('poc_price') is not None:
            ax_price.axhline(vol_profile_data['poc_price'], color='orange', linestyle='-', lw=1.5, label=f"POC: {vol_profile_data['poc_price']:.2f}")
        ax_price.legend()

        ax_vol.set_title("Volume Profile", fontsize=14)

        # --- FIX: Re-calculate the full profile for plotting ---
        if vol_profile_data:
            temp_profiler = VolumeProfiler(tick_size=self.tick_size_volume)
            full_profile_series = temp_profiler.calculate_full_profile_for_plotting(df)
            if not full_profile_series.empty:
                ax_vol.barh(full_profile_series.index, full_profile_series.values, height=self.tick_size_volume * 0.9, align='center', color='gray', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

    def plot_market_analysis(self, df: pd.DataFrame, market_profile_data: dict, vwap_df: pd.DataFrame, title: str):
        fig, (ax_tpo, ax_price) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 3]}, sharey=True)
        fig.suptitle(f"{title}\n(Market Profile Analysis)", fontsize=16)

        self._plot_candlesticks(ax_price, df, "Price Action")
        ax_price.plot(vwap_df.index, vwap_df['vwap'], color='blue', linestyle='--', label='VWAP')
        if market_profile_data and market_profile_data.get('poc_price') is not None:
            ax_price.axhline(market_profile_data['poc_price'], color='purple', linestyle='-', lw=1.5, label=f"TPO POC: {market_profile_data['poc_price']:.2f}")
        ax_price.legend()

        ax_tpo.set_title("Market Profile (TPO)", fontsize=14)
        ax_tpo.invert_xaxis()
        ax_tpo.set_xticks([])
        ax_tpo.set_ylabel("Price ($)")

        # --- FIX: Re-calculate the internal TPO profile to get the letters for plotting ---
        if not df.empty:
            temp_profiler = MarketProfiler(tick_size=self.tick_size_market)
            tpo_letters_profile = temp_profiler._calculate_tpo_profile_by_interval(df)
            if tpo_letters_profile:
                for price, tpos in sorted(tpo_letters_profile.items()):
                    # Use a smaller font size and ensure alignment
                    ax_tpo.text(0.95, price, "".join(sorted(tpos)),
                                fontsize=6, ha='right', va='center',
                                transform=ax_tpo.get_yaxis_transform())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    try:
        analyzer = MarketAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()