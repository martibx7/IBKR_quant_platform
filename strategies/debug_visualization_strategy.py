import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from strategies.base import BaseStrategy
from analytics.profiles import MarketProfiler, VolumeProfiler, get_session
from analytics.indicators import calculate_vwap
import pytz

class DebugVisualizationStrategy(BaseStrategy):
    """
    A non-trading strategy that generates a detailed, multi-session plot to
    verify that indicators are being calculated independently for Pre-Market,
    Regular, and After-Hours sessions.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        if len(self.symbols) > 1:
            print("WARNING: This strategy is designed for a single symbol. Using the first one provided.")
        self.symbol = self.symbols[0]
        self.daily_bars = []
        self.tick_size = kwargs.get('tick_size', 0.05)
        self.tz_str = kwargs.get('timezone', 'America/New_York')
        self.timezone = pytz.timezone(self.tz_str)

    def scan_for_candidates(self, trade_date, historical_data: dict[str, pd.DataFrame]):
        return self.symbols

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        self.daily_bars = []

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict, analytics: dict):
        if self.symbol in current_bar_data:
            self.daily_bars.append(current_bar_data[self.symbol])

    def on_session_end(self):
        """Calculates indicators for each session and generates the multi-plot."""
        if not self.daily_bars:
            print("No bars collected for the day. Cannot generate plot.")
            return

        day_df = pd.DataFrame(self.daily_bars)
        date_str = day_df.index[0].date()
        print(f"\n--- Generating Multi-Session Debug Plot for {self.symbol} on {date_str} ---")

        sessions = ['Pre-Market', 'Regular', 'After-Hours']

        # Create a figure with 3 rows, 1 for each session
        fig, axes = plt.subplots(3, 1, figsize=(20, 24), sharex=True)
        fig.suptitle(f'{self.symbol} - {date_str} | Multi-Session Analysis (Tick Size: ${self.tick_size})', fontsize=20)

        for i, session_name in enumerate(sessions):
            ax_price = axes[i]

            # Filter the day's data for the specific session
            session_df = day_df[day_df.index.to_series().apply(
                lambda dt: get_session(dt.tz_convert(self.timezone)) == session_name
            )]

            if session_df.empty:
                ax_price.set_title(f'{session_name}: No Data', fontsize=14)
                ax_price.grid(True)
                continue

            # Calculate indicators for this session only
            vwap_df = calculate_vwap(session_df.copy())
            vol_profile = VolumeProfiler(session_df, tick_size=self.tick_size)

            # Plot the summary for this session
            self.plot_session_summary(ax_price, session_df, vwap_df, vol_profile, session_name)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
        plt.show()

    def plot_session_summary(self, ax_price, price_df, vwap_df, vol_profile, session_name):
        """Plots a single session's data on a given axes object."""
        ax_price.set_title(f'{session_name} Session', fontsize=14)

        # Candlestick
        width = 0.0005
        up = price_df[price_df.Close >= price_df.Open]
        down = price_df[price_df.Close < price_df.Open]
        ax_price.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.7)
        ax_price.bar(up.index, up.High - up.Close, width/4, bottom=up.Close, color='green', alpha=0.7)
        ax_price.bar(up.index, up.Low - up.Open, width/4, bottom=up.Open, color='green', alpha=0.7)
        ax_price.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.7)
        ax_price.bar(down.index, down.High - down.Open, width/4, bottom=down.Open, color='red', alpha=0.7)
        ax_price.bar(down.index, down.Low - down.Close, width/4, bottom=down.Close, color='red', alpha=0.7)

        # VWAP and Profile Lines
        ax_price.plot(vwap_df.index, vwap_df['vwap'], color='blue', linestyle='--', label='VWAP', zorder=10)
        if vol_profile.poc_price is not None:
            ax_price.axhline(vol_profile.vah, color='green', linestyle=':', lw=1.2, label=f'VAH: {vol_profile.vah:.2f}')
            ax_price.axhline(vol_profile.val, color='red', linestyle=':', lw=1.2, label=f'VAL: {vol_profile.val:.2f}')
            ax_price.axhline(vol_profile.poc_price, color='orange', linestyle='-', lw=1.2, label=f'POC: {vol_profile.poc_price:.2f}')
        ax_price.legend(loc='upper left')
        ax_price.set_ylabel('Price ($)')
        ax_price.grid(True, linestyle='--', alpha=0.6)

        # Overlay Volume Profile from the left
        ax_vol = ax_price.twiny()
        ax_vol.set_xticks([])
        profile_data = vol_profile.profile

        if not profile_data.empty:
            time_span = mdates.date2num(price_df.index[-1]) - mdates.date2num(price_df.index[0])
            max_bar_width = time_span * 0.3
            normalized_volumes = (profile_data.values / profile_data.max()) * max_bar_width
            ax_vol.barh(profile_data.index, normalized_volumes, left=mdates.date2num(price_df.index[0]), height=self.tick_size * 0.9, align='center', color='gray', alpha=0.5, zorder=5)

        # Format the shared x-axis
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=self.timezone))
        ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax_price.get_xticklabels(), rotation=0, ha='center')