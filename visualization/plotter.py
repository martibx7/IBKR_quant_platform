# visualization/plotter.py

import matplotlib.pyplot as plt
import pandas as pd

from analytics.profiles import VolumeProfiler, MarketProfiler

def plot_volume_profile(profiler: VolumeProfiler, symbol: str, ax=None):
    """
    Plots the volume profile on a matplotlib Axes object.

    Args:
        profiler (VolumeProfiler): The calculated VolumeProfiler object.
        symbol (str): The ticker symbol for the chart title.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    profile_series = profiler.profile

    # Plot the main profile as horizontal bars
    ax.barh(profile_series.index, profile_series.values, height=profiler.tick_size, align='center', color='skyblue', alpha=0.6)

    # Highlight the Value Area
    va_prices = profile_series.loc[profiler.val:profiler.vah]
    ax.barh(va_prices.index, va_prices.values, height=profiler.tick_size, align='center', color='royalblue', alpha=0.8)

    # Highlight the Point of Control (POC)
    ax.axhline(profiler.poc_price, color='red', linestyle='--', linewidth=1.5, label=f'POC: {profiler.poc_price:.2f}')

    ax.set_ylabel('Price')
    ax.set_xlabel('Volume')
    ax.set_title(f'{symbol} Volume Profile')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Invert y-axis to have prices ascending from bottom to top
    ax.invert_yaxis()


def plot_market_profile(profiler: MarketProfiler, symbol: str, ax=None):
    """
    Plots the TPO profile on a matplotlib Axes object.

    Args:
        profiler (MarketProfiler): The calculated MarketProfiler object.
        symbol (str): The ticker symbol for the chart title.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    tpo_counts = pd.Series({price: len(tpos) for price, tpos in profiler.tpo_profile.items()})

    # Plot the main TPO profile as horizontal bars
    ax.barh(tpo_counts.index, tpo_counts.values, height=profiler.tick_size, align='center', color='lightgreen', alpha=0.6)

    # Highlight the Value Area
    va_prices = tpo_counts.loc[profiler.val:profiler.vah] # Note: TPO val/vah are inverted due to sorting
    ax.barh(va_prices.index, va_prices.values, height=profiler.tick_size, align='center', color='seagreen', alpha=0.8)

    # Highlight the Point of Control (POC)
    ax.axhline(profiler.poc_price, color='red', linestyle='--', linewidth=1.5, label=f'POC: {profiler.poc_price:.2f}')

    ax.set_ylabel('Price')
    ax.set_xlabel('TPO Count')
    ax.set_title(f'{symbol} Market Profile (TPO)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.invert_yaxis()

def plot_price_and_profiles(bars_df: pd.DataFrame, volume_profiler: VolumeProfiler, market_profiler: MarketProfiler, symbol: str):
    """
    Creates a combined plot showing price action, volume profile, and market profile.
    """
    fig = plt.figure(figsize=(20, 10))

    # Define grid layout
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])

    # Main price chart
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(bars_df['Date'], bars_df['Close'], label='Close Price')
    ax0.set_title(f'{symbol} Price Action')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Price')
    ax0.grid(True, linestyle='--', alpha=0.5)
    plt.setp(ax0.get_xticklabels(), rotation=45, ha='right')

    # Volume Profile chart
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    plot_volume_profile(volume_profiler, symbol, ax=ax1)
    plt.setp(ax1.get_yticklabels(), visible=False) # Hide y-axis labels to avoid clutter

    # Market Profile chart
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    plot_market_profile(market_profiler, symbol, ax=ax2)
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.tight_layout()
    plt.show()