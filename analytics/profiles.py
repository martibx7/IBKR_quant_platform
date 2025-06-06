# analytics/profiles.py

import pandas as pd
import numpy as np
import string
from scipy.signal import find_peaks
from collections import defaultdict

class VolumeProfiler:
    """
    Calculates a Volume Profile from a DataFrame of market bars.
    This version is optimized for performance using vectorized operations.
    """

    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01):
        if bars_df.empty:
            self.profile = pd.Series(dtype=np.float64)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self._calculate_profile_vectorized()
        self._calculate_poc_and_value_area()

    def _calculate_profile_vectorized(self):
        """
        --- Refactoring Note ---
        This vectorized version replaces the slow iterrows() loop. It calculates
        the volume distribution much more efficiently.
        """
        # 1. Determine the overall price range and create price bins
        min_price = self.bars_df['Low'].min()
        max_price = self.bars_df['High'].max()

        # Total number of price bins
        num_bins = int(np.ceil((max_price - min_price) / self.tick_size))
        price_bins = min_price + np.arange(num_bins + 1) * self.tick_size
        volume_profile = np.zeros(num_bins + 1)

        # 2. Get data as numpy arrays for speed
        lows = self.bars_df['Low'].values
        highs = self.bars_df['High'].values
        volumes = self.bars_df['Volume'].values

        # 3. Calculate bin indices for each bar's high and low
        low_indices = np.floor((lows - min_price) / self.tick_size).astype(int)
        high_indices = np.ceil((highs - min_price) / self.tick_size).astype(int)

        # 4. Distribute volume for each bar
        for i in range(len(self.bars_df)):
            if volumes[i] == 0:
                continue

            # Add 1 to include the top bin
            num_ticks_in_bar = high_indices[i] - low_indices[i] + 1
            volume_per_tick = volumes[i] / num_ticks_in_bar

            # Use numpy slicing to add volume to the correct bins at once
            volume_profile[low_indices[i]:high_indices[i]+1] += volume_per_tick

        self.profile = pd.Series(volume_profile, index=np.round(price_bins, 2)).round().astype(int)
        self.profile = self.profile[self.profile > 0] # Remove zero-volume rows

    def _calculate_poc_and_value_area(self, va_percentage: float = 0.70):
        if self.profile.empty:
            return

        total_volume = self.profile.sum()
        self.poc_price = self.profile.idxmax()
        va_volume = total_volume * va_percentage

        # The rest of this logic is standard and remains the same
        poc_index_pos = self.profile.index.get_loc(self.poc_price)
        lower_bound_pos = poc_index_pos
        upper_bound_pos = poc_index_pos
        current_va_volume = self.profile.iloc[poc_index_pos]

        while current_va_volume < va_volume and (lower_bound_pos > 0 or upper_bound_pos < len(self.profile) - 1):
            one_down_vol = self.profile.iloc[lower_bound_pos - 1] if lower_bound_pos > 0 else -1
            one_up_vol = self.profile.iloc[upper_bound_pos + 1] if upper_bound_pos < len(self.profile) - 1 else -1

            if one_down_vol > one_up_vol:
                current_va_volume += one_down_vol
                lower_bound_pos -= 1
            else:
                current_va_volume += one_up_vol
                upper_bound_pos += 1

        self.val = self.profile.index[lower_bound_pos]
        self.vah = self.profile.index[upper_bound_pos]

    def get_volume_nodes(self, min_prominence_ratio: float = 0.1, min_distance_ticks: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        --- Refactoring Note ---
        Prominence is now a ratio of the POC's volume, making it adaptive
        to different stocks and volatility levels instead of a fixed number.
        """
        if self.profile.empty:
            return np.array([]), np.array([])

        poc_volume = self.profile.loc[self.poc_price]
        min_prominence = poc_volume * min_prominence_ratio

        hvn_indices, _ = find_peaks(self.profile.values, prominence=min_prominence, distance=min_distance_ticks)
        hvn_prices = self.profile.index[hvn_indices]

        lvn_prices = []
        for i in range(len(hvn_indices) - 1):
            start = hvn_indices[i]
            end = hvn_indices[i+1]
            lvn_index_local = self.profile.iloc[start:end].idxmin()
            lvn_prices.append(lvn_index_local)

        return hvn_prices, np.array(lvn_prices)

class MarketProfiler:
    """Calculates a Market Profile (TPO) from a DataFrame of market bars."""

    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01, session_start: str = '09:30', tpo_period_minutes: int = 30):
        if bars_df.empty:
            self.tpo_profile = defaultdict(list)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self.session_start_time = pd.to_datetime(session_start).time()
        self.tpo_period_minutes = tpo_period_minutes
        self.tpo_periods = list(string.ascii_uppercase)

        self._calculate_tpo_profile()
        self._calculate_tpo_poc_and_value_area()

    def _get_tpo_period(self, bar_time: pd.Timestamp) -> str:
        minutes_from_start = (bar_time.hour * 60 + bar_time.minute) - \
                             (self.session_start_time.hour * 60 + self.session_start_time.minute)
        period_index = minutes_from_start // self.tpo_period_minutes
        return self.tpo_periods[period_index] if 0 <= period_index < len(self.tpo_periods) else ''

    def _calculate_tpo_profile(self):
        """
        --- Refactoring Note ---
        Slightly optimized to calculate TPO letters once and then process.
        The nature of TPO is more iterative, but this is cleaner.
        """
        self.tpo_profile = defaultdict(list)

        # Vectorized calculation of TPO letters for all bars at once
        time_diff_minutes = (self.bars_df['Date'].dt.hour * 60 + self.bars_df['Date'].dt.minute) - \
                            (self.session_start_time.hour * 60 + self.session_start_time.minute)
        period_indices = time_diff_minutes // self.tpo_period_minutes

        # Map indices to letters, handling out-of-bounds cases
        self.bars_df['tpo_letter'] = [self.tpo_periods[i] if 0 <= i < len(self.tpo_periods) else '' for i in period_indices]

        for _, row in self.bars_df.iterrows():
            if not row['tpo_letter']: continue

            start_price = round(row['Low'] / self.tick_size) * self.tick_size
            end_price = round(row['High'] / self.tick_size) * self.tick_size

            for price in np.arange(start_price, end_price + self.tick_size, self.tick_size):
                price_level = round(price, 2)
                if row['tpo_letter'] not in self.tpo_profile[price_level]:
                    self.tpo_profile[price_level].append(row['tpo_letter'])

    def _calculate_tpo_poc_and_value_area(self, va_percentage: float = 0.70):
        # This logic is sound and remains the same
        if not self.tpo_profile: return
        tpo_counts = pd.Series({price: len(tpos) for price, tpos in self.tpo_profile.items()})
        tpo_counts.sort_index(ascending=False, inplace=True)

        total_tpos = tpo_counts.sum()
        self.poc_price = tpo_counts.idxmax()
        va_tpos = total_tpos * va_percentage

        poc_index_pos = tpo_counts.index.get_loc(self.poc_price)
        lower_bound_pos, upper_bound_pos = poc_index_pos, poc_index_pos
        current_va_tpos = tpo_counts.iloc[poc_index_pos]

        while current_va_tpos < va_tpos and (lower_bound_pos > 0 or upper_bound_pos < len(tpo_counts) - 1):
            one_down_tpos = tpo_counts.iloc[lower_bound_pos - 1] if lower_bound_pos > 0 else -1
            one_up_tpos = tpo_counts.iloc[upper_bound_pos + 1] if upper_bound_pos < len(tpo_counts) - 1 else -1

            if one_down_tpos > one_up_tpos:
                current_va_tpos += one_down_tpos; lower_bound_pos -= 1
            else:
                current_va_tpos += one_up_tpos; upper_bound_pos += 1

        self.val = tpo_counts.index[upper_bound_pos]
        self.vah = tpo_counts.index[lower_bound_pos]

# --- Example Usage (Combined and cleaned) ---
if __name__ == '__main__':
    try:
        bars_df = pd.read_csv('AAT_20230101_1min.csv')
        bars_df['Date'] = pd.to_datetime(bars_df['Date'], utc=True)
    except FileNotFoundError:
        print("AAT_20230101_1min.csv not found. Exiting.")
        exit()

    print("--- Full Session Analysis ---")
    volume_profiler = VolumeProfiler(bars_df, tick_size=0.01)
    market_profiler = MarketProfiler(bars_df, tick_size=0.01)

    print("\n--- Volume Profile ---")
    print(f"POC: {volume_profiler.poc_price}")
    print(f"Value Area: {volume_profiler.val} - {volume_profiler.vah}")
    hvns, lvns = volume_profiler.get_volume_nodes(min_prominence_ratio=0.05)
    print(f"High Volume Nodes: {hvns}")
    print(f"Low Volume Nodes: {lvns}")

    print("\n--- Market Profile (TPO) ---")
    print(f"POC: {market_profiler.poc_price}")
    print(f"Value Area: {market_profiler.val} - {market_profiler.vah}")