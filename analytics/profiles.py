import pandas as pd
import numpy as np
import string
from scipy.signal import find_peaks
from collections import defaultdict

class VolumeProfiler:
    """
    Calculates a Volume Profile from a DataFrame of market bars.
    """
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01):
        # --- FIX IS HERE: Add a check for an empty DataFrame ---
        if bars_df.empty:
            self.profile = pd.Series(dtype=np.float64)
            self.poc_price = None
            self.vah = None
            self.val = None
            return # Exit immediately

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self.profile = None
        self._calculate_profile_vectorized()

        if self.profile is not None and not self.profile.empty:
            self._calculate_poc_and_value_area()
        else:
            self.poc_price = None
            self.vah = None
            self.val = None

    def _calculate_profile_vectorized(self):
        """
        Calculates the volume distribution efficiently using vectorized operations.
        """
        min_price = self.bars_df['Low'].min()
        max_price = self.bars_df['High'].max()

        if pd.isna(min_price) or pd.isna(max_price):
            self.profile = pd.Series(dtype=np.float64)
            return

        num_bins = int(np.ceil((max_price - min_price) / self.tick_size))
        price_bins = min_price + np.arange(num_bins + 1) * self.tick_size
        volume_profile = np.zeros(num_bins + 1)

        lows = self.bars_df['Low'].values
        highs = self.bars_df['High'].values
        volumes = self.bars_df['Volume'].values

        low_indices = np.floor((lows - min_price) / self.tick_size).astype(int)
        high_indices = np.ceil((highs - min_price) / self.tick_size).astype(int)

        for i in range(len(self.bars_df)):
            if volumes[i] == 0 or high_indices[i] <= low_indices[i]:
                continue

            num_ticks_in_bar = high_indices[i] - low_indices[i]
            volume_per_tick = volumes[i] / num_ticks_in_bar
            volume_profile[low_indices[i]:high_indices[i]] += volume_per_tick

        self.profile = pd.Series(volume_profile, index=np.round(price_bins, 2)).round().astype(int)
        self.profile = self.profile[self.profile > 0]

    def _calculate_poc_and_value_area(self, va_percentage: float = 0.70):
        if self.profile is None or self.profile.empty:
            return

        total_volume = self.profile.sum()
        self.poc_price = self.profile.idxmax()
        va_volume = total_volume * va_percentage

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
        if self.profile is None or self.profile.empty:
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
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01):
        if bars_df.empty:
            self.tpo_profile = defaultdict(list)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase) # Extended for more periods
        self._calculate_tpo_profile()
        self._calculate_tpo_poc_and_value_area()

    def _calculate_tpo_profile(self):
        """Calculates the TPO profile using the DataFrame's index for time."""
        self.tpo_profile = defaultdict(list)
        tpo_period_minutes = 30 # Standard 30-min TPO periods

        # Vectorized calculation of TPO letters for all bars at once
        time_diff_minutes = (self.bars_df.index.hour * 60 + self.bars_df.index.minute) - \
                            (self.bars_df.index[0].hour * 60 + self.bars_df.index[0].minute)
        period_indices = time_diff_minutes // tpo_period_minutes

        # Add TPO letters to a temporary column
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
        if not self.tpo_profile:
            self.poc_price, self.val, self.vah = None, None, None
            return

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