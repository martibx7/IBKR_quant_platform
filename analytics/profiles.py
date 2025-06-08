import pandas as pd
import numpy as np
import string
from scipy.signal import find_peaks
from collections import defaultdict

def get_session(dt):
    """
    Classifies a datetime object into a trading session.
    """
    # This assumes the datetime is already in the correct local timezone (e.g., America/New_York)
    if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
        return 'Pre-Market'
    elif dt.hour >= 16:
        return 'After-Hours'
    else:
        return 'Regular'

class VolumeProfiler:
    """
    Calculates a Volume Profile from a DataFrame of market bars for a specific session.
    """
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01, session: str = None):
        if bars_df.empty:
            self.profile = pd.Series(dtype=np.float64)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        if session:
            if not isinstance(self.bars_df.index, pd.DatetimeIndex):
                raise TypeError("bars_df must have a DatetimeIndex to filter by session.")
            self.bars_df['session'] = self.bars_df.index.to_series().apply(get_session)
            self.bars_df = self.bars_df[self.bars_df['session'] == session]

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
        if self.bars_df.empty:
            self.profile = pd.Series(dtype=np.float64)
            return

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

        # --- FIX: Handle cases where idxmax returns a Series (multiple POCs) ---
        poc = self.profile.idxmax()
        if isinstance(poc, pd.Series):
            poc = poc.iloc[0] # Just take the first one in case of a tie
        self.poc_price = poc

        va_volume = total_volume * va_percentage

        try:
            # get_loc can also return a slice or a boolean array if the index is not unique
            loc_result = self.profile.index.get_loc(self.poc_price)
            if isinstance(loc_result, (slice, np.ndarray)):
                poc_index_pos = loc_result.start # Take the first position
            else:
                poc_index_pos = loc_result
        except KeyError:
            self.vah = self.val = self.poc_price
            return

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
        if self.profile is None or self.profile.empty or self.poc_price is None:
            return np.array([]), np.array([])

        poc_volume = self.profile.loc[self.poc_price]
        if isinstance(poc_volume, pd.Series): # Handle multiple POCs
            poc_volume = poc_volume.iloc[0]

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
    """Calculates a Market Profile (TPO) from a DataFrame of market bars for a specific session."""
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01, session: str = 'Regular'):
        if bars_df.empty:
            self.tpo_profile = defaultdict(list)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        if not isinstance(self.bars_df.index, pd.DatetimeIndex):
            raise TypeError("bars_df must have a DatetimeIndex to filter by session.")

        self.tick_size = tick_size
        self.session = session
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase)
        self._calculate_tpo_profile()
        self._calculate_tpo_poc_and_value_area()

    def _calculate_tpo_profile(self):
        """Calculates the TPO profile using the DataFrame's index for time."""
        self.tpo_profile = defaultdict(list)
        tpo_period_minutes = 30

        self.bars_df['session'] = self.bars_df.index.to_series().apply(get_session)
        session_bars = self.bars_df[self.bars_df['session'] == self.session].copy()

        if session_bars.empty:
            return

        time_diff_minutes = (session_bars.index.hour * 60 + session_bars.index.minute) - \
                            (session_bars.index[0].hour * 60 + session_bars.index[0].minute)
        period_indices = time_diff_minutes // tpo_period_minutes

        session_bars['tpo_letter'] = [self.tpo_periods[i] if 0 <= i < len(self.tpo_periods) else '' for i in period_indices]

        for _, row in session_bars.iterrows():
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

        # --- FIX: Handle cases where idxmax returns a Series (multiple POCs) ---
        poc = tpo_counts.idxmax()
        if isinstance(poc, pd.Series):
            poc = poc.iloc[0]
        self.poc_price = poc

        va_tpos = total_tpos * va_percentage

        try:
            loc_result = tpo_counts.index.get_loc(self.poc_price)
            if isinstance(loc_result, (slice, np.ndarray)):
                poc_index_pos = loc_result.start
            else:
                poc_index_pos = loc_result
        except KeyError:
            self.vah = self.val = self.poc_price
            return

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