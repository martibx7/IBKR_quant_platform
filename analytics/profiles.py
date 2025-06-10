# analytics/profiles.py

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
import string
from collections import defaultdict

def get_session_times(session_name: str) -> tuple[time, time]:
    """Returns the start and end times for a given session name."""
    session_defs = {
        'Pre-Market': (time(4, 0), time(9, 30)),
        'Regular': (time(9, 30), time(16, 0)),
        'Post-Market': (time(16, 0), time(20, 0))
    }
    return session_defs.get(session_name, (None, None))

def get_session(df: pd.DataFrame, target_date: datetime.date, session_name: str, tz_str: str = 'America/New_York') -> pd.DataFrame:
    """
    Filters a DataFrame for a specific trading session on a given date.
    """
    if df.empty:
        return pd.DataFrame()

    start_time, end_time = get_session_times(session_name)
    if not start_time or not end_time:
        return pd.DataFrame()

    timezone = pytz.timezone(tz_str)
    start_dt = timezone.localize(datetime.combine(target_date, start_time))
    end_dt = timezone.localize(datetime.combine(target_date, end_time))
    return df[(df.index >= start_dt) & (df.index < end_dt)]


class VolumeProfiler:
    """Calculates a Volume Profile from a DataFrame of market bars."""
    def __init__(self, tick_size: float):
        self.tick_size = tick_size

    def calculate(self, df: pd.DataFrame) -> dict:
        if df.empty or df['Volume'].sum() == 0:
            return None

        min_price = df['Low'].min()
        max_price = df['High'].max()

        price_levels = np.arange(min_price, max_price + self.tick_size, self.tick_size)

        # --- FIX: Initialize with a float dtype to prevent performance-killing type conversions ---
        volume_distribution = pd.Series(0.0, index=np.round(price_levels, 5))

        for _, row in df.iterrows():
            low, high, vol = row['Low'], row['High'], row['Volume']
            if vol == 0 or high <= low: continue

            # Use boolean masking for efficiency
            relevant_levels_mask = (volume_distribution.index >= low) & (volume_distribution.index <= high)
            num_levels = relevant_levels_mask.sum()

            if num_levels > 0:
                volume_per_level = vol / num_levels
                volume_distribution.loc[relevant_levels_mask] += volume_per_level

        if volume_distribution.sum() == 0: return None

        poc_price = volume_distribution.idxmax()
        total_volume = volume_distribution.sum()

        sorted_volume = volume_distribution.sort_values(ascending=False)
        cumulative_volume = sorted_volume.cumsum()
        value_area_limit = total_volume * 0.7
        value_area_prices = sorted_volume[cumulative_volume <= value_area_limit].index

        vah = value_area_prices.max()
        val = value_area_prices.min()

        return {'poc_price': poc_price, 'value_area_high': vah, 'value_area_low': val}

class MarketProfiler:
    """Calculates a comprehensive Market Profile (TPO)."""
    def __init__(self, tick_size: float = 0.05):
        self.tick_size = tick_size
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase)

    def calculate(self, df: pd.DataFrame) -> dict:
        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            return None

        tpo_profile = self._calculate_tpo_profile_by_interval(df)
        if not tpo_profile:
            return None

        poc, vah, val = self._calculate_poc_and_value_area(tpo_profile)

        return {'poc_price': poc, 'value_area_high': vah, 'value_area_low': val}

    def _calculate_tpo_profile_by_interval(self, df: pd.DataFrame) -> defaultdict:
        tpo_profile = defaultdict(list)
        session_start_time = df.index[0].floor('30min')

        for i, tpo_letter in enumerate(self.tpo_periods):
            period_start = session_start_time + pd.Timedelta(minutes=30 * i)
            period_end = period_start + pd.Timedelta(minutes=30)
            period_bars = df[(df.index >= period_start) & (df.index < period_end)]

            if period_bars.empty:
                if period_start > df.index[-1]: break
                else: continue

            for _, row in period_bars.iterrows():
                start_tick = int(row['Low'] / self.tick_size)
                end_tick = int(row['High'] / self.tick_size)
                for tick in range(start_tick, end_tick + 1):
                    price_level = round(tick * self.tick_size, 2)
                    if tpo_letter not in tpo_profile[price_level]:
                        tpo_profile[price_level].append(tpo_letter)
        return tpo_profile

    def _calculate_poc_and_value_area(self, tpo_profile: defaultdict) -> tuple:
        if not tpo_profile: return None, None, None

        tpo_counts = pd.Series({price: len(tpos) for price, tpos in tpo_profile.items()})
        poc_price = tpo_counts.idxmax()
        total_tpos = tpo_counts.sum()
        value_area_tpos = total_tpos * 0.7
        current_tpos = tpo_counts.get(poc_price, 0)
        value_area_prices = [poc_price]
        prices_above = tpo_counts[tpo_counts.index > poc_price].index
        prices_below = tpo_counts[tpo_counts.index < poc_price].sort_index(ascending=False).index
        idx_above, idx_below = 0, 0

        while current_tpos < value_area_tpos:
            vol_above = tpo_counts.get(prices_above[idx_above], 0) if idx_above < len(prices_above) else -1
            vol_below = tpo_counts.get(prices_below[idx_below], 0) if idx_below < len(prices_below) else -1
            if vol_above == -1 and vol_below == -1: break
            if vol_above > vol_below:
                current_tpos += vol_above
                value_area_prices.append(prices_above[idx_above])
                idx_above += 1
            else:
                current_tpos += vol_below
                value_area_prices.append(prices_below[idx_below])
                idx_below += 1

        return poc_price, max(value_area_prices), min(value_area_prices)