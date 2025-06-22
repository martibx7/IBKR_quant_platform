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
        'pre-market':  (time(4, 0),  time(9, 30)),
        'regular':     (time(9, 30), time(16, 0)),
        'post-market': (time(16, 0), time(20, 0)),
    }
    return session_defs.get(session_name.lower(), (None, None))

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
    """
    Calculates a Volume Profile using a fast, vectorized algorithm.
    """
    def __init__(self, tick_size: float, value_area_pct: float = 0.70):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct

    def _calculate_distribution(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or df['volume'].sum() == 0:
            return pd.Series(dtype=float)
        min_p = df['low'].min()
        max_p = df['high'].max()
        levels = np.arange(
            np.floor(min_p / self.tick_size) * self.tick_size,
            np.ceil(max_p  / self.tick_size) * self.tick_size,
            self.tick_size
        )
        levels = np.round(levels, 8)
        dist   = np.zeros_like(levels, dtype=float)

        lows    = df['low'].to_numpy()
        highs   = df['high'].to_numpy()
        vols    = df['volume'].to_numpy()
        starts  = np.searchsorted(levels, lows,  side='left')
        ends    = np.searchsorted(levels, highs, side='right')
        spans   = ends - starts
        mask    = (vols > 0) & (spans > 0)
        if not mask.any():
            return pd.Series(dtype=float)

        vppl    = vols[mask] / spans[mask]
        s_idx   = starts[mask]
        e_idx   = ends[mask]
        for i, v in enumerate(vppl):
            dist[s_idx[i]:e_idx[i]] += v

        return pd.Series(dist, index=levels)

    @staticmethod
    def _classify_shape(vd: pd.Series, poc: float) -> str:
        """Return 'D', 'P', 'L', 'B', or 'T' for yesterday’s profile."""
        total = vd.sum()
        if total == 0:
            return "T"

        upper = vd[vd.index > poc].sum()
        lower = vd[vd.index < poc].sum()
        tail_ratio = abs(upper - lower) / total

        if tail_ratio < 0.10:
            return "D"

        peaks = vd.sort_values(ascending=False).head(3).index
        if len(peaks) >= 2 and abs(peaks[0] - peaks[1]) > 3 * vd.index.to_series().diff().median():
            return "B"

        return "P" if upper > lower else "L"

    def calculate(self, df: pd.DataFrame) -> dict | None:
        vd = self._calculate_distribution(df)
        if vd.empty or vd.sum() == 0:
            return None
        poc = vd.idxmax()
        tv  = vd.sum()
        sv  = vd.sort_values(ascending=False)
        cum = sv.cumsum()
        va_limit = tv * self.value_area_pct
        va_sv    = sv[cum <= va_limit]
        if va_sv.empty:
            va_sv = sv.head(1)
        prices = va_sv.index
        shape  = self._classify_shape(vd, poc)
        return {
            'poc_price':        poc,
            'value_area_high':  prices.max(),
            'value_area_low':   prices.min(), # FIXED: Added missing comma
            'shape':            shape,
        }

    def calculate_full_profile_for_plotting(self, df: pd.DataFrame) -> pd.Series:
        return self._calculate_distribution(df)

class MarketProfiler:
    """
    Calculates a comprehensive Market Profile (TPO) using a vectorized approach.
    """
    def __init__(self, tick_size: float = 0.05, value_area_pct: float = 0.70):
        self.tick_size   = tick_size
        self.value_area_pct = value_area_pct
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase)

    @staticmethod
    def _classify_shape(counts: pd.Series, poc: float) -> str:
        total = counts.sum()
        if total == 0:
            return "T"
        upper = counts[counts.index > poc].sum()
        lower = counts[counts.index < poc].sum()
        tail_ratio = abs(upper - lower) / total
        if tail_ratio < 0.10:
            return "D"
        peaks = counts.sort_values(ascending=False).head(3).index
        if len(peaks) >= 2 and abs(peaks[0] - peaks[1]) > 3 * counts.index.to_series().diff().median():
            return "B"
        return "P" if upper > lower else "L"

    def calculate(self, df: pd.DataFrame) -> dict | None:
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None

        tpo_series = self._calculate_tpo_profile_vectorized(df)

        if tpo_series is None or tpo_series.empty:
            return None

        profile_dict = tpo_series.to_dict()
        # FIXED: Unpack all four values, including the 'shape'
        poc, vah, val, shape = self._calculate_poc_and_value_area(profile_dict)

        # FIXED: Add check to handle cases where profile calculation fails
        if poc is None:
            return None

        return {'poc_price': poc, 'value_area_high': vah, 'value_area_low': val, 'shape': shape}

    def _calculate_tpo_profile_vectorized(self, df: pd.DataFrame):
        """Calculates the TPO profile using a vectorized approach."""
        start_time = df.index[0].floor('30min')
        time_deltas_seconds = (df.index - start_time).total_seconds()
        period_indices = (time_deltas_seconds / 1800).astype(int)

        period_indices = np.clip(period_indices, 0, len(self.tpo_periods) - 1)

        df = df.assign(tpo_letter=np.array(self.tpo_periods)[period_indices])

        df = df.assign(
            low_tick=(df['low'] / self.tick_size).astype(int),
            high_tick=(df['high'] / self.tick_size).astype(int)
        )

        records = [
            {'tick': tick, 'tpo': row.tpo_letter}
            for row in df.itertuples()
            for tick in range(row.low_tick, row.high_tick + 1)
        ]

        if not records: return None
        exploded_df = pd.DataFrame.from_records(records)
        tpo_profile = exploded_df.drop_duplicates().groupby('tick')['tpo'].apply(list)
        if tpo_profile.empty: return None

        tpo_profile.index = np.round(tpo_profile.index * self.tick_size, 8)
        return tpo_profile

    def _calculate_poc_and_value_area(self, profile: dict) -> tuple:
        """Calculates POC, Value Area, and shape from a profile dictionary."""
        # FIXED: Return tuple of four Nones for consistency
        if not profile: return (None, None, None, None)

        counts = pd.Series({p: len(v) for p, v in profile.items()})
        poc    = counts.idxmax()
        shape = self._classify_shape(counts, poc) # 'shape' is calculated here

        total  = counts.sum()
        target = total * self.value_area_pct
        current = counts[poc]
        prices = [poc]

        above_poc = counts.index[counts.index > poc]
        below_poc = counts.index[counts.index < poc]

        sorted_above = counts.loc[above_poc].sort_index()
        sorted_below = counts.loc[below_poc].sort_index(ascending=False)

        len_above, len_below = len(sorted_above), len(sorted_below)
        ia, ib = 0, 0
        while current < target and (ia < len_above or ib < len_below):
            price_a = sorted_above.index[ia] if ia < len_above else None
            price_b = sorted_below.index[ib] if ib < len_below else None

            if price_a is not None and (price_b is None or (abs(price_a - poc) < abs(price_b - poc))):
                current += sorted_above.iloc[ia]
                prices.append(price_a)
                ia += 1
            elif price_b is not None:
                current += sorted_below.iloc[ib]
                prices.append(price_b)
                ib += 1
            else:
                break

        # FIXED: Return the calculated 'shape' along with other values
        return poc, max(prices), min(prices), shape