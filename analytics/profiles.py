# analytics/profiles.py

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
import string
from collections import defaultdict

def get_session_times(session_name: str) -> tuple[time, time]:
    session_defs = {
        'pre-market':  (time(4, 0),  time(9, 30)),
        'regular':     (time(9, 30), time(16, 0)),
        'post-market': (time(16, 0), time(20, 0)),
    }
    return session_defs.get(session_name.lower(), (None, None))

def get_session(df: pd.DataFrame, target_date: datetime.date,
                session_name: str, tz: str = 'America/New_York') -> pd.DataFrame:
    if df.empty:
        return df
    start_t, end_t = get_session_times(session_name)
    if not start_t or not end_t:
        return pd.DataFrame()
    if isinstance(tz, str):
        tz = pytz.timezone(tz)
    start_dt = tz.localize(datetime.combine(target_date, start_t))
    end_dt   = tz.localize(datetime.combine(target_date, end_t))
    return df[(df.index >= start_dt) & (df.index < end_dt)]

class VolumeProfiler:
    def __init__(self, tick_size: float):
        self.tick_size = tick_size

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

    def calculate(self, df: pd.DataFrame) -> dict | None:
        vd = self._calculate_distribution(df)
        if vd.empty or vd.sum() == 0:
            return None
        poc = vd.idxmax()
        tv  = vd.sum()
        sv  = vd.sort_values(ascending=False)
        cum = sv.cumsum()
        va_limit = tv * 0.7
        va_sv    = sv[cum <= va_limit]
        if va_sv.empty:
            va_sv = sv.head(1)
        prices = va_sv.index
        return {
            'poc_price':        poc,
            'value_area_high':  prices.max(),
            'value_area_low':   prices.min()
        }

    def calculate_full_profile_for_plotting(self, df: pd.DataFrame) -> pd.Series:
        return self._calculate_distribution(df)

class MarketProfiler:
    def __init__(self, tick_size: float = 0.05):
        self.tick_size   = tick_size
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase)

    def calculate(self, df: pd.DataFrame) -> dict | None:
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None
        tpo = self._calculate_tpo_profile_by_interval(df)
        if not tpo:
            return None
        poc, vah, val = self._calculate_poc_and_value_area(tpo)
        return {'poc_price': poc, 'value_area_high': vah, 'value_area_low': val}

    def _calculate_tpo_profile_by_interval(self, df: pd.DataFrame) -> defaultdict:
        profile = defaultdict(list)
        start   = df.index[0].floor('30min')
        for i, letter in enumerate(self.tpo_periods):
            period_start = start + pd.Timedelta(minutes=30 * i)
            period_end   = period_start + pd.Timedelta(minutes=30)
            bars = df[(df.index >= period_start) & (df.index < period_end)]
            if bars.empty:
                if period_start > df.index[-1]:
                    break
                continue
            for _, row in bars.iterrows():
                lo = int(row['low']  / self.tick_size)
                hi = int(row['high'] / self.tick_size)
                for t in range(lo, hi + 1):
                    price = round(t * self.tick_size, 8)
                    if letter not in profile[price]:
                        profile[price].append(letter)
        return profile

    def _calculate_poc_and_value_area(self, profile: defaultdict) -> tuple:
        if not profile:
            return (None, None, None)
        counts = pd.Series({p: len(v) for p, v in profile.items()})
        poc    = counts.idxmax()
        total  = counts.sum()
        target = total * 0.7
        current = counts[poc]
        prices = [poc]
        above = counts[counts.index > poc].index
        below = counts[counts.index < poc].sort_values(ascending=False).index
        ia, ib = 0, 0
        while current < target:
            va = counts.get(above[ia], -1) if ia < len(above) else -1
            vb = counts.get(below[ib], -1) if ib < len(below) else -1
            if va == -1 and vb == -1:
                break
            if va > vb:
                current += va
                prices.append(above[ia])
                ia += 1
            else:
                current += vb
                prices.append(below[ib])
                ib += 1
        return poc, max(prices), min(prices)
