# analytics/relative_volume.py

import pandas as pd
from datetime import timedelta

def has_high_relative_volume(todays_intraday: pd.DataFrame,
                             hist_cum_vol_profiles: dict,
                             bar_ts: pd.Timestamp,
                             ratio_threshold: float = 2.0) -> bool:
    """
    Return True if today's cum-vol is >= `ratio_threshold` × the average
    historical cum-vol at the same time-of-day. This version uses
    pre-calculated historical profiles for high performance.
    """
    if todays_intraday.empty or not hist_cum_vol_profiles:
        return False

    today_cum_vol = todays_intraday['volume'].sum()

    minutes_since_open = (bar_ts.hour * 60 + bar_ts.minute) - 570
    if minutes_since_open < 0: return False

    cum_vol_samples = []
    for date, daily_cum_vol_series in hist_cum_vol_profiles.items():
        start_of_day = daily_cum_vol_series.index[0].replace(hour=9, minute=30, second=0)
        ts_match = start_of_day + timedelta(minutes=minutes_since_open)

        hist_val = daily_cum_vol_series.asof(ts_match)
        if pd.notna(hist_val):
            cum_vol_samples.append(hist_val)

    if not cum_vol_samples:
        return True

    avg_cum_vol = sum(cum_vol_samples) / len(cum_vol_samples)
    if avg_cum_vol == 0: return True

    current_ratio = today_cum_vol / avg_cum_vol
    return current_ratio >= ratio_threshold