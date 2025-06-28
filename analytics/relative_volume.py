# analytics/relative_volume.py

import pandas as pd
from datetime import timedelta

def has_high_relative_volume(
        current_cumulative_volume: float,
        hist_cum_vol_profiles: dict,
        bar_time: pd.Timestamp.time,
        ratio_threshold: float = 2.0
) -> bool:
    """
    Checks if the current cumulative volume is significantly higher than the
    historical average at the same time of day.
    """
    if not hist_cum_vol_profiles:
        return True # If no historical data, pass the filter

    # Find the closest historical time key
    closest_hist_time = min(hist_cum_vol_profiles.keys(), key=lambda t: abs(
        (pd.Timestamp.combine(pd.Timestamp.today().date(), t) -
         pd.Timestamp.combine(pd.Timestamp.today().date(), bar_time)).total_seconds()
    ))

    historical_avg_vol = hist_cum_vol_profiles.get(closest_hist_time)

    if not historical_avg_vol or historical_avg_vol == 0:
        return True # Avoid division by zero, pass the filter

    # Calculate the relative volume ratio
    rvol_ratio = current_cumulative_volume / historical_avg_vol

    return rvol_ratio >= ratio_threshold