# analytics/indicators.py

import pandas as pd
import numpy as np

def calculate_vwap(df: pd.DataFrame, session: str = None) -> pd.DataFrame:
    """
    Calculates the Volume Weighted Average Price (VWAP) for a given session.
    """
    # --- FIX: Create a copy to avoid SettingWithCopyWarning ---
    df = df.copy()

    if session:
        df['session'] = df.index.to_series().apply(get_session)
        df = df[df['session'] == session]

    if df.empty:
        return df

    # Use .assign() to create all new columns in one step, which is more efficient
    # and also avoids the SettingWithCopyWarning.
    df = df.assign(
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3,
        tp_x_volume = lambda x: x['typical_price'] * x['Volume']
    )

    df = df.assign(
        cumulative_tp_x_volume = df['tp_x_volume'].cumsum(),
        cumulative_volume = df['Volume'].cumsum()
    )

    df['vwap'] = df['cumulative_tp_x_volume'] / df['cumulative_volume']

    # Drop the intermediate columns
    return df.drop(columns=['typical_price', 'tp_x_volume', 'cumulative_tp_x_volume', 'cumulative_volume'])