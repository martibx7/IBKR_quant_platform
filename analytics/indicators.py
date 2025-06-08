# analytics/indicators.py

import pandas as pd
import numpy as np

def calculate_vwap(df: pd.DataFrame, session: str = None) -> pd.DataFrame:
    """
    Calculates the Volume Weighted Average Price (VWAP) for a given session.
    """
    if session:
        df['session'] = df.index.to_series().apply(get_session)
        df = df[df['session'] == session]

    if df.empty:
        return df

    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_x_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_tp_x_volume'] = df['tp_x_volume'].cumsum()
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['vwap'] = df['cumulative_tp_x_volume'] / df['cumulative_volume']

    return df.drop(columns=['typical_price', 'tp_x_volume', 'cumulative_tp_x_volume', 'cumulative_volume'])

    return df