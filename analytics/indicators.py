# analytics/indicators.py

import pandas as pd
import numpy as np

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Volume-Weighted Average Price (VWAP) for an intraday DataFrame.
    The DataFrame must have 'High', 'Low', 'Close', and 'Volume' columns.
    """
    # Typical Price for each bar
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3

    # Cumulative sum of Typical Price * Volume
    cumulative_tpv = (typical_price * df['Volume']).cumsum()

    # Cumulative sum of Volume
    cumulative_volume = df['Volume'].cumsum()

    # Calculate VWAP, handling potential division by zero
    df['VWAP'] = cumulative_tpv / cumulative_volume
    # Fill any initial NaN values (for the first bar if volume is 0)
    df['VWAP'].fillna(method='ffill', inplace=True)

    return df