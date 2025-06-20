# analytics/indicators.py

import pandas as pd
import numpy as np
# --- UPDATED: Import the new day moving average function ---
from .moving_averages import calculate_sma, calculate_ema, calculate_day_moving_average

def calculate_vwap(df: pd.DataFrame, session: str = None) -> pd.DataFrame:
    """
    Calculates the Volume Weighted Average Price (VWAP) for a given session.
    """
    # --- FIX: Create a copy to avoid SettingWithCopyWarning ---
    df = df.copy()

    if session:
        # This function 'get_session' is assumed to be available in the context where this is called
        # e.g., from analytics.profiles import get_session
        df['session'] = df.index.to_series().apply(get_session)
        df = df[df['session'] == session]

    if df.empty:
        return df

    # Use .assign() to create all new columns in one step, which is more efficient
    # and also avoids the SettingWithCopyWarning.
    df = df.assign(
        typical_price = (df['high'] + df['low'] + df['close']) / 3,
        tp_x_volume = lambda x: x['typical_price'] * x['volume']
    )

    df = df.assign(
        cumulative_tp_x_volume = df['tp_x_volume'].cumsum(),
        cumulative_volume = df['volume'].cumsum()
    )

    df['vwap'] = df['cumulative_tp_x_volume'] / df['cumulative_volume']

    # Drop the intermediate columns
    return df.drop(columns=['typical_price', 'tp_x_volume', 'cumulative_tp_x_volume', 'cumulative_volume'])


def calculate_moving_averages(df: pd.DataFrame, field: str = 'Close', periods: list = [9, 20, 50, 200]) -> pd.DataFrame:
    """
    Calculates a set of Simple and Exponential Moving Averages based on minutes.
    """
    df_out = df.copy()
    for period in periods:
        df_out[f'sma_{field.lower()}_{period}m'] = calculate_sma(df_out, period, field)
        df_out[f'ema_{field.lower()}_{period}m'] = calculate_ema(df_out, period, field)
    return df_out

# --- NEW: ATR Calculation Function ---
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) on a DataFrame with daily OHLC data.
    """
    df_atr = df.copy()
    df_atr['h-l'] = df_atr['high'] - df_atr['low']
    df_atr['h-pc'] = abs(df_atr['high'] - df_atr['close'].shift(1))
    df_atr['l-pc'] = abs(df_atr['low'] - df_atr['close'].shift(1))
    df_atr['tr'] = df_atr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df_atr['tr'].rolling(window=period).mean()


def calculate_200_day_ema(df: pd.DataFrame, field: str = 'Close') -> pd.DataFrame:
    """
    A wrapper to calculate the 200-day EMA on minute data.
    """
    df_out = df.copy()
    df_out[f'ema_{field.lower()}_200d'] = calculate_day_moving_average(df_out, days=200, field=field, ma_type='ema')
    return df_out

# --- NEW: Convenience function for the 200-day SMA on minute data ---
def calculate_200_day_sma(df: pd.DataFrame, field: str = 'Close') -> pd.DataFrame:
    """
    A wrapper to calculate the 200-day SMA on minute data.
    """
    df_out = df.copy()
    df_out[f'sma_{field.lower()}_200d'] = calculate_day_moving_average(df_out, days=200, field=field, ma_type='sma')
    return df_out