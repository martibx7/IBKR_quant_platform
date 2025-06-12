# analytics/moving_averages.py

import pandas as pd

# --- UPDATED: Added min_periods parameter to handle incomplete windows ---
def calculate_sma(df: pd.DataFrame, period: int, field: str = 'Close', min_periods: int = None) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given field.
    """
    if field not in df.columns or df[field].isnull().all():
        return pd.Series(dtype=float)

    # If min_periods is not set, it defaults to the window size
    if min_periods is None:
        min_periods = period

    return df[field].rolling(window=period, min_periods=min_periods).mean()

def calculate_ema(df: pd.DataFrame, period: int, field: str = 'Close') -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) for a given field.
    The `ewm` function with `adjust=False` inherently handles shorter periods at the start.
    """
    if field not in df.columns or df[field].isnull().all():
        return pd.Series(dtype=float)
    return df[field].ewm(span=period, adjust=False).mean()

# --- NEW: Function to calculate moving averages over a day-based period ---
def calculate_day_moving_average(df: pd.DataFrame, days: int, field: str = 'Close', ma_type: str = 'ema') -> pd.Series:
    """
    Calculates a moving average based on a specified number of days, using minute data.

    Args:
        df (pd.DataFrame): The input DataFrame with minute-frequency data.
        days (int): The number of days for the moving average period (e.g., 200).
        field (str): The column to use for calculation.
        ma_type (str): The type of moving average ('sma' or 'ema').

    Returns:
        pd.Series: The calculated moving average series.
    """
    # Standard US equity market has 6.5 hours of trading (9:30 AM to 4:00 PM ET)
    # 6.5 hours * 60 minutes/hour = 390 minutes per day
    minutes_per_day = 390
    period_in_minutes = days * minutes_per_day

    print(f"Calculating {days}-day {ma_type.upper()} using a {period_in_minutes}-minute window.")

    if ma_type.lower() == 'sma':
        # For the day-based SMA, we ensure a value is calculated even with partial data.
        # We set min_periods=1 so that it doesn't return NaN if we don't have the full 200 days of data.
        return calculate_sma(df, period=period_in_minutes, field=field, min_periods=1)
    elif ma_type.lower() == 'ema':
        # EMA's standard implementation already gives weight to recent data and works with
        # an expanding window at the start, so it naturally handles the "use what you have" requirement.
        return calculate_ema(df, period=period_in_minutes, field=field)
    else:
        raise ValueError("Invalid ma_type specified. Choose 'sma' or 'ema'.")