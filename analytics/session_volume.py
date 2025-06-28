import pandas as pd
from analytics.profiles import get_session

def session_volume(intraday_data: pd.DataFrame, session: str = "RTH") -> int:
    """
    Calculates the total summed volume for a specific trading session.

    This is a simple helper to avoid redefining session logic in strategies.
    It filters the data for the requested session (e.g., "RTH") and returns the
    total volume.

    Args:
        intraday_data (pd.DataFrame):
            DataFrame with a DatetimeIndex and a 'volume' column.
        session (str):
            The session to analyze. Can be "RTH", "ON", or "Globex".

    Returns:
        The total volume as an integer for the specified session.
    """
    # Use the existing helper to get the DataFrame for the session
    session_data = get_session(intraday_data, session)

    if session_data.empty:
        return 0

    # Return the sum of the volume column
    return int(session_data['volume'].sum())