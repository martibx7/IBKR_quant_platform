import numpy as np
import pandas as pd

class VWAPCalculator:
    """
    Calculates intraday VWAP and its standard deviation bands on a running, bar-by-bar basis.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the calculator for a new session."""
        self.count = 0
        self.cumulative_tp_vol = 0  # Sum of (Typical Price * Volume)
        self.cumulative_vol = 0     # Sum of Volume
        self.cumulative_tp_sq_vol = 0 # Sum of (Typical Price^2 * Volume) for variance

    def update(self, bar: pd.Series):
        """Update the calculator with a new bar."""
        typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
        volume = bar['volume']

        if volume <= 0:
            return

        self.count += 1
        self.cumulative_tp_vol += typical_price * volume
        self.cumulative_vol += volume
        self.cumulative_tp_sq_vol += (typical_price ** 2) * volume

    def get_vwap_bands(self, sigmas: list[float] = [1.0, 2.0, 3.0]) -> dict:
        """
        Calculate and return the current VWAP and its standard deviation bands.
        """
        if self.cumulative_vol == 0 or self.count < 2:
            return {'vwap': np.nan, 'bands': {}}

        # Calculate VWAP
        vwap = self.cumulative_tp_vol / self.cumulative_vol

        # Calculate the running variance and standard deviation of VWAP
        variance = (self.cumulative_tp_sq_vol / self.cumulative_vol) - (vwap ** 2)
        std_dev = np.sqrt(variance) if variance > 0 else 0

        bands = {}
        for s in sigmas:
            bands[f'upper_{s}s'] = vwap + (s * std_dev)
            bands[f'lower_{s}s'] = vwap - (s * std_dev)

        return {'vwap': vwap, 'bands': bands}

    def get_cumulative_volume(self) -> float:
        """Returns the current running cumulative volume."""
        return self.cumulative_volume