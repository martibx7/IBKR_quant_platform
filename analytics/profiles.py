# analytics/profiles.py

import pandas as pd
import numpy as np
import string
from collections import defaultdict

def get_session(dt):
    """
    Categorizes a datetime into a trading session for a single calendar day,
    based on the US Equity Market times in the provided timezone.

    - Pre-Market: Beginning of the day to 9:30 AM.
    - Regular: 9:30 AM to 4:00 PM.
    - After-Hours: 4:00 PM to the end of the day.
    """
    # --- MODIFIED: Simplified and corrected session logic ---
    market_open = pd.to_datetime('09:30:00').time()
    market_close = pd.to_datetime('16:00:00').time()

    current_time = dt.time()

    if current_time < market_open:
        return 'Pre-Market'
    elif market_open <= current_time < market_close:
        return 'Regular'
    else: # current_time >= market_close
        return 'After-Hours'

class VolumeProfiler:
    """Calculates a Volume Profile from a DataFrame of market bars."""
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.01):
        if bars_df.empty:
            self.profile = pd.Series(dtype=np.float64)
            self.poc_price = None
            self.vah = None
            self.val = None
            return

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self.profile = self._calculate_profile()
        self.poc_price, self.vah, self.val = self._calculate_poc_and_value_area()

    def _calculate_profile(self) -> pd.Series:
        """Aggregates volume at each price tick."""
        if self.bars_df.empty:
            return pd.Series(dtype=np.float64)

        all_ticks = []
        for _, row in self.bars_df.iterrows():
            num_ticks = int((row['High'] - row['Low']) / self.tick_size) + 1
            ticks = np.linspace(row['Low'], row['High'], num_ticks)
            volume_per_tick = row['Volume'] / num_ticks if num_ticks > 0 else 0
            for tick in ticks:
                all_ticks.append((round(tick / self.tick_size) * self.tick_size, volume_per_tick))

        profile = pd.DataFrame(all_ticks, columns=['price', 'volume']).groupby('price')['volume'].sum().sort_index()
        return profile

    def _calculate_poc_and_value_area(self) -> tuple:
        """Calculates the Point of Control (POC) and Value Area (VAH, VAL)."""
        if self.profile.empty:
            return None, None, None

        total_volume = self.profile.sum()
        poc_price = self.profile.idxmax()

        value_area_volume = total_volume * 0.70

        current_volume = self.profile.get(poc_price, 0)
        value_area_prices = [poc_price]

        prices_above = self.profile[self.profile.index > poc_price].index
        prices_below = self.profile[self.profile.index < poc_price].sort_index(ascending=False).index

        idx_above, idx_below = 0, 0

        while current_volume < value_area_volume:
            vol_above = self.profile.get(prices_above[idx_above]) if idx_above < len(prices_above) else -1
            vol_below = self.profile.get(prices_below[idx_below]) if idx_below < len(prices_below) else -1

            if vol_above == -1 and vol_below == -1: break

            if vol_above > vol_below:
                current_volume += vol_above
                value_area_prices.append(prices_above[idx_above])
                idx_above += 1
            else:
                current_volume += vol_below
                value_area_prices.append(prices_below[idx_below])
                idx_below += 1

        vah = max(value_area_prices)
        val = min(value_area_prices)
        return poc_price, vah, val

class MarketProfiler:
    """Calculates a comprehensive Market Profile (TPO) from a DataFrame of market bars."""
    def __init__(self, bars_df: pd.DataFrame, tick_size: float = 0.05):
        if not isinstance(bars_df.index, pd.DatetimeIndex) or bars_df.empty:
            self._initialize_empty()
            return

        self.bars_df = bars_df.copy()
        self.tick_size = tick_size
        self.tpo_periods = list(string.ascii_uppercase + string.ascii_lowercase)

        self._initialize_empty()

        # This is the corrected TPO calculation method
        self._calculate_tpo_profile_by_interval()

        self._calculate_poc_and_value_area()
        self._calculate_initial_balance()
        self._calculate_tails()

    def _initialize_empty(self):
        """Helper to set all attributes to default empty state."""
        self.tpo_profile = defaultdict(list)
        self.poc_price = None
        self.vah = None
        self.val = None
        self.ib_high = None
        self.ib_low = None
        self.selling_tail = None
        self.buying_tail = None

    def _calculate_tpo_profile_by_interval(self):
        """
        MODIFIED: Calculates the TPO profile by iterating through 30-minute intervals
        starting from the beginning of the session's actual data.
        """
        if self.bars_df.empty:
            return

        # --- MODIFIED: Start TPO periods from the first bar of the session ---
        session_start_time = self.bars_df.index[0].floor('30min')

        for i, tpo_letter in enumerate(self.tpo_periods):
            period_start = session_start_time + pd.Timedelta(minutes=30 * i)
            period_end = period_start + pd.Timedelta(minutes=30)

            period_bars = self.bars_df[
                (self.bars_df.index >= period_start) & (self.bars_df.index < period_end)
                ]

            if period_bars.empty:
                if period_start > self.bars_df.index[-1]:
                    break
                else:
                    continue

            for _, row in period_bars.iterrows():
                start_tick = int(row['Low'] / self.tick_size)
                end_tick = int(row['High'] / self.tick_size)

                for tick in range(start_tick, end_tick + 1):
                    price_level = round(tick * self.tick_size, 2)
                    if tpo_letter not in self.tpo_profile[price_level]:
                        self.tpo_profile[price_level].append(tpo_letter)

    # ... (the rest of the MarketProfiler class remains unchanged) ...
    def _calculate_poc_and_value_area(self):
        """Calculates TPO Point of Control (POC) and Value Area (VAH, VAL)."""
        if not self.tpo_profile: return

        tpo_counts = pd.Series({price: len(tpos) for price, tpos in self.tpo_profile.items()})
        self.poc_price = tpo_counts.idxmax()

        total_tpos = tpo_counts.sum()
        value_area_tpos = total_tpos * 0.7

        current_tpos = tpo_counts.get(self.poc_price, 0)
        value_area_prices = [self.poc_price]

        prices_above = tpo_counts[tpo_counts.index > self.poc_price].index
        prices_below = tpo_counts[tpo_counts.index < self.poc_price].sort_index(ascending=False).index

        idx_above, idx_below = 0, 0
        while current_tpos < value_area_tpos:
            vol_above = tpo_counts.get(prices_above[idx_above], 0) if idx_above < len(prices_above) else -1
            vol_below = tpo_counts.get(prices_below[idx_below], 0) if idx_below < len(prices_below) else -1

            if vol_above == -1 and vol_below == -1: break

            if vol_above > vol_below:
                current_tpos += vol_above
                value_area_prices.append(prices_above[idx_above])
                idx_above += 1
            else:
                current_tpos += vol_below
                value_area_prices.append(prices_below[idx_below])
                idx_below += 1

        self.vah = max(value_area_prices)
        self.val = min(value_area_prices)

    def _calculate_initial_balance(self):
        """Calculates the high and low of the Initial Balance (first hour of regular session)."""
        if not self.tpo_profile: return

        ib_prices = [price for price, tpos in self.tpo_profile.items() if 'A' in tpos or 'B' in tpos]
        if ib_prices:
            self.ib_high = max(ib_prices)
            self.ib_low = min(ib_prices)

    def _calculate_tails(self):
        """Identifies buying and selling tails (areas of excess)."""
        if not self.tpo_profile: return

        tpo_counts = pd.Series({price: len(tpos) for price, tpos in self.tpo_profile.items()})
        tpo_counts.sort_index(ascending=False, inplace=True)

        selling_tail_prices = []
        for price, count in tpo_counts.items():
            if count == 1:
                selling_tail_prices.append(price)
            else:
                break
        if len(selling_tail_prices) >= 2:
            self.selling_tail = (min(selling_tail_prices), max(selling_tail_prices))

        buying_tail_prices = []
        for price, count in tpo_counts.sort_index(ascending=True).items():
            if count == 1:
                buying_tail_prices.append(price)
            else:
                break
        if len(buying_tail_prices) >= 2:
            self.buying_tail = (min(buying_tail_prices), max(buying_tail_prices))

    def get_profile_str(self) -> str:
        """Generates a formatted string representation of the TPO Profile."""
        if not self.tpo_profile:
            return "No TPO Profile data available."

        prices = sorted(self.tpo_profile.keys(), reverse=True)
        output = []

        for price in prices:
            tpos = "".join(sorted(self.tpo_profile[price]))

            poc_marker = " POC" if price == self.poc_price else ""
            vah_marker = " VAH" if price == self.vah else ""
            val_marker = " VAL" if price == self.val else ""
            ib_high_marker = " IB High" if price == self.ib_high else ""
            ib_low_marker = " IB Low" if price == self.ib_low else ""

            line = f"{price:8.2f} | {tpos:<52} |{poc_marker}{vah_marker}{val_marker}{ib_high_marker}{ib_low_marker}"
            output.append(line)

        return "\n".join(output)