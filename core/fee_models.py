# core/fee_models.py

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime

class BaseFeeModel(ABC):
    """Abstract base class for all fee models."""
    @abstractmethod
    def calculate_fee(self, quantity: float, price: float) -> float:
        pass

class TieredIBFeeModel(BaseFeeModel):
    """
    A tiered fee model simulating the Interactive Brokers structure.
    The fee per share decreases as the monthly trading volume increases.
    """
    _TIER = (
        (300_000, 0.0035),
        (3_000_000, 0.0020),
        (20_000_000, 0.0015),
        (100_000_000, 0.0010),
        (float('inf'), 0.0005)
    )

    def __init__(self, **kwargs):
        # This implementation does not properly handle MTD volume across backtest runs.
        # It's a simplified model for demonstration.
        self._shares_mtd = defaultdict(int)
        self.min_trade_fee = kwargs.get('min_trade_fee', 0.35)
        self.fee_per_share = kwargs.get('fee_per_share', 0.0035) # Default to lowest tier
        self.max_fee_pct_of_trade = kwargs.get('max_fee_pct_of_trade', 0.01)


    def calculate_fee(self, quantity: float, price: float) -> float:
        """
        Calculates the fee for a given order based on the tiered structure.
        NOTE: This implementation does not use the timestamp and resets MTD volume for each run.
        """
        key = (datetime.now().year, datetime.now().month) # Simplified key for single run
        self._shares_mtd[key] += quantity

        # Determine the fee rate
        current_tier_rate = self._TIER[-1][1]
        for limit, rate in self._TIER:
            if self._shares_mtd[key] <= limit:
                current_tier_rate = rate
                break

        # Base fee
        fee_calculated = quantity * current_tier_rate

        # Apply min fee
        fee_with_min = max(self.min_trade_fee, fee_calculated)

        # Apply max fee (1% of notional)
        notional_value_cap = self.max_fee_pct_of_trade * quantity * price
        final_fee = min(fee_with_min, notional_value_cap)

        return final_fee

class ZeroFeeModel(BaseFeeModel):
    """A simple fee model with zero commission."""
    def __init__(self, **kwargs):
        pass # No parameters needed

    def calculate_fee(self, quantity: float, price: float) -> float:
        return 0.0

class FixedFeeModel(BaseFeeModel):
    """
    A simple fee model that charges a flat fee for every trade.
    """
    def __init__(self, **kwargs):
        self.trade_fee = kwargs.get('trade_fee', 1.00)

    def calculate_fee(self, quantity: float, price: float) -> float:
        return self.trade_fee