# core/ledger.py

import pandas as pd
from core.fee_models import TieredIBFeeModel

class BacktestLedger:
    """
    Manages the financial state of the backtest, including cash, positions,
    and trade history.
    """
    def __init__(self, initial_cash: float, fee_model: TieredIBFeeModel):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_model = fee_model
        self.open_positions = {}  # symbol -> {'quantity': float, 'entry_price': float}
        self.trade_history = []
        self.equity_curve = [{'timestamp': None, 'equity': initial_cash}]
        self.total_fees = 0.0

    def record_trade(self, timestamp, symbol: str, quantity: float, price: float, order_type: str, market_prices: dict):
        """Records a trade and updates the ledger state."""
        # --- FIX IS HERE: Use the correct method name 'get_order_fee' ---
        fee = self.fee_model.get_order_fee(timestamp, quantity, price)
        self.total_fees += fee

        trade_pnl = 0

        if order_type.upper() == 'BUY':
            self.cash -= (quantity * price) + fee
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {'quantity': 0, 'entry_price': 0}

            # Update position with weighted average entry price
            current_quantity = self.open_positions[symbol]['quantity']
            current_value = current_quantity * self.open_positions[symbol]['entry_price']
            new_value = quantity * price

            total_quantity = current_quantity + quantity
            if total_quantity > 0:
                self.open_positions[symbol]['entry_price'] = (current_value + new_value) / total_quantity
            self.open_positions[symbol]['quantity'] = total_quantity

        elif order_type.upper() == 'SELL':
            self.cash += (quantity * price) - fee
            if symbol in self.open_positions:
                entry_price = self.open_positions[symbol]['entry_price']
                trade_pnl = (price - entry_price) * quantity - fee
                self.open_positions[symbol]['quantity'] -= quantity
                if self.open_positions[symbol]['quantity'] <= 0:
                    del self.open_positions[symbol]

        self.trade_history.append({
            'timestamp': timestamp, 'symbol': symbol, 'quantity': quantity,
            'price': price, 'order_type': order_type, 'fee': fee, 'pnl': trade_pnl
        })

        equity_data = {'timestamp': timestamp, 'equity': self.get_total_equity(market_prices)}
        self.equity_curve.append(equity_data)


    def get_total_equity(self, market_prices: dict) -> float:
        """
        Calculates the total current equity of the account (cash + market value of open positions).

        Args:
            market_prices (dict): A dictionary mapping symbols to their current market price.

        Returns:
            float: The total equity.
        """
        positions_value = 0.0
        for symbol, position_details in self.open_positions.items():
            current_price = market_prices.get(symbol)
            if current_price is not None:
                positions_value += position_details['quantity'] * current_price
            else:
                # Fallback to entry price if current market price isn't available
                positions_value += position_details['quantity'] * position_details['entry_price']

        return self.cash + positions_value