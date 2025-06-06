# core/ledger.py

import pandas as pd
import numpy as np
from collections import defaultdict
from core.fee_models import TieredIBFeeModel

class BacktestLedger:
    def __init__(self, initial_cash: float = 100000.0, fee_model: TieredIBFeeModel = None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_model = fee_model

        # To track open positions: { 'symbol': {'quantity': float, 'entry_price': float} }
        self.open_positions = defaultdict(dict)

        # To log all transactions
        self.history = []
        # To log the result of each completed round-trip trade
        self.closed_trades = []
        # To track portfolio value over time for the equity curve
        self.equity_curve = []

    def _update_equity_curve(self, timestamp, market_prices: dict):
        """Updates the equity curve with the current portfolio value."""
        # Calculate market value of all open positions
        positions_value = 0
        for symbol, position_info in self.open_positions.items():
            current_price = market_prices.get(symbol, position_info['entry_price'])
            positions_value += position_info['quantity'] * current_price

        total_value = self.cash + positions_value
        self.equity_curve.append({'timestamp': timestamp, 'value': total_value})

    def record_trade(self, timestamp, symbol: str, quantity: float, price: float, order_type: str, market_prices: dict):
        """Records a trade, updates portfolio, and logs history."""
        commission = self.fee_model.get_order_fee(timestamp, quantity, price) if self.fee_model else 0
        self.cash -= commission

        if order_type.upper() == 'BUY':
            self.cash -= quantity * price
            # Update or create position
            if symbol in self.open_positions:
                # Averaging down/up logic
                current_qty = self.open_positions[symbol]['quantity']
                current_price = self.open_positions[symbol]['entry_price']
                new_total_qty = current_qty + quantity
                self.open_positions[symbol]['entry_price'] = ((current_price * current_qty) + (price * quantity)) / new_total_qty
                self.open_positions[symbol]['quantity'] = new_total_qty
            else:
                self.open_positions[symbol] = {'quantity': quantity, 'entry_price': price, 'entry_time': timestamp}

        elif order_type.upper() == 'SELL':
            if symbol not in self.open_positions or self.open_positions[symbol]['quantity'] < quantity:
                print(f"Warning: Attempted to sell {quantity} of {symbol} but position is smaller. Ignoring.")
                return

            self.cash += quantity * price
            entry_price = self.open_positions[symbol]['entry_price']
            entry_time = self.open_positions[symbol]['entry_time']

            # Log the closed trade for performance analysis
            pnl = (price - entry_price) * quantity - commission
            self.closed_trades.append({
                'symbol': symbol, 'pnl': pnl, 'entry_price': entry_price, 'exit_price': price,
                'entry_time': entry_time, 'exit_time': timestamp, 'quantity': quantity
            })

            # Update position
            self.open_positions[symbol]['quantity'] -= quantity
            if self.open_positions[symbol]['quantity'] == 0:
                del self.open_positions[symbol]

        # Log every transaction
        self.history.append({'timestamp': timestamp, 'symbol': symbol, 'type': order_type, 'qty': quantity, 'price': price, 'commission': commission})

        # Update portfolio value after every transaction
        self._update_equity_curve(timestamp, market_prices)