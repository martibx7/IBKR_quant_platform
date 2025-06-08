import pandas as pd
from core.fee_models import BaseFeeModel

class BacktestLedger:
    """
    Manages all financial records for a backtest, including cash,
    positions, and trade history.
    """
    def __init__(self, initial_cash: float, fee_model: BaseFeeModel):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_model = fee_model
        self.open_positions = {}
        self.closed_trades = []
        self.history = [{'timestamp': None, 'equity': initial_cash}]

    def _update_equity(self, timestamp: pd.Timestamp, market_prices: dict):
        """
        Calculates and records the current portfolio equity.
        """
        open_positions_value = 0
        for symbol, position in self.open_positions.items():
            current_price = market_prices.get(symbol, position['entry_price'])
            open_positions_value += position['quantity'] * current_price
        current_equity = self.cash + open_positions_value
        self.history.append({'timestamp': timestamp, 'equity': current_equity})

    def record_trade(self, timestamp: pd.Timestamp, symbol: str, quantity: int, price: float, order_type: str, market_prices: dict) -> bool:
        """
        Records a trade, updating cash and positions.
        Returns True if the trade was successful, False otherwise.
        """
        if order_type.upper() not in ['BUY', 'SELL']:
            print(f"ERROR: Invalid order type '{order_type}'")
            return False

        fees = self.fee_model.calculate_fee(quantity, price)

        if order_type.upper() == 'BUY':
            cost = (quantity * price) + fees
            if self.cash < cost:
                print(f"WARNING: Insufficient cash to buy {quantity} of {symbol}. Skipping trade.")
                return False

            self.cash -= cost
            if symbol in self.open_positions:
                old_qty = self.open_positions[symbol]['quantity']
                old_cost = self.open_positions[symbol]['entry_price'] * old_qty
                new_qty = old_qty + quantity
                new_cost = old_cost + (quantity * price)
                self.open_positions[symbol]['entry_price'] = new_cost / new_qty
                self.open_positions[symbol]['quantity'] = new_qty
            else:
                self.open_positions[symbol] = { 'quantity': quantity, 'entry_price': price, 'entry_time': timestamp }

        elif order_type.upper() == 'SELL':
            if symbol not in self.open_positions or self.open_positions[symbol]['quantity'] < quantity:
                print(f"WARNING: Attempting to sell {quantity} of {symbol}, but not enough held. Skipping trade.")
                return False

            proceeds = (quantity * price) - fees
            self.cash += proceeds
            position = self.open_positions[symbol]
            self.closed_trades.append({
                'symbol': symbol, 'quantity': quantity, 'entry_price': position['entry_price'],
                'exit_price': price, 'entry_time': position['entry_time'], 'exit_time': timestamp, 'fees': fees
            })
            position['quantity'] -= quantity
            if position['quantity'] == 0:
                del self.open_positions[symbol]

        self._update_equity(timestamp, market_prices)
        return True

    def get_trade_log(self):
        """Returns the list of all closed trades."""
        return self.closed_trades