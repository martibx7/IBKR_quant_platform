import pandas as pd
from datetime import timedelta
from collections import deque
import logging
from .fee_models import BaseFeeModel

engine_logger = logging.getLogger(__name__)

class BacktestLedger:
    """
    Manages all financial records for a backtest.
    Includes T+1 settlement, slippage, and detailed trade logging.
    """
    def __init__(self, initial_cash: float, fee_model: BaseFeeModel, slippage_model: str, slippage_pct: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.buying_power = initial_cash
        self.open_positions = {}
        self.closed_trades = []  # This will be the official log of completed trades
        self.equity_curve = [{'timestamp': None, 'equity': initial_cash}]
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.slippage_pct = slippage_pct
        self.pending_settlements = deque()

    def get_cash(self) -> float:
        """Returns the current total settled cash amount."""
        return self.cash

    def _apply_slippage(self, price: float, trade_type: str) -> float:
        """Applies slippage to the execution price based on the configured model."""
        if self.slippage_model == 'percent':
            if trade_type.upper() == 'BUY':
                return price * (1 + self.slippage_pct)
            elif trade_type.upper() == 'SELL':
                return price * (1 - self.slippage_pct)
        return price

    def _update_buying_power(self, cash_change: float):
        """Updates cash immediately. For a cash account, buying power equals settled cash."""
        self.cash += cash_change
        self.buying_power = self.cash

    def get_total_equity(self, market_prices: dict) -> float:
        """Calculates the total current value of the portfolio (cash + open positions)."""
        open_positions_value = sum(
            pos['quantity'] * market_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.open_positions.items()
        )
        return self.cash + open_positions_value

    def _update_equity(self, timestamp: pd.Timestamp, market_prices: dict):
        """Calculates and records the current portfolio equity."""
        current_equity = self.get_total_equity(market_prices)
        self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

    def settle_funds(self, current_date):
        """
        Checks the settlement queue and adds any settled funds to cash and buying power.
        This should be called once at the start of each new trading day.
        """
        while self.pending_settlements and self.pending_settlements[0][0] <= current_date:
            settlement_date, amount = self.pending_settlements.popleft()
            self._update_buying_power(amount)
            engine_logger.debug(f"[{current_date}] Settled ${amount:.2f}")

    def record_trade(self, timestamp: pd.Timestamp, symbol: str, quantity: int, price: float, order_type: str, market_prices: dict, exit_reason: str = None) -> bool:
        """Records a trade, applying slippage and fees, updating cash and positions."""
        execution_price = self._apply_slippage(price, order_type)
        fees = self.fee_model.calculate_fee(quantity, execution_price)

        if order_type.upper() == 'BUY':
            cost = (quantity * execution_price) + fees
            if self.buying_power < cost:
                engine_logger.warning(f"[{timestamp}] Not enough buying power for BUY {symbol}. Needed: {cost:.2f}, Have: {self.buying_power:.2f}")
                return False

            self._update_buying_power(-cost)
            self.open_positions[symbol] = {'quantity': quantity, 'entry_price': execution_price, 'entry_time': timestamp}

        elif order_type.upper() == 'SELL':
            if symbol not in self.open_positions:
                engine_logger.warning(f"[{timestamp}] Attempted to SELL {symbol} but position not found.")
                return False

            position = self.open_positions.pop(symbol) # Remove position
            proceeds = (quantity * execution_price) - fees
            settlement_date = timestamp.date() + timedelta(days=1) # T+1 Settlement
            self.pending_settlements.append((settlement_date, proceeds))

            pnl = (execution_price - position['entry_price']) * quantity - fees

            # Append the full, closed trade details to self.closed_trades
            self.closed_trades.append({
                'symbol': symbol, 'quantity': quantity,
                'entry_price': position['entry_price'], 'exit_price': execution_price,
                'entry_time': position['entry_time'], 'exit_time': timestamp,
                'fees': fees, 'pnl': pnl, 'exit_reason': exit_reason
            })

        self._update_equity(timestamp, market_prices)
        return True

    def get_equity_curve(self):
        """Returns the portfolio equity curve as a pandas DataFrame."""
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty and 'timestamp' in equity_df.columns:
            equity_df.dropna(subset=['timestamp'], inplace=True)
            equity_df.set_index('timestamp', inplace=True)
        return equity_df

    def get_trade_log(self):
        """Returns the list of all closed trades as a DataFrame."""
        return pd.DataFrame(self.closed_trades) # This now correctly returns the closed trades.