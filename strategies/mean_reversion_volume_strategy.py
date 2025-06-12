# strategies/mean_reversion_volume_strategy.py

import logging
import pandas as pd
from datetime import datetime, time
import pytz

# Configure logging for the strategy
strategy_logger = logging.getLogger(__name__)
# Ensure basicConfig is only called once, usually in main.py or at a top level
# For strategy-specific logging, consider adding handlers or propagating to root logger
# if logging.root.handlers is empty:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseStrategy:
    """
    A base class for all trading strategies, providing common functionalities
    and abstract methods to be implemented by concrete strategies.
    """
    def __init__(self, symbols: list, ledger, **kwargs):
        self.symbols = symbols
        self.ledger = ledger
        self.current_prices = {s: 0.0 for s in symbols} # Stores the latest close price for each symbol
        self.open_trades = {} # Track trades for a specific strategy instance
        self.strategy_params = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        # Ensure that logger does not duplicate messages if parent logger already handles it
        self.logger.propagate = True

    def scan_for_candidates(self, all_daily_data: dict, previous_day_date: datetime.date) -> list:
        """
        Abstract method to scan for potential trading candidates.
        Implement in child classes.
        """
        raise NotImplementedError("scan_for_candidates method must be implemented by subclasses.")

    def on_session_start(self, session_data: dict):
        """
        Hook called at the beginning of a trading session.
        Can be used to reset daily state or perform pre-session analysis.
        """
        self.open_trades = {} # Reset open trades for the new session
        self.logger.info(f"Session starting for {list(session_data.keys())}.")
        pass

    def on_bar(self, symbol: str, bar: pd.Series):
        """
        Abstract method to process each incoming bar (tick) for a symbol.
        Implement in child classes.
        """
        raise NotImplementedError("on_bar method must be implemented by subclasses.")

    def on_session_end(self):
        """
        Hook called at the end of a trading session.
        Can be used to clean up or finalize daily operations.
        """
        self.logger.info(f"Session ended. Open positions: {self.ledger.open_positions}")
        pass

class MeanReversionVolumeStrategy(BaseStrategy):
    """
    Implements the Mean Reversion with Volume Confirmation Strategy.
    """
    def __init__(self, symbols: list, ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        self.sma_period = self.strategy_params.get('sma_period', 20)
        self.atr_period = self.strategy_params.get('atr_period', 14)
        self.deviation_multiplier = self.strategy_params.get('deviation_multiplier', 1.5)
        self.volume_sma_period = self.strategy_params.get('volume_sma_period', 50)
        self.risk_per_trade_pct = self.strategy_params.get('risk_per_trade_pct', 0.01)
        self.max_allocation_pct = self.strategy_params.get('max_allocation_pct', 0.20)
        self.stop_loss_pct = self.strategy_params.get('stop_loss_pct', 0.015)
        self.entry_time_str = self.strategy_params.get('entry_time', '09:35:00')
        self.exit_time_str = self.strategy_params.get('exit_time', '15:50:00')
        self.timezone_str = self.strategy_params.get('timezone', 'America/New_York')

        self.trading_timezone = pytz.timezone(self.timezone_str)
        self.entry_time = datetime.strptime(self.entry_time_str, '%H:%M:%S').time()
        self.exit_time = datetime.strptime(self.exit_time_str, '%H:%M:%S').time()

        self.logger.info(f"MeanReversionVolumeStrategy initialized with parameters: {self.strategy_params}")
        self.positions_today = {} # To track positions opened by THIS strategy instance for the current day

    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculates Simple Moving Average."""
        return data.rolling(window=period).mean()

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculates Average True Range."""
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # ATR is the SMA of True Range
        return true_range.rolling(window=period).mean()

    def scan_for_candidates(self, all_daily_data: dict, previous_day_date: datetime.date) -> list:
        """
        Scans for potential trading candidates based on daily data.
        For this strategy, we'll use daily data to pre-screen symbols
        that showed a significant dip on the previous day.
        """
        candidates = []
        for symbol, df_full in all_daily_data.items():
            # Ensure we're using daily data for scanning
            # If df_full is minute data, convert it to daily
            if df_full.index.normalize().nunique() > 1: # Check if it's likely minute data
                df_daily = df_full.resample('D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                df_daily = df_full # Already daily data

            if previous_day_date not in df_daily.index.date:
                continue

            prev_day_data = df_daily.loc[str(previous_day_date.date())]

            # Ensure enough historical data for SMA and ATR
            if len(df_daily) < max(self.sma_period, self.atr_period, self.volume_sma_period) + 1:
                self.logger.debug(f"Not enough historical data for {symbol} for scanning.")
                continue

            # Calculate indicators using all available daily data up to previous_day_date
            historical_data_for_indicators = df_daily.loc[df_daily.index.date <= previous_day_date].copy()
            historical_data_for_indicators['SMA'] = self._calculate_sma(historical_data_for_indicators['Close'], self.sma_period)
            historical_data_for_indicators['ATR'] = self._calculate_atr(
                historical_data_for_indicators['High'],
                historical_data_for_indicators['Low'],
                historical_data_for_indicators['Close'],
                self.atr_period
            )
            historical_data_for_indicators['Volume_SMA'] = self._calculate_sma(historical_data_for_indicators['Volume'], self.volume_sma_period)

            # Get the indicator values for the previous day
            prev_day_sma = historical_data_for_indicators['SMA'].iloc[-1]
            prev_day_atr = historical_data_for_indicators['ATR'].iloc[-1]
            prev_day_volume_sma = historical_data_for_indicators['Volume_SMA'].iloc[-1]


            # Conditions for previous day's significant dip (long entry criteria, adapted for daily scan)
            # This is a pre-screen, not the final entry logic.
            # We look for a previous day close significantly below its SMA.
            if (prev_day_data['Close'] < (prev_day_sma - (self.deviation_multiplier * prev_day_atr))) and \
                    (prev_day_data['Close'] > prev_day_data['Open']): # Previous day was an up candle (potential reversal)
                candidates.append(symbol)
                self.logger.debug(f"Candidate {symbol} found for {previous_day_date} (Close: {prev_day_data['Close']:.2f}, SMA: {prev_day_sma:.2f}, ATR: {prev_day_atr:.2f})")
        return candidates

    def on_session_start(self, session_data: dict):
        """
        Called at the beginning of a trading session.
        Pre-calculates indicators for candidate symbols using minute data.
        """
        super().on_session_start(session_data)
        self.daily_indicators = {}
        for symbol, df in session_data.items():
            if len(df) < max(self.sma_period, self.atr_period, self.volume_sma_period):
                self.logger.warning(f"Not enough historical data for {symbol} for in-session indicator calculation. Skipping for today.")
                continue

            # Ensure datetime index is localized for comparison with entry/exit times
            df_localized = df.tz_localize(self.trading_timezone, errors='coerce') if df.index.tz is None else df.tz_convert(self.trading_timezone)

            # Calculate SMA, ATR, Volume SMA for the entire session data (minute bars)
            df_localized['SMA'] = self._calculate_sma(df_localized['Close'], self.sma_period)
            df_localized['ATR'] = self._calculate_atr(df_localized['High'], df_localized['Low'], df_localized['Close'], self.atr_period)
            df_localized['Volume_SMA'] = self._calculate_sma(df_localized['Volume'], self.volume_sma_period)
            self.daily_indicators[symbol] = df_localized
            self.positions_today[symbol] = None # No position opened for this symbol yet today by this strategy

    def on_bar(self, symbol: str, bar: pd.Series):
        """
        Processes each incoming bar for a symbol to evaluate entry and exit conditions.
        """
        timestamp = bar.name # Bar index is the timestamp
        current_time = timestamp.time()
        current_price = bar['Close']
        current_volume = bar['Volume']

        # Ensure current_time is timezone-aware for comparison
        current_time_localized = self.trading_timezone.localize(datetime.combine(timestamp.date(), current_time)).time()

        if symbol not in self.daily_indicators:
            return # Skip if no pre-calculated indicators for this symbol

        df_today = self.daily_indicators[symbol]

        # Ensure the current bar's timestamp is in the pre-calculated dataframe index
        if timestamp not in df_today.index:
            self.logger.warning(f"Timestamp {timestamp} not found in {symbol} daily indicators. Skipping bar.")
            return

        # Get indicator values for the current bar (using .loc for exact timestamp)
        current_bar_data = df_today.loc[timestamp]
        current_sma = current_bar_data['SMA']
        current_atr = current_bar_data['ATR']
        current_volume_sma = current_bar_data['Volume_SMA']

        # --- Exit Logic (Check for open positions first) ---
        if symbol in self.ledger.open_positions and self.positions_today[symbol] == 'OPEN':
            position = self.ledger.open_positions[symbol]
            entry_price = position['entry_price']

            # Take Profit
            if current_price >= current_sma:
                self.ledger.record_trade(timestamp, symbol, position['quantity'], current_price, 'SELL', self.current_prices)
                self.logger.info(f"Closed LONG {symbol} for TP at {current_price:.2f} (Entry: {entry_price:.2f}).")
                self.positions_today[symbol] = 'CLOSED'
                return

            # Stop Loss
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                self.ledger.record_trade(timestamp, symbol, position['quantity'], current_price, 'SELL', self.current_prices)
                self.logger.info(f"Closed LONG {symbol} for SL at {current_price:.2f} (Entry: {entry_price:.2f}).")
                self.positions_today[symbol] = 'CLOSED'
                return

            # Time-Based Exit
            if current_time_localized >= self.exit_time:
                self.ledger.record_trade(timestamp, symbol, position['quantity'], current_price, 'SELL', self.current_prices)
                self.logger.info(f"Closed LONG {symbol} for EOD at {current_price:.2f} (Entry: {entry_price:.2f}).")
                self.positions_today[symbol] = 'CLOSED'
                return

        # --- Entry Logic (Only if no position currently open for this symbol and within entry time window) ---
        # Also ensure we haven't already opened and closed a position for this symbol today
        if symbol not in self.ledger.open_positions and self.positions_today[symbol] is None:
            if current_time_localized >= self.entry_time and current_time_localized < self.exit_time:
                # Condition 1: Price Deviation
                price_deviation_condition = (current_price < (current_sma - (self.deviation_multiplier * current_atr)))

                # Condition 2: Reversal Candle (Close > Open)
                reversal_candle_condition = (bar['Close'] > bar['Open'])

                # Condition 3: Volume Confirmation
                volume_confirmation_condition = (current_volume > current_volume_sma)

                if pd.isna(current_sma) or pd.isna(current_atr) or pd.isna(current_volume_sma):
                    self.logger.debug(f"Skipping {symbol} at {timestamp} due to NaN indicator values.")
                    return

                if price_deviation_condition and reversal_candle_condition and volume_confirmation_condition:
                    # Calculate quantity based on risk and allocation
                    account_equity = self.ledger.get_total_equity(self.current_prices)
                    risk_amount = account_equity * self.risk_per_trade_pct

                    # Stop loss distance in percentage of entry price
                    potential_stop_loss_distance_pct = self.stop_loss_pct

                    # Calculate max shares based on risk per share
                    # We assume entry at next open, but we use current close for risk calculation
                    # A more robust approach would use a defined stop level, not just % of entry
                    # For simplicity, we'll use a fixed % of current price as the implied risk for position sizing
                    implied_risk_per_share = current_price * potential_stop_loss_distance_pct

                    if implied_risk_per_share <= 0:
                        self.logger.warning(f"Implied risk per share for {symbol} is non-positive ({implied_risk_per_share:.4f}). Skipping entry.")
                        return

                    # Calculate target quantity based on risk
                    target_quantity_by_risk = int(risk_amount / implied_risk_per_share)

                    # Calculate max quantity based on max allocation
                    max_allocation_amount = account_equity * self.max_allocation_pct
                    target_quantity_by_allocation = int(max_allocation_amount / current_price) if current_price > 0 else 0

                    # Use the minimum of the two quantities, and round down to nearest 100 or 1 for partial shares
                    quantity = min(target_quantity_by_risk, target_quantity_by_allocation)

                    # Ensure quantity is positive and handle tick size if needed, not implemented here
                    if quantity <= 0:
                        self.logger.debug(f"Calculated quantity for {symbol} is {quantity}. Skipping entry.")
                        return

                    # Execute trade (buy at next open price for simplicity, using current price as proxy for now)
                    # In a real backtest, this would be a market order at the next bar's open.
                    # For simplicity, using current_price for ledger record, assuming immediate execution.
                    if self.ledger.record_trade(timestamp, symbol, quantity, current_price, 'BUY', self.current_prices):
                        self.logger.info(f"Entered LONG {symbol} at {current_price:.2f} with {quantity} shares. Equity: {account_equity:.2f}")
                        self.positions_today[symbol] = 'OPEN'
