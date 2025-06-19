# strategies/volume_accumulation_strategy.py

import pandas as pd
from datetime import time

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler

class VolumeAccumulationStrategy(BaseStrategy):
    """
    Final, bias-free version. Pre-calculates historical metrics at market open,
    then evaluates breakout conditions on a bar-by-bar basis.
    """
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        super().__init__(symbols, ledger, config, **params)
        self.min_price               = self.params.get('min_price', 5.00)
        self.max_price               = self.params.get('max_price', 100.00)
        self.consolidation_days      = self.params.get('consolidation_days', 10)
        self.volume_lookback_days    = self.params.get('volume_lookback_days', 20)
        self.consolidation_range_pct = self.params.get('consolidation_range_pct', 0.07)
        self.breakout_volume_ratio   = self.params.get('breakout_volume_ratio', 1.5)
        self.risk_per_trade_pct      = self.params.get('risk_per_trade_pct', 0.01)
        self.profit_target_r         = self.params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r     = self.params.get('breakeven_trigger_r', 0.75)
        self.tick_size               = self.config['backtest'].get('tick_size_volume_profile', 0.01)

        # --- Strategy State ---
        self.candidates = {}
        self.intraday_data = {}

    def get_required_lookback(self) -> int:
        # Ask for enough data to satisfy consolidation or volume lookback requirements
        return max(self.consolidation_days, self.volume_lookback_days)

    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        """
        Called once at the start of each day with historical data ONLY.
        Pre-calculates consolidation metrics and identifies potential stocks to watch.
        """
        self.logger.info(f"--- Pre-calculating consolidation metrics for {len(historical_data)} symbols ---")
        self.candidates.clear()
        self.intraday_data.clear()

        for symbol, df in historical_data.items():
            if df.empty or len(set(df.index.date)) < self.consolidation_days:
                continue

            # Pre-calculate all consolidation metrics using only historical data
            consol_high = df['high'].max()
            consol_low = df['low'].min()
            if consol_low == 0: continue

            consol_range_pct = (consol_high - consol_low) / consol_low
            if consol_range_pct > self.consolidation_range_pct: continue

            avg_daily_volume = df.groupby(df.index.date)['volume'].sum().mean()

            profiler = VolumeProfiler(self.tick_size)
            profile = profiler.calculate(df)

            if profile:
                self.candidates[symbol] = {
                    'status': 'monitoring', # Initial state: watching for a breakout
                    'consol_high': consol_high,
                    'stop_loss': consol_low,
                    'avg_daily_volume': avg_daily_volume,
                    'poc': profile['poc_price'],
                }
        self.logger.info(f"Found {len(self.candidates)} symbols with valid consolidations to monitor for breakouts.")

    def on_bar(self, symbol: str, bar: pd.Series):
        """
        Processes each intraday bar, checking for signals and managing trades.
        """
        super().on_bar(symbol, bar)

        # Accumulate intraday data for real-time calculations
        if symbol not in self.intraday_data:
            self.intraday_data[symbol] = []
        self.intraday_data[symbol].append(bar)

        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
        elif symbol in self.candidates:
            self._scan_and_handle_entry(symbol, bar)

    def _scan_and_handle_entry(self, symbol: str, bar: pd.Series):
        """
        Checks for breakout signals and manages the entry process on each bar.
        """
        candidate = self.candidates[symbol]

        # State 1: Monitoring for a breakout signal
        if candidate['status'] == 'monitoring':
            # Perform real-time relative volume check
            trading_day_minutes = 390
            minutes_since_open = max(1, ((bar.name.hour - 9) * 60 + (bar.name.minute - 30)))
            fraction_of_day = minutes_since_open / trading_day_minutes
            expected_volume = candidate['avg_daily_volume'] * fraction_of_day
            current_volume = pd.DataFrame(self.intraday_data[symbol])['volume'].sum()

            is_high_volume = current_volume > (expected_volume * self.breakout_volume_ratio)
            is_breakout = bar['close'] > candidate['consol_high']
            price_ok = self.min_price <= bar['close'] <= self.max_price

            if price_ok and is_high_volume and is_breakout:
                self.logger.info(f"  [BREAKOUT DETECTED] {symbol} @ {bar['close']:.2f} on high relative volume.")
                candidate['status'] = 'watching_for_retest' # Transition to next state

        # State 2: Breakout confirmed, watching for a re-test of the POC
        elif candidate['status'] == 'watching_for_retest' and bar['low'] <= candidate['poc']:
            candidate['status'] = 'triggered' # Transition to final entry state
            candidate['confirmation_high'] = bar['high']
            self.logger.info(f"  [SETUP] {symbol} re-tested POC {candidate['poc']:.2f}. Waiting for confirmation above {bar['high']:.2f}.")

        # State 3: POC re-test confirmed, waiting for entry confirmation
        elif candidate['status'] == 'triggered' and bar['high'] > candidate['confirmation_high']:
            entry_price = candidate['confirmation_high']
            stop_loss_price = candidate['stop_loss']
            risk_per_share = entry_price - stop_loss_price

            if risk_per_share <= 0:
                del self.candidates[symbol]
                return

            equity = self.ledger.get_total_equity(self.current_prices)
            dollar_risk = equity * self.risk_per_trade_pct
            quantity = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

            if quantity > 0:
                self.logger.info(f"  [ENTRY] {symbol}: Confirmed above {entry_price:.2f}.")
                if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                    profit_target = entry_price + (risk_per_share * self.profit_target_r)
                    self.active_trades[symbol] = {
                        'entry_price': entry_price, 'stop_loss': stop_loss_price,
                        'profit_target': profit_target, 'quantity': quantity,
                        'risk_per_share': risk_per_share
                    }
            if symbol in self.candidates:
                del self.candidates[symbol]

    def _manage_active_trade(self, symbol: str, bar: pd.Series):
        trade = self.active_trades.get(symbol)
        if not trade: return

        if not trade.get('is_breakeven', False):
            breakeven_trigger_price = trade['entry_price'] + (trade['risk_per_share'] * self.breakeven_trigger_r)
            if bar['high'] >= breakeven_trigger_price:
                trade['stop_loss'] = trade['entry_price']
                trade['is_breakeven'] = True
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")

        exit_price, reason = None, None
        if bar['low'] <= trade['stop_loss']:
            exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['high'] >= trade.get('profit_target', float('inf')):
            exit_price, reason = trade['profit_target'], "Profit Target"

        # --- FIX: Convert bar timestamp to New York time before checking the time ---
        market_time = bar.name.tz_convert('America/New_York').time()
        if market_time >= time(15, 50):
            exit_price, reason = bar['close'], "End of Day"

        if exit_price and reason:
            self.logger.info(f"  [EXIT] {symbol} @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices, exit_reason=reason)
            if symbol in self.active_trades:
                del self.active_trades[symbol]

    def on_session_end(self):
        """Clear any intraday data at the end of the session."""
        self.intraday_data.clear()