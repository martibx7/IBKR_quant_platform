import pandas as pd
from datetime import time, datetime
import pytz

from strategies.base import BaseStrategy
from analytics.indicators import calculate_atr

class VolumeAccumulationStrategy(BaseStrategy):
    """
    Implements a classic "breakout and retest" strategy with a multi-step
    entry confirmation to avoid whipsaws.
    """
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        super().__init__(symbols, ledger, config, **params)
        self.params = params
        self.consolidation_days      = self.params.get('consolidation_days', 10)
        self.risk_per_trade_pct      = self.params.get('risk_per_trade_pct', 0.01)
        self.profit_target_r         = self.params.get('profit_target_r', 1.2)
        self.breakeven_trigger_r     = self.params.get('breakeven_trigger_r', 0.75)
        self.breakout_volume_ratio   = self.params.get('breakout_volume_ratio', 3.0)

        atr_params = self.params.get('atr_filter', {})
        self.atr_period = atr_params.get('period', 14)
        self.min_atr_pct = atr_params.get('min_atr_pct', 0.015)

        time_params = self.params.get('entry_time_window', {})
        self.entry_start_time = time.fromisoformat(time_params.get('start', '10:00:00'))
        self.entry_end_time = time.fromisoformat(time_params.get('end', '13:00:00'))

        self.ny_timezone = pytz.timezone('America/New_York')
        self.candidates = {}
        self.todays_traded = set()
        self.report_stats = {'candidates': 0, 'filtered_by_atr': 0, 'breakouts': 0, 'setups': 0, 'entries': 0}
        self.logger.info(f"Strategy '{self.__class__.__name__}' initialized with parameters: {self.params}")

    def get_required_lookback(self) -> int:
        return self.consolidation_days + self.atr_period + 5

    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        self.logger.info(f"--- Pre-calculating consolidation metrics ---")
        self.candidates.clear()
        self.todays_traded.clear()
        self.report_stats = {key: 0 for key in self.report_stats}

        for symbol, df in historical_data.items():
            if df.empty or len(set(df.index.date)) < (self.consolidation_days + self.atr_period): continue

            df.index = df.index.tz_convert(self.ny_timezone)
            rth_df = df.between_time('09:30', '16:00', inclusive='left')
            if rth_df.empty: continue

            daily_df = rth_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            if len(daily_df) < self.atr_period: continue

            daily_df['atr'] = calculate_atr(daily_df, period=self.atr_period)
            if daily_df.empty or pd.isna(daily_df['atr'].iloc[-1]): continue

            last_atr = daily_df['atr'].iloc[-1]
            last_close = daily_df['close'].iloc[-1]

            if last_close == 0 or (last_atr / last_close) < self.min_atr_pct:
                self.report_stats['filtered_by_atr'] += 1
                continue

            consolidation_df = daily_df.tail(self.consolidation_days)
            self.candidates[symbol] = {
                'consol_high': consolidation_df['high'].max(),
                'stop_loss': consolidation_df['low'].min(),
                'avg_daily_volume': consolidation_df['volume'].mean(),
                'status': 'monitoring',
            }
        self.report_stats['candidates'] = len(self.candidates)
        self.logger.info(f"Found {self.report_stats['candidates']} candidates after filtering. ({self.report_stats['filtered_by_atr']} filtered by ATR).")

    def on_bar(self, symbol: str, bar: pd.Series):
        if symbol in self.todays_traded: return
        super().on_bar(symbol, bar)

        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
        elif symbol in self.candidates:
            self._scan_for_entry(symbol, bar)

    def _scan_for_entry(self, symbol: str, bar: pd.Series):
        candidate = self.candidates[symbol]
        ny_time = bar.name.tz_convert(self.ny_timezone).time()

        # State 1: Monitor for a breakout above the historical consolidation high
        if candidate['status'] == 'monitoring':
            if bar['close'] > candidate['consol_high']:
                self.logger.info(f"  [BREAKOUT] {symbol} broke consolidation high {candidate['consol_high']:.2f}.")
                candidate['status'] = 'watching_for_retest'
                self.report_stats['breakouts'] += 1

        # State 2: After breakout, wait for a pullback to touch the old high
        elif candidate['status'] == 'watching_for_retest':
            if bar['low'] <= candidate['consol_high']:
                self.logger.info(f"  [RETEST] {symbol} re-tested consolidation high at {candidate['consol_high']:.2f}.")
                candidate['status'] = 'retest_confirmed'
                self.report_stats['setups'] += 1

        # State 3: After retest, wait for price to close back above the level to confirm entry
        elif candidate['status'] == 'retest_confirmed':
            # Only attempt entry within the configurable time window
            if not (self.entry_start_time <= ny_time < self.entry_end_time):
                return

            if bar['close'] > candidate['consol_high']:
                self.logger.info(f"  [CONFIRMATION] {symbol} confirmed support at {candidate['consol_high']:.2f}.")
                self._execute_trade(symbol, bar, candidate)
                return

    def _execute_trade(self, symbol: str, bar: pd.Series, candidate: dict):
        if symbol in self.active_trades: return

        entry_price = bar['close'] # Enter at the close of the confirmation bar
        stop_loss_price = candidate['stop_loss']
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0: return

        equity = self.ledger.get_total_equity(self.current_prices)
        quantity = int((equity * self.risk_per_trade_pct) / risk_per_share)
        quantity = min(quantity, int(bar['volume'] * 0.05))

        if quantity < 1:
            self.todays_traded.add(symbol)
            return

        self.todays_traded.add(symbol)
        self.report_stats['entries'] += 1
        self.logger.info(f"  [ENTRY ATTEMPT] {symbol}")

        if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
            self.active_trades[symbol] = {
                'entry_price': entry_price, 'stop_loss': stop_loss_price,
                'profit_target': entry_price + (risk_per_share * self.profit_target_r),
                'quantity': quantity, 'risk_per_share': risk_per_share
            }
            candidate['status'] = 'trade_taken'
        else:
            self.todays_traded.discard(symbol)
            self.report_stats['entries'] -= 1

    def _manage_active_trade(self, symbol: str, bar: pd.Series):
        trade = self.active_trades.get(symbol)
        if not trade: return
        if not trade.get('is_breakeven', False):
            if bar['high'] >= trade['entry_price'] + (trade['risk_per_share'] * self.breakeven_trigger_r):
                trade.update({'stop_loss': trade['entry_price'], 'is_breakeven': True})
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")
        exit_price, reason = None, None
        if bar['low'] <= trade['stop_loss']: exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['high'] >= trade.get('profit_target', float('inf')): exit_price, reason = trade['profit_target'], "Profit Target"
        ny_time = bar.name.tz_convert(self.ny_timezone).time()
        if ny_time >= time(15, 50): exit_price, reason = bar['close'], "End of Day"
        if exit_price and reason:
            self.logger.info(f"  [EXIT] {symbol} @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices, exit_reason=reason)
            if symbol in self.active_trades: del self.active_trades[symbol]

    def on_session_end(self):
        self.logger.info(f"Funnel stats: {self.report_stats}")