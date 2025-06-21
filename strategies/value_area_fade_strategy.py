import pandas as pd
from datetime import time, timedelta
import pytz
import os

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler

class ValueAreaFadeStrategy(BaseStrategy):
    """
    Implements a "Value-Area Extension Failure" strategy with additional
    time-based and quality-based filters to refine entries.
    """
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        # This super call is what initializes the logger from BaseStrategy
        super().__init__(symbols, ledger, config, **params)

        self.params = params
        self.timezone = pytz.timezone(self.params.get('timezone', 'America/New_York'))

        self.use_symbol_file = self.params.get('use_symbol_file', False)
        self.symbol_file = self.params.get('symbol_file', '')
        self.watchlist = set()

        self.risk_per_trade_pct = self.params.get('risk_per_trade_pct', 0.01)
        self.liquidity_cap_pct = self.params.get('liquidity_cap_pct', 0.05)
        self.value_area_pct = self.params.get('value_area_pct', 0.70)
        self.confirmation_window = timedelta(minutes=self.params.get('confirmation_window_minutes', 5))
        self.stop_buffer_ticks = self.params.get('stop_buffer_ticks', 2)
        self.primary_target = self.params.get('primary_target', 'poc')
        self.breakeven_trigger_r = self.params.get('breakeven_trigger_r', 0.75)

        time_params = self.params.get('entry_time_window', {})
        self.entry_start_time = time.fromisoformat(time_params.get('start', '10:00:00'))
        self.entry_end_time = time.fromisoformat(time_params.get('end', '15:00:00'))
        self.min_dip_ticks = self.params.get('min_dip_ticks', 3)
        self.min_reward_risk_ratio = self.params.get('min_reward_risk_ratio', 1.0)

        self.tick_size = config.get('backtest', {}).get('tick_size_market_profile', 0.01)

        self.candidates = {}
        self.active_trades = {}
        self.todays_traded = set()
        self.session_dataframes = {}

        self.logger.info(f"Strategy '{self.__class__.__name__}' initialized with parameters: {self.params}")

    def get_required_lookback(self) -> int:
        return 2

    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        self.logger.info("--- New Session ---")
        self.candidates.clear()
        self.active_trades.clear()
        self.todays_traded.clear()
        self.session_dataframes.clear()
        self.watchlist.clear()

        if self.use_symbol_file and self.symbol_file:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, self.symbol_file)
            try:
                with open(filepath, 'r') as f:
                    self.watchlist = {line.strip().upper() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(self.watchlist)} symbols from watchlist file: {filepath}")
            except FileNotFoundError:
                self.logger.error(f"Watchlist file not found at {filepath}. No symbols will be traded.")
        elif self.use_symbol_file:
            self.logger.warning("`use_symbol_file` is true but `symbol_file` is not specified in config.")

    def on_bar(self, symbol: str, bar: pd.Series):
        if self.use_symbol_file and symbol not in self.watchlist:
            return

        super().on_bar(symbol, bar)

        if symbol not in self.session_dataframes:
            self.session_dataframes[symbol] = pd.DataFrame([bar], index=[bar.name])
        else:
            self.session_dataframes[symbol].loc[bar.name] = bar

        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
            return

        if symbol not in self.candidates:
            self.candidates[symbol] = {'val_fail_state': 'watch'}

        ny_time = bar.name.tz_convert(self.timezone).time()
        if not (self.entry_start_time <= ny_time <= self.entry_end_time):
            return

        if len(self.session_dataframes[symbol]) < 30:
            return

        session_df = self.session_dataframes[symbol]
        profiler = VolumeProfiler(tick_size=self.tick_size, value_area_pct=self.value_area_pct)
        profile = profiler.calculate(session_df)

        if not profile: return

        val = profile['value_area_low']
        vah = profile['value_area_high']
        poc = profile['poc_price']

        self._scan_for_val_fail_long(symbol, bar, val, poc, vah)

    def _scan_for_val_fail_long(self, symbol: str, bar: pd.Series, val: float, poc: float, vah: float):
        trade_key = (symbol, bar.name.date(), 'val_fail_long')
        if trade_key in self.todays_traded:
            return

        state = self.candidates[symbol].get('val_fail_state', 'watch')

        if state == 'watch':
            dip_threshold = val - (self.tick_size * self.min_dip_ticks)
            if bar['close'] < dip_threshold:
                self.logger.info(f"  [ARMED LONG] {symbol} dipped below VAL {val:.2f} at {bar.name.time()}.")
                self.candidates[symbol]['val_fail_state'] = 'armed'
                self.candidates[symbol]['extension_bar_low'] = bar['low']
                self.candidates[symbol]['armed_time'] = bar.name

        elif state == 'armed':
            if bar.name > self.candidates[symbol]['armed_time'] + self.confirmation_window:
                self.candidates[symbol]['val_fail_state'] = 'watch'
                return

            if bar['close'] >= val:
                entry_price = bar['close']
                extension_low = self.candidates[symbol]['extension_bar_low']
                stop_loss_price = extension_low - (self.tick_size * self.stop_buffer_ticks)
                profit_target = poc if self.primary_target == 'poc' else vah

                potential_risk = entry_price - stop_loss_price
                if potential_risk <= 0:
                    self.candidates[symbol]['val_fail_state'] = 'watch'
                    return

                potential_reward = abs(profit_target - entry_price)
                reward_risk_ratio = potential_reward / potential_risk

                if reward_risk_ratio >= self.min_reward_risk_ratio:
                    self.logger.info(f"  [CONFIRMED LONG] {symbol} R:R {reward_risk_ratio:.2f} >= {self.min_reward_risk_ratio}. Executing trade.")
                    self._execute_fade_trade(symbol, bar, stop_loss_price, profit_target)
                    self.todays_traded.add(trade_key)
                    self.candidates[symbol]['val_fail_state'] = 'trade_taken'
                else:
                    self.logger.info(f"  [REJECTED] {symbol} trade rejected due to poor R:R of {reward_risk_ratio:.2f}.")
                    self.candidates[symbol]['val_fail_state'] = 'watch'

    def _execute_fade_trade(self, symbol: str, bar: pd.Series, stop_loss_price: float, profit_target: float):
        entry_price = bar['close']
        risk_per_share = entry_price - stop_loss_price

        quantity = self._calculate_quantity(entry_price, risk_per_share, bar['volume'])
        if quantity == 0: return

        if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
            self.active_trades[symbol] = {
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'profit_target': profit_target,
                'quantity': quantity,
                'risk_per_share': risk_per_share,
                'path': 'val_fail_long'
            }
            self.logger.info(f"  [ENTRY] {symbol} (VAL Fade) @ {entry_price:.2f} | Target: {profit_target:.2f} | SL: {stop_loss_price:.2f}")

    def _calculate_quantity(self, entry_price, risk_per_share, bar_volume):
        equity = self.ledger.get_total_equity(self.current_prices)
        dollar_risk = equity * self.risk_per_trade_pct
        if risk_per_share <= 0: return 0

        quantity_by_risk = int(dollar_risk / risk_per_share)
        quantity_by_liquidity = int(bar_volume * self.liquidity_cap_pct)
        if quantity_by_liquidity == 0 and bar_volume > 0:
            quantity_by_liquidity = 1

        return min(quantity_by_risk, quantity_by_liquidity)

    def _manage_active_trade(self, symbol: str, bar: pd.Series):
        trade = self.active_trades.get(symbol)
        if not trade: return

        if not trade.get('is_breakeven', False):
            breakeven_trigger = trade['entry_price'] + (trade['risk_per_share'] * self.breakeven_trigger_r)
            if bar['high'] >= breakeven_trigger:
                trade.update({'stop_loss': trade['entry_price'], 'is_breakeven': True})
                self.logger.info(f"  [MOVE TO BREAKEVEN] {symbol} stop moved to {trade['stop_loss']:.2f}")

        exit_price, reason = None, None
        if bar['low'] <= trade['stop_loss']:
            exit_price, reason = trade['stop_loss'], "Stop Loss"
        elif bar['high'] >= trade.get('profit_target', float('inf')):
            exit_price, reason = trade['profit_target'], "Profit Target"

        ny_time = bar.name.tz_convert(self.timezone).time()
        if not exit_price and ny_time >= time(15, 50):
            exit_price, reason = bar['close'], "End of Day"

        if exit_price and reason:
            self.logger.info(f"  [EXIT] {symbol} ({trade['path']}) @ {exit_price:.2f} ({reason})")
            self.ledger.record_trade(bar.name, symbol, trade['quantity'], exit_price, 'SELL', self.current_prices, exit_reason=reason)
            del self.active_trades[symbol]