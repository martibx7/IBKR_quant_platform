import pandas as pd
from datetime import time
import pytz

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler
from analytics.indicators import calculate_vwap
from analytics.relative_volume import has_high_relative_volume

class VolumeAccumulationStrategy(BaseStrategy):
    """
    A multi-path strategy that uses dual-profile analysis and relative
    volume to identify and trade high-probability breakout-retest scenarios.
    """
    def __init__(self, symbols: list[str], ledger, config: dict, **params):
        super().__init__(symbols, ledger, config, **params)
        self.params = params
        self.timezone = pytz.timezone(self.params.get('timezone', 'America/New_York'))
        self.risk_per_trade_pct = self.params.get('risk_per_trade_pct', 0.01)
        self.min_price = self.params.get('min_price', 2.0)
        self.max_price = self.params.get('max_price', 2000.0)

        # Path 1: Consolidation Retest Params
        self.consolidation_days = self.params.get('consolidation_days', 10)
        self.consolidation_range_pct = self.params.get('consolidation_range_pct', 0.12)
        self.poc_midpoint_tolerance_pct = self.params.get('poc_midpoint_tolerance_pct', 0.20)
        self.value_area_pct = self.params.get('value_area_pct', 0.70)
        self.stop_loss_poc_buffer_pct = self.params.get('stop_loss_poc_buffer_pct', 0.01)
        self.min_stop_loss_ticks = self.params.get('min_stop_loss_ticks', 3)
        self.liquidity_cap_pct = self.params.get('liquidity_cap_pct', 0.02)
        self.profit_target_r = self.params.get('profit_target_r', 1.5)
        self.breakeven_trigger_r = self.params.get('breakeven_trigger_r', 0.75)

        # --- NEW: Configurable confirmation bars ---
        self.retest_confirmation_bars = self.params.get('retest_confirmation_bars', 2)

        time_params = self.params.get('entry_time_window', {})
        self.entry_start_time = time.fromisoformat(time_params.get('start', '10:00:00'))
        self.entry_end_time = time.fromisoformat(time_params.get('end', '15:00:00'))

        # Relative Volume Params
        self.enable_rel_vol_check = self.params.get('enable_relative_volume_check', True)
        self.rel_vol_lookback = self.params.get('relative_volume_lookback', 20)
        self.rel_vol_ratio = self.params.get('relative_volume_ratio', 2.5)

        # Path 2: Open-Drive Params
        self.enable_open_drive = self.params.get('enable_open_drive', False)
        self.open_drive_r_target = self.params.get('open_drive_r_target', 1.5)
        self.vwap_stop_pct = self.params.get('vwap_stop_pct', 0.005)

        # General properties
        self.tick_size_volume = config.get('backtest', {}).get('tick_size_volume_profile', 0.01)
        self.tick_size_market = config.get('backtest', {}).get('tick_size_market_profile', 0.01)

        self.candidates = {}
        self.active_trades = {}
        self.todays_traded = set()
        self.session_dataframes = {}
        self.historical_intraday = {}

        # --- RESTORED: Parameter logging ---
        self.logger.info(f"Strategy '{self.__class__.__name__}' initialized with parameters: {self.params}")

    def get_required_lookback(self) -> int:
        return max(self.consolidation_days, self.rel_vol_lookback) + 5

    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        self.logger.info("--- Screening for candidates ---")
        self.candidates.clear()
        self.todays_traded.clear()
        self.session_dataframes.clear()
        self.historical_intraday = historical_data

        for symbol, df in historical_data.items():
            if df.empty or len(set(df.index.date)) < max(self.consolidation_days, self.rel_vol_lookback):
                continue

            rth_df = df.between_time('09:30', '16:00', inclusive='left')
            if rth_df.empty: continue

            daily_df = rth_df.resample('D').agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            ).dropna()

            if daily_df.empty: continue

            last_close = daily_df.iloc[-1]['close']
            if not (self.min_price <= last_close <= self.max_price):
                continue

            consolidation_df = daily_df.tail(self.consolidation_days)
            if len(consolidation_df) < self.consolidation_days: continue

            consol_high = consolidation_df['high'].max()
            consol_low = consolidation_df['low'].min()
            if consol_low <= 0 or not (((consol_high - consol_low) / consol_low) <= self.consolidation_range_pct):
                continue

            volume_profiler = VolumeProfiler(tick_size=self.tick_size_volume, value_area_pct=self.value_area_pct)
            vol_profile = volume_profiler.calculate(consolidation_df)
            if not vol_profile: continue

            poc = vol_profile['poc_price']
            midpoint = consol_low + ((consol_high - consol_low) * 0.5)
            tolerance = (consol_high - consol_low) * self.poc_midpoint_tolerance_pct
            if not (midpoint - tolerance <= poc <= midpoint + tolerance):
                continue

            market_profiler = MarketProfiler(tick_size=self.tick_size_market, value_area_pct=self.value_area_pct)
            mkt_profile = market_profiler.calculate(consolidation_df)
            if not mkt_profile: continue

            self.candidates[symbol] = {
                'volume_poc': vol_profile['poc_price'],
                'volume_vah': vol_profile['value_area_high'],
                'tpo_vah': mkt_profile['value_area_high'],
                'status_retest': 'pending',
                'confirmation_count': 0, # NEW: Add counter for confirmation bars
                'hist_cum_vol_profiles': {}
            }

            if self.enable_rel_vol_check:
                hist_groups = rth_df.groupby(rth_df.index.date)
                sorted_dates = sorted([date for date, _ in hist_groups], reverse=True)
                for date in sorted_dates[1:self.rel_vol_lookback+1]:
                    day_df = hist_groups.get_group(date).copy()
                    day_df.sort_index(inplace=True)
                    self.candidates[symbol]['hist_cum_vol_profiles'][date] = day_df['volume'].cumsum()

            if self.enable_open_drive:
                if symbol not in self.candidates: self.candidates[symbol] = {}
                self.candidates[symbol]['status_drive'] = 'monitoring_drive'
                self.candidates[symbol]['opening_high'] = 0

        self.logger.info(f"Found {len(self.candidates)} potential candidates for all paths.")

    def on_bar(self, symbol: str, bar: pd.Series):
        if symbol in self.todays_traded: return
        super().on_bar(symbol, bar)

        if symbol not in self.session_dataframes:
            self.session_dataframes[symbol] = pd.DataFrame([bar], index=[bar.name])
        else:
            self.session_dataframes[symbol].loc[bar.name] = bar

        if symbol in self.active_trades:
            self._manage_active_trade(symbol, bar)
        elif symbol in self.candidates:
            self._scan_for_retest_entry(symbol, bar)
            if self.enable_open_drive:
                self._scan_for_drive_entry(symbol, bar)

    def _scan_for_retest_entry(self, symbol: str, bar: pd.Series):
        candidate = self.candidates[symbol]
        if candidate.get('status_retest') == 'trade_taken': return

        status = candidate.get('status_retest', 'invalid')
        if status == 'invalid': return

        breakout_level = candidate['tpo_vah']
        retest_level = candidate['volume_vah']

        if status == 'pending':
            if bar['close'] > breakout_level:
                is_volume_confirmed = True
                if self.enable_rel_vol_check:
                    is_volume_confirmed = has_high_relative_volume(
                        todays_intraday=self.session_dataframes[symbol],
                        hist_cum_vol_profiles=candidate['hist_cum_vol_profiles'],
                        bar_ts=bar.name,
                        ratio_threshold=self.rel_vol_ratio
                    )
                if is_volume_confirmed:
                    self.logger.info(f"  [RETEST PATH - BREAKOUT] {symbol} broke TPO VAH {breakout_level:.2f} with volume confirmation.")
                    candidate['status_retest'] = 'watching_for_retest'

        elif status == 'watching_for_retest' and bar['low'] <= retest_level:
            self.logger.info(f"  [RETEST PATH - RETEST] {symbol} re-tested Volume VAH at {retest_level:.2f}.")
            candidate['status_retest'] = 'retest_confirmed'

        elif status == 'retest_confirmed':
            ny_time = bar.name.tz_convert(self.timezone).time()
            if not (self.entry_start_time <= ny_time <= self.entry_end_time):
                return

            if bar['close'] > retest_level:
                candidate['confirmation_count'] += 1
                self.logger.info(f"  [RETEST PATH - CONFIRMATION BAR {candidate['confirmation_count']}/{self.retest_confirmation_bars}] for {symbol}.")
            else:
                candidate['confirmation_count'] = 0

            if candidate['confirmation_count'] >= self.retest_confirmation_bars:
                self.logger.info(f"  [RETEST PATH - FULL CONFIRMATION] {symbol} confirmed support at {retest_level:.2f}.")
                self._execute_retest_trade(symbol, bar, candidate)

    def _scan_for_drive_entry(self, symbol: str, bar: pd.Series):
        candidate = self.candidates[symbol]
        if candidate.get('status_drive') != 'monitoring_drive': return

        if candidate.get('opening_high', 0) == 0 and bar.name.time() >= time(9, 31):
            candidate['opening_high'] = self.session_dataframes[symbol].iloc[0].high

        if bar.name.time() == time(10, 00):
            current_df = self.session_dataframes[symbol]
            vwap_series = calculate_vwap(current_df)['vwap']
            if vwap_series.empty: return

            if bar['close'] > candidate['opening_high'] and bar['close'] > vwap_series.iloc[-1]:
                self.logger.info(f"  [DRIVE PATH - TRIGGER] {symbol} met open-drive criteria.")
                self._execute_drive_trade(symbol, bar, vwap_series.iloc[-1])

    def _execute_retest_trade(self, symbol: str, bar: pd.Series, candidate: dict):
        entry_price = bar['close']
        poc = candidate.get('volume_poc')
        if not poc: return

        buffer_from_pct = poc * self.stop_loss_poc_buffer_pct
        buffer_from_ticks = self.tick_size_volume * self.min_stop_loss_ticks
        final_buffer = max(buffer_from_pct, buffer_from_ticks)
        stop_loss_price = poc - final_buffer
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0: return
        quantity = self._calculate_quantity(entry_price, risk_per_share, bar['volume'])
        if quantity == 0: return

        if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
            self.active_trades[symbol] = {
                'entry_price': entry_price, 'stop_loss': stop_loss_price,
                'profit_target': entry_price + (risk_per_share * self.profit_target_r),
                'quantity': quantity, 'risk_per_share': risk_per_share, 'path': 'retest'
            }
            self.todays_traded.add(symbol)
            candidate['status_retest'] = 'trade_taken'

    def _execute_drive_trade(self, symbol: str, bar: pd.Series, vwap: float):
        entry_price = bar['close']
        stop_loss_price = vwap * (1 - self.vwap_stop_pct)
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0: return
        quantity = self._calculate_quantity(entry_price, risk_per_share, bar['volume'])
        if quantity == 0: return

        if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
            self.active_trades[symbol] = {
                'entry_price': entry_price, 'stop_loss': stop_loss_price,
                'profit_target': entry_price + (risk_per_share * self.open_drive_r_target),
                'quantity': quantity, 'risk_per_share': risk_per_share, 'path': 'drive'
            }
            self.todays_traded.add(symbol)
            self.candidates[symbol]['status_drive'] = 'trade_taken'

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

    def on_session_end(self):
        pass