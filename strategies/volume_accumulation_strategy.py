import pandas as pd
from datetime import time
import pytz

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, get_session

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
        self.ny_timezone             = pytz.timezone('America/New_York')

        # --- Strategy State ---
        self.candidates = {}
        self.intraday_data = {}

        # --- NEW: Reporting Stats ---
        self.report_stats = {
            'total_symbols_evaluated': 0,
            'passed_consolidation': 0,
            'breakouts_detected': 0,
            'setups_triggered': 0,
            'entries_confirmed': 0,
        }

    def get_required_lookback(self) -> int:
        return max(self.consolidation_days, self.volume_lookback_days)

    def on_market_open(self, historical_data: dict[str, pd.DataFrame]):
        self.logger.info(f"--- Pre-calculating consolidation metrics for {len(historical_data)} symbols ---")
        self.candidates.clear()
        self.intraday_data.clear()
        self.report_stats['total_symbols_evaluated'] += len(historical_data)

        for symbol, df in historical_data.items():
            if df.empty or len(set(df.index.date)) < self.consolidation_days:
                continue

            df.index = df.index.tz_convert(self.ny_timezone)
            rth_df = df.between_time('09:30', '16:00', inclusive='left')

            if rth_df.empty: continue

            consol_high = rth_df['high'].max()
            consol_low = rth_df['low'].min()
            if consol_low == 0 or (consol_high - consol_low) / consol_low > self.consolidation_range_pct: continue

            avg_daily_volume = rth_df.groupby(rth_df.index.date)['volume'].sum().mean()
            profile = VolumeProfiler(self.tick_size).calculate(rth_df)

            if profile:
                self.candidates[symbol] = {
                    'status': 'monitoring', 'consol_high': consol_high,
                    'stop_loss': consol_low, 'avg_daily_volume': avg_daily_volume,
                    'poc': profile['poc_price'],
                }

        self.report_stats['passed_consolidation'] += len(self.candidates)
        self.logger.info(f"Found {len(self.candidates)} symbols with valid consolidations to monitor.")

    def on_bar(self, symbol: str, bar: pd.Series):
        super().on_bar(symbol, bar)
        if symbol not in self.intraday_data: self.intraday_data[symbol] = []
        self.intraday_data[symbol].append(bar)

        if symbol in self.active_trades: self._manage_active_trade(symbol, bar)
        elif symbol in self.candidates: self._scan_and_handle_entry(symbol, bar)

    def _scan_and_handle_entry(self, symbol: str, bar: pd.Series):
        ny_time = bar.name.tz_convert(self.ny_timezone)
        if not (time(9, 30) <= ny_time.time() < time(16, 0)): return

        candidate = self.candidates[symbol]

        if candidate['status'] == 'monitoring':
            minutes_since_open = (ny_time.hour - 9) * 60 + (ny_time.minute - 30)
            fraction_of_day = min(1.0, minutes_since_open / 390)
            rth_bars = [b for b in self.intraday_data[symbol] if time(9, 30) <= b.name.tz_convert(self.ny_timezone).time() < time(16, 0)]
            current_rth_volume = sum(b['volume'] for b in rth_bars)
            expected_volume = candidate['avg_daily_volume'] * fraction_of_day
            is_high_volume = current_rth_volume > (expected_volume * self.breakout_volume_ratio)
            is_breakout = bar['close'] > candidate['consol_high']
            price_ok = self.min_price <= bar['close'] <= self.max_price

            if price_ok and is_high_volume and is_breakout:
                self.logger.info(f"  [BREAKOUT DETECTED] {symbol} @ {bar['close']:.2f} on high relative volume.")
                candidate['status'] = 'watching_for_retest'
                self.report_stats['breakouts_detected'] += 1

        elif candidate['status'] == 'watching_for_retest' and bar['low'] <= candidate['poc']:
            candidate.update({'status': 'triggered', 'confirmation_high': bar['high']})
            self.logger.info(f"  [SETUP] {symbol} re-tested POC {candidate['poc']:.2f}. Waiting for confirmation above {bar['high']:.2f}.")
            self.report_stats['setups_triggered'] += 1

        elif candidate['status'] == 'triggered' and bar['high'] > candidate['confirmation_high']:
            entry_price = candidate['confirmation_high']
            risk_per_share = entry_price - candidate['stop_loss']
            if risk_per_share <= 0:
                del self.candidates[symbol]
                return

            equity = self.ledger.get_total_equity(self.current_prices)
            quantity = int((equity * self.risk_per_trade_pct) / risk_per_share)

            if quantity > 0:
                self.logger.info(f"  [ENTRY] {symbol}: Confirmed above {entry_price:.2f}.")
                if self.ledger.record_trade(bar.name, symbol, quantity, entry_price, 'BUY', self.current_prices):
                    self.active_trades[symbol] = {
                        'entry_price': entry_price, 'stop_loss': candidate['stop_loss'],
                        'profit_target': entry_price + (risk_per_share * self.profit_target_r),
                        'quantity': quantity, 'risk_per_share': risk_per_share
                    }
                    self.report_stats['entries_confirmed'] += 1
            if symbol in self.candidates: del self.candidates[symbol]

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
        self.intraday_data.clear()

    def generate_report(self):
        self.logger.info("\n--- Strategy Funnel Report ---")

        # --- Phase 1: Consolidation ---
        passed_consol = self.report_stats['passed_consolidation']
        total_eval = self.report_stats['total_symbols_evaluated']
        failed_consol = total_eval - passed_consol
        pass_pct = (passed_consol / total_eval * 100) if total_eval > 0 else 0
        self.logger.info(f"\n[Phase 1: Consolidation Screening]")
        self.logger.info(f"  - Total Symbols Evaluated: {total_eval}")
        self.logger.info(f"  - Passed: {passed_consol} ({pass_pct:.2f}%)")
        self.logger.info(f"  - Failed: {failed_consol}")

        # --- Phase 2: Breakout ---
        breakouts = self.report_stats['breakouts_detected']
        no_breakout = passed_consol - breakouts
        pass_pct = (breakouts / passed_consol * 100) if passed_consol > 0 else 0
        self.logger.info(f"\n[Phase 2: Breakout Detection (from Consolidated)]")
        self.logger.info(f"  - Passed: {breakouts} ({pass_pct:.2f}%)")
        self.logger.info(f"  - Failed (No Breakout): {no_breakout}")

        # --- Phase 3: POC Retest (Setup) ---
        setups = self.report_stats['setups_triggered']
        no_retest = breakouts - setups
        pass_pct = (setups / breakouts * 100) if breakouts > 0 else 0
        self.logger.info(f"\n[Phase 3: POC Retest (from Breakouts)]")
        self.logger.info(f"  - Passed: {setups} ({pass_pct:.2f}%)")
        self.logger.info(f"  - Failed (No Retest): {no_retest}")

        # --- Phase 4: Entry Confirmation ---
        entries = self.report_stats['entries_confirmed']
        no_confirm = setups - entries
        pass_pct = (entries / setups * 100) if setups > 0 else 0
        self.logger.info(f"\n[Phase 4: Entry Confirmation (from Setups)]")
        self.logger.info(f"  - Passed (Trades Entered): {entries} ({pass_pct:.2f}%)")
        self.logger.info(f"  - Failed (No Confirmation): {no_confirm}")

        self.logger.info("\n--------------------------------\n")