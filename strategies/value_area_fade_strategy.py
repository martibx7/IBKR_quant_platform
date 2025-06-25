from __future__ import annotations

"""
Intraday Value-Area Fade – v4.1.0

Upgrades vs. v4.0.x
-----------------
* Separated VWAP Stdev logic into a flexible Trigger and an independent Gate.
* Configuration is now more modular to allow combining different trigger/gate rules.
"""

from datetime import time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import time as perf_timer
import json
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session_times
from analytics.indicators import calculate_atr
from analytics.relative_volume import has_high_relative_volume
from analytics.vwap import VWAPCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ValueAreaFadeStrategy(BaseStrategy):
    """Long-only value-area fade with volatility-aware risk controls."""

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_symbol_file(path_like: str) -> Set[str]:
        p = Path(path_like).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Symbol file not found: {p}")
        with p.open() as fh:
            return {
                line.strip().upper()
                for line in fh
                if line.strip() and not line.lstrip().startswith("#")
            }

    # ------------------------------------------------------------------ #
    # Life-cycle                                                          #
    # ------------------------------------------------------------------ #
    def __init__(self, symbols: List[str], ledger, config: Dict, **params):
        super().__init__(symbols, ledger, config, **params)

        # -------- watch-list ----------------------------------------- #
        self.symbol_whitelist: Optional[Set[str]] = None
        if params.get("use_symbol_file"):
            symbol_file = params.get("symbol_file")
            if not symbol_file:
                raise ValueError("'symbol_file' required when 'use_symbol_file' is true")
            self.symbol_whitelist = self._load_symbol_file(symbol_file)
            self.symbols = [s for s in self.symbols if s in self.symbol_whitelist]
            self.current_prices = {s: 0.0 for s in self.symbols}


        # -------- timing --------------------------------------------- #
        win = params.get("entry_time_window", {})
        self.entry_start_time = time.fromisoformat(win.get("start", "10:00:00"))
        self.entry_end_time   = time.fromisoformat(win.get("end",   "15:00:00"))
        self.exit_time        = time.fromisoformat(params.get("exit_time", "15:55:00"))
        self.session_open_time = time.fromisoformat(
            params.get("session_open_time", "09:30:00")
        )
        self.cooldown_period = timedelta(
            minutes=params.get("cooldown", {}).get("minutes", 30)
        )

        # -------- params / filters ----------------------------------- #
        self.lookback_days   = params.get("lookback_days", 30)
        self.poc_lookback_days = params.get("poc_lookback_days", 10)
        self.value_area_pct  = params.get("value_area_pct", 0.70)
        self.confirm_minutes = params.get("confirmation_window_minutes", 0)

        # --- NEW: Profile and Stability Filter Parameters ---
        self.profile_type = params.get("profile_type", "volume").lower()
        self.poc_dispersion_threshold = params.get("poc_dispersion_threshold", 0.015) # e.g., 1.5% dispersion max
        self.min_atr_pct = params.get("min_atr_pct", 0.02) # e.g., Require ATR to be at least 2% of the stock price


        self.atr_stop_mult   = params.get("atr_stop_mult", 0.5)
        self.gap_skip_threshold = params.get('gap_skip_threshold', 1.0)
        self.allowed_shapes  = set(map(str.upper, params.get("allowed_shapes", ["D"])))

        self.min_dip_ticks   = params.get("min_dip_ticks", 3)
        self.min_rr = params.get('min_reward_risk_ratio', 1.0)
        primary = str(params.get("primary_target", "poc")).lower()
        if primary not in {"poc", "vah"}:
            raise ValueError(f"Invalid primary_target: {primary}")
        self.use_poc_target = (primary == "poc")
        self.use_vah_target = (primary == "vah")

        # --- Simple VWAP Filter (kept for backward compatibility) ---
        self.use_vwap_filter = params.get("use_vwap_filter", False)

        # --- Relative Volume filter parameters ---
        rel_vol_params = params.get("relative_volume_filter", {})
        self.use_rel_vol_filter = rel_vol_params.get("enable", False)
        self.rel_vol_ratio      = rel_vol_params.get("ratio_threshold", 2.0)

        # -------- risk / sizing -------------------------------------- #
        self.liquidity_cap_pct = params.get("liquidity_cap_pct", 0.05)

        # <-- MODIFIED: Separated Trigger and Gate configurations -->
        # --- Flexible Entry Trigger ---
        trigger_params = params.get("trigger_config", {})
        self.trigger_mode = trigger_params.get("mode", "val_only")
        self.trigger_vwap_sigma = float(trigger_params.get("vwap_sigma_level", 2.0))
        self.weight_val = float(trigger_params.get("weight_val", 0.5))
        self.weight_vwap = float(trigger_params.get("weight_vwap", 0.5))
        if self.trigger_mode not in ["val_only", "vwap_only", "weighted_average"]:
            raise ValueError(f"Invalid trigger mode: {self.trigger_mode}")
        if self.trigger_mode == "weighted_average" and not np.isclose(self.weight_val + self.weight_vwap, 1.0):
            raise ValueError("Weights for VAL and VWAP must sum to 1.0")

        # --- Independent VWAP Standard Deviation Gate ---
        gate_params = params.get("vwap_stdev_gate", {})
        self.use_vwap_stdev_gate = gate_params.get("enable", False)
        self.vwap_gate_sigma = float(gate_params.get("sigma_level", 2.0))
        self.risk_per_trade_pct = params.get('risk_per_trade_pct', 0.01)
        # <-- END MODIFIED -->

        # -------- tick size ------------------------------------------ #
        self.tick_size = float(
            self.config.get("backtest", {}).get(
                "tick_size_market_profile",
                self.config.get("backtest", {}).get("tick_size_volume_profile", 0.01),
            )
        )

        # -------- ATR period ----------------------------------------- #
        self.atr_period = int(
            self.config.get("backtest", {}).get("atr_period", 14)
        )

        # -------- profile engine ------------------------------------- #
        if self.profile_type not in {"volume", "market"}:
            raise ValueError("profile_type must be 'volume' or 'market'")
        self._profiler = (
            VolumeProfiler(self.tick_size, self.value_area_pct)
            if self.profile_type == "volume"
            else MarketProfiler(self.tick_size, self.value_area_pct)
        )

        # -------- state containers ----------------------------------- #
        self.value_areas: Dict[str, Dict] = {s: {} for s in self.symbols}
        self.last_trade_time: Dict[str, pd.Timestamp] = {}
        self.dip_started: Dict[str, pd.Timestamp] = {}
        self.skip_day: Dict[str, pd.Timestamp] = {}
        self.active_trades: Dict[str, Dict] = {}
        self.hist_cum_vol: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
        self.vwap_calculators: Dict[str, VWAPCalculator] = {}
        self.daily_poc_cache: Dict[str, list] = {}

        self.log_parameters()

    def _precalculate_historical_volume(self, historical_df: pd.DataFrame, lookback_days: int) -> dict:
        """
        Calculates the average cumulative volume at each minute of the trading day
        over a specified lookback period.
        """
        # Ensure we only look at the last N days from the historical data provided
        unique_dates = sorted(historical_df.index.date.unique())
        lookback_dates = unique_dates[-lookback_days:]
        df = historical_df[historical_df.index.date.isin(lookback_dates)]

        # Group by time of day and calculate the average cumulative volume
        df = df.copy()
        df['time_of_day'] = df.index.time
        df['cumulative_volume'] = df.groupby(df.index.date)['volume'].cumsum()

        # Calculate the average cumulative volume for each minute
        avg_cumulative_volume_profile = df.groupby('time_of_day')['cumulative_volume'].mean().to_dict()

        return avg_cumulative_volume_profile

    def log_parameters(self):
        """Helper to log all strategy parameters."""
        # <-- MODIFIED: Updated logging for new config structure -->
        params_to_log = {
            "Symbols": f"{len(self.symbols)} loaded" + (f" from {self.symbol_whitelist}" if self.symbol_whitelist else ""),
            "Time Window": f"{self.entry_start_time.isoformat()} - {self.entry_end_time.isoformat()}",
            "Exit Time": self.exit_time.isoformat(),
            "Cooldown Period (minutes)": self.cooldown_period.total_seconds() / 60,
            "Lookback Days": self.lookback_days,
            "Value Area %": self.value_area_pct,
            "Confirmation Minutes": self.confirm_minutes,
            "Profile Type": self.profile_type,
            "POC Lookback Days": self.poc_lookback_days,
            "POC Dispersion Threshold": self.poc_dispersion_threshold,
            "Min ATR %": self.min_atr_pct,
            "ATR Stop Multiplier": self.atr_stop_mult,
            "Gap Skip Threshold (ATRs)": self.gap_skip_threshold,
            "Allowed Shapes": list(self.allowed_shapes),
            "Min Dip (ticks)": self.min_dip_ticks,
            "Min Reward/Risk Ratio": self.min_rr,
            "Risk Per Trade %": self.risk_per_trade_pct,
            "Liquidity Cap %": self.liquidity_cap_pct,
            "Tick Size": self.tick_size,
            "Use VWAP Filter (Simple)": self.use_vwap_filter,
            "Use Relative Volume Filter": self.use_rel_vol_filter,
            "Relative Volume Ratio": self.rel_vol_ratio,
            "Trigger Config": {
                "Mode": self.trigger_mode,
                "VWAP Sigma": self.trigger_vwap_sigma,
                "Weight VAL": self.weight_val,
                "Weight VWAP": self.weight_vwap
            },
            "VWAP Stdev Gate": {
                "Enable": self.use_vwap_stdev_gate,
                "Sigma Level": self.vwap_gate_sigma
            }
        }
        # <-- END MODIFIED -->
        log_message = "Strategy Initialized with Parameters:\n"
        log_message += json.dumps(params_to_log, indent=2)
        self.logger.info(log_message)


    def get_required_lookback(self) -> int:
        return max(self.lookback_days, self.poc_lookback_days, self.atr_period) + 1

    def on_market_open(self, historical_data: dict[str, pd.DataFrame], intraday_data: dict[str, pd.DataFrame]):
        self.vwap_calculators.clear()
        self.value_areas.clear()
        self.dip_started.clear()
        self.last_trade_time.clear()
        valid_candidates = []

        self.logger.info("Starting pre-market screening...")

        for sym in tqdm(self.symbols, desc="Screening Candidates"):
            df = historical_data.get(sym)
            if df is None or df.empty:
                # self.logger.debug(f"[{sym}] REJECTED: no data")
                continue

            unique_days = df.index.normalize().unique()
            if len(unique_days) < max(self.poc_lookback_days, self.atr_period):
                # self.logger.debug(f"[{sym}] REJECTED: not enough history days ({len(unique_days)} < {max(self.poc_lookback_days, self.atr_period)})")
                continue

            today = unique_days.max()
            poc_lookback_df = df[df.index < today]
            poc_unique_days = poc_lookback_df.index.normalize().unique()

            if len(poc_unique_days) < self.poc_lookback_days:
                # self.logger.debug(f"[{sym}] REJECTED: not enough POC lookback days ({len(poc_unique_days)} < {self.poc_lookback_days})")
                continue

            recent_poc_days = sorted(poc_unique_days, reverse=True)[:self.poc_lookback_days]
            pocs = []
            for d in recent_poc_days:
                day_data = poc_lookback_df[poc_lookback_df.index.date == d.date()]
                if not day_data.empty:
                    stats = self._profiler.calculate(day_data)
                    if stats and stats.get('poc_price'):
                        pocs.append(stats['poc_price'])

            if len(pocs) < self.poc_lookback_days * 0.8:
                # self.logger.debug(f"[{sym}] REJECTED: too few valid POCs ({len(pocs)} < {self.poc_lookback_days*0.8:.1f})")
                continue

            if not pocs or np.mean(pocs) == 0:
                self.logger.debug(f"[{sym}] REJECTED: mean POC zero or empty")
                continue

            poc_dispersion = np.std(pocs) / np.mean(pocs)
            if poc_dispersion > self.poc_dispersion_threshold:
                self.logger.debug(f"[{sym}] REJECTED: dispersion {poc_dispersion:.4f} > {self.poc_dispersion_threshold:.4f}")
                continue

            daily_agg = df.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            if len(daily_agg) < self.atr_period:
                self.logger.debug(f"[{sym}] REJECTED: not enough daily bars for ATR ({len(daily_agg)} < {self.atr_period})")
                continue

            atr_series = calculate_atr(daily_agg, period=self.atr_period)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                self.logger.debug(f"[{sym}] REJECTED: ATR series empty or NaN")
                continue

            yday_atr = atr_series.iloc[-1]
            yday_close = daily_agg['close'].iloc[-1]

            today_intraday_df = intraday_data.get(sym) # Always define the variable
            if self.gap_skip_threshold > 0:
                if today_intraday_df is None or today_intraday_df.empty:
                    self.logger.warning(f"[{sym}] Could not find intraday data for today to check gap. Skipping.")
                    continue
                today_open_price = today_intraday_df.iloc[0]['open']

                gap_size = abs(today_open_price - yday_close)
                max_allowed_gap = yday_atr * self.gap_skip_threshold

                if gap_size > max_allowed_gap:
                    self.logger.debug(f"[{sym}] REJECTED: Opening gap {gap_size:.2f} > threshold {max_allowed_gap:.2f}")
                    continue

            atr_pct = (yday_atr / yday_close) if yday_close > 0 else 0

            if atr_pct < self.min_atr_pct:
                self.logger.debug(f"[{sym}] REJECTED: ATR% {atr_pct:.4f} < {self.min_atr_pct:.4f}")
                continue

            if self.use_rel_vol_filter:
                # We use poc_lookback_days for this, but could be a separate param
                hist_vol_profile = self._precalculate_historical_volume(df, self.poc_lookback_days)
                if hist_vol_profile:
                    self.hist_cum_vol[sym] = hist_vol_profile
                else:
                    self.logger.warning(f"[{sym}] Could not generate historical volume profile, rel-vol filter may fail.")

            self.logger.info(f"[{sym}] PASSED SCREENING. POC Dispersion: {poc_dispersion:.4f}, ATR %: {atr_pct:.4f}")
            valid_candidates.append(sym)



        for sym in valid_candidates:
            df = historical_data.get(sym)
            if df is None: continue

            yday_df = df[df.index.date == df.index.date.max()]
            if yday_df.empty: continue

            yday_stats = self._profiler.calculate(yday_df)

            daily_agg = df.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
            atr_series = calculate_atr(daily_agg, period=self.atr_period)
            if atr_series.empty: continue

            yday_atr = atr_series.iloc[-1]

            if yday_stats and yday_stats.get('poc_price'):
                self.value_areas[sym] = {
                    "VAH": yday_stats["value_area_high"],
                    "VAL": yday_stats["value_area_low"],
                    "POC": yday_stats["poc_price"],
                    "shape": yday_stats.get("shape", "D").upper(),
                    "ATR": yday_atr,
                }
        self.logger.info(
            f"Screen summary: {len(self.symbols)} input -> {len(valid_candidates)} passed"
        )

    def _position_size(self, entry_price: float, stop_price: float, volume: float) -> int:
        """
        Returns the share count that keeps per-trade risk ≤ risk_per_trade_pct
        **and** caps size to a fixed % of the bar’s volume.
        """
        cash = self.ledger.get_cash()
        risk_per_trade = cash * self.risk_per_trade_pct
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0

        # liquidity cap: don’t take more than X % of this bar’s printed volume
        liq_cap_shares = int(volume * self.liquidity_cap_pct)

        num_shares = min(int(risk_per_trade / risk_per_share), liq_cap_shares)
        return max(num_shares, 0)

    def on_bar(self, symbol: str, bar: pd.Series):
        # 1) whitelist filter
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
            return
        super().on_bar(symbol, bar)

        # --- TIMEZONE FIX: Convert the incoming bar's timestamp to the strategy's timezone ---
        now_eastern = bar.name.astimezone(self.tz)
        today = now_eastern.date()
        now_time = now_eastern.time()

        # --- EXIT LOGIC MOVED AND CORRECTED ---
        if symbol in self.ledger.open_positions:
            pos = self.ledger.open_positions[symbol]
            trade_details = self.active_trades.get(symbol)

            if trade_details: # Ensure there is an active trade record
                hit_sl = bar["close"] <= trade_details["stop"]
                hit_tp = bar["close"] >= trade_details["target"]
                # Use the timezone-aware `now_time` for the EOD check
                eod = now_time >= self.exit_time

                if hit_sl or hit_tp or eod:
                    reason = "SL" if hit_sl else "TP" if hit_tp else "EOD"
                    self.logger.info(f"{bar.name} | {symbol} | EXIT {pos['quantity']}@{bar['close']:.2f} ({reason})")
                    self.ledger.record_trade(
                        timestamp=bar.name,
                        symbol=symbol,
                        quantity=pos["quantity"],
                        price=bar["close"],
                        order_type="SELL",
                        market_prices=self.current_prices,
                        exit_reason=reason,
                    )
                    self.active_trades.pop(symbol, None)
                    return # Stop processing this bar for this symbol after an exit

        # --- ENTRY LOGIC ---

        # 2) entry‐window filter
        if not (self.entry_start_time <= now_time <= self.entry_end_time):
            return


        # 4) VWAP simple filter
        calculator = self.vwap_calculators.setdefault(symbol, VWAPCalculator())
        calculator.update(bar)
        vwap_data = calculator.get_vwap_bands(sigmas=[self.trigger_vwap_sigma, self.vwap_gate_sigma])
        current_vwap = vwap_data.get('vwap')
        if self.use_vwap_filter and current_vwap and bar['close'] <= current_vwap:
            self.logger.debug(f"{symbol} | {now_time} | VWAP fail: close {bar['close']:.2f} <= {current_vwap:.2f}")
            return

        # 5) relative‐volume filter
        if self.use_rel_vol_filter:
            calculator = self.vwap_calculators.get(symbol)
            if not calculator:
                return
            current_cumulative_volume = calculator.get_cumulative_volume()
            hist_profile = self.hist_cum_vol.get(symbol, {})
            if not has_high_relative_volume(
                    current_cumulative_volume=current_cumulative_volume,
                    hist_cum_vol_profiles=hist_profile,
                    bar_time=now_time
            ):
                self.logger.debug(f"{symbol} | {now_time} | relative-volume fail")
                return

        # 6) VWAP‐stdev gate
        if self.use_vwap_stdev_gate:
            lower = vwap_data['bands'].get(f'lower_{self.vwap_gate_sigma}s')
            if lower is None or bar['close'] >= lower:
                self.logger.debug(f"{symbol} | {now_time} | stdev gate fail: close {bar['close']:.2f} >= {lower}")
                return

        # 7) yesterday's value‐area
        lv = self.value_areas.get(symbol)
        if not lv or pd.isna(lv["ATR"]):
            return

        # 8) shape filter
        if lv["shape"] not in self.allowed_shapes:
            return

        # 9) cooldown
        last = self.last_trade_time.get(symbol)
        if last and (bar.name - last) < self.cooldown_period:
            return

        # 10) trigger price
        lower_band = vwap_data['bands'].get(f'lower_{self.trigger_vwap_sigma}s')
        if self.trigger_mode == "val_only":
            trigger = lv["VAL"]
        elif self.trigger_mode == "vwap_only":
            trigger = lower_band
        else:  # weighted_average
            trigger = (lv["VAL"] * self.weight_val + (lower_band or lv["VAL"]) * self.weight_vwap)

        if trigger is None:
            self.logger.debug(f"{symbol} | {now_time} | no trigger price")
            return

        # 11) dip detection
        is_below = bar["close"] < trigger - self.min_dip_ticks * self.tick_size
        if not is_below:
            return
        if symbol not in self.dip_started:
            self.dip_started[symbol] = bar.name
            self.logger.info(f"{symbol} | {now_time} | dip started @ {bar['close']:.2f}")

        # 12) confirmation window
        if self.confirm_minutes > 0:
            elapsed = (bar.name - self.dip_started[symbol]).seconds
            if elapsed < self.confirm_minutes * 60:
                return

        # 13) already in a trade? (This is now a final check)
        if symbol in self.ledger.open_positions:
            return

        # 14) size & stop logic
        stop_price = bar["close"] - self.atr_stop_mult * lv["ATR"]
        if stop_price >= bar["close"]:
            self.logger.debug(f"{symbol} | {now_time} | bad stop {stop_price:.2f} >= close")
            return
        qty = self._position_size(bar["close"], stop_price, bar["volume"])
        if qty <= 0:
            return

        # 15) place entry
        target_price = lv["POC"] if self.use_poc_target else lv["VAH"]
        reward = target_price - bar["close"]
        risk = bar["close"] - stop_price
        if risk <= 0:
            self.logger.debug(f"{symbol} | {now_time} | Invalid risk calculation. Risk: {risk:.2f}")
            return
        reward_risk_ratio = reward / risk
        if reward_risk_ratio < self.min_rr:
            return

        self.logger.info(f"{bar.name} | {symbol} | ENTER {qty}@{bar['close']:.2f} TP:{target_price:.2f} SL:{stop_price:.2f}")
        self.ledger.record_trade(
            timestamp=bar.name,
            symbol=symbol,
            quantity=qty,
            price=bar["close"],
            order_type="BUY",
            market_prices=self.current_prices,
        )
        self.active_trades[symbol] = {"stop": stop_price, "target": target_price}
        self.last_trade_time[symbol] = bar.name


    def on_session_end(self):
        """No end-of-day actions yet."""
        pass