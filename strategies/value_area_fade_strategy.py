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

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session_times
from analytics.indicators import calculate_atr
from analytics.relative_volume import has_high_relative_volume
from analytics.vwap import VWAPCalculator

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
        self.gap_skip_thresh = params.get("gap_skip_threshold", 1.0)
        self.allowed_shapes  = set(map(str.upper, params.get("allowed_shapes", ["D"])))

        self.min_dip_ticks   = params.get("min_dip_ticks", 3)
        self.min_rr          = params.get("min_reward_risk_ratio", 1.0)
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
        self.risk_pct          = params.get("risk_per_trade_pct", 0.01)
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
        self.profile_type = params.get("profile_type", "volume").lower()
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
        self.todays_bars: Dict[str, pd.DataFrame] = {}
        self.vwap_calculators: Dict[str, VWAPCalculator] = {}
        self.daily_poc_cache: Dict[str, list] = {}

        self.log_parameters()

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
            "Gap Skip Threshold (ATRs)": self.gap_skip_thresh,
            "Allowed Shapes": list(self.allowed_shapes),
            "Min Dip (ticks)": self.min_dip_ticks,
            "Min Reward/Risk Ratio": self.min_rr,
            "Risk Per Trade %": self.risk_pct,
            "Liquidity Cap %": self.liquidity_cap_pct,
            "Tick Size": self.tick_size,
            "Profile Engine": self.profile_type,
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

    def on_market_open(self, historical_data: Dict[str, pd.DataFrame]):
        self.todays_bars = {}
        self.vwap_calculators = {}

        # A list to hold symbols that pass all our filters for the day
        valid_candidates = []

        self.logger.info(
            "Starting pre-market screening with POC Cluster filter using %s...",
            self._profiler.__class__.__name__,
        )

        for sym, df in historical_data.items():
            if df.empty or len(df.index.normalize().unique()) < self.poc_lookback_days:
                continue

            today = df.index.normalize().max()

            # --- EFFICIENT POC CLUSTER CHECK ---
            if sym not in self.daily_poc_cache:
                self.daily_poc_cache[sym] = []
                # Initial cache fill on the first run for this symbol
                lookback_start_date = today - pd.Timedelta(days=self.poc_lookback_days)
                lookback_df = df[(df.index >= lookback_start_date) & (df.index < today)]

                # Get the most recent N unique days
                unique_days_in_lookback = sorted(lookback_df.index.normalize().unique())
                if len(unique_days_in_lookback) < self.poc_lookback_days:
                    continue

                actual_lookback_days = unique_days_in_lookback[-self.poc_lookback_days:]

                for date in actual_lookback_days:
                    day_df = lookback_df[lookback_df.index.date == date.date()]
                    if day_df.empty: continue
                    stats = self._profiler.calculate(day_df)
                    if stats and 'poc_price' in stats:
                        self.daily_poc_cache[sym].append(stats['poc_price'])
            else:
                yday_df = df[df.index.date == (today.date() - pd.Timedelta(days=1))]
                if not yday_df.empty:
                    stats = self._profiler.calculate(yday_df)
                    if stats and 'poc_price' in stats:
                        self.daily_poc_cache[sym].append(stats['poc_price'])
                        while len(self.daily_poc_cache[sym]) > self.poc_lookback_days:
                            self.daily_poc_cache[sym].pop(0)

            # Now, perform the dispersion check on the cached POCs
            cached_pocs = self.daily_poc_cache[sym]
            if len(cached_pocs) < self.poc_lookback_days * 0.8:
                continue

            poc_mean = np.mean(cached_pocs)
            poc_std_dev = np.std(cached_pocs)
            if poc_mean == 0: continue

            poc_dispersion = poc_std_dev / poc_mean
            if poc_dispersion > self.poc_dispersion_threshold:
                continue # Skip if dispersion is too high

            # --- MINIMUM ATR CHECK ---
            daily_agg = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
            if len(daily_agg) < 2: continue

            atr_series = calculate_atr(daily_agg, period=self.atr_period)
            yday_atr = atr_series.iloc[-2]
            yday_close = daily_agg['close'].iloc[-2]

            if yday_close > 0 and (yday_atr / yday_close) < self.min_atr_pct:
                continue # Skip if volatility is too low

            # If all checks pass, this is a valid candidate for today
            valid_candidates.append(sym)

        # --- FINAL LEVEL CALCULATION (only for valid candidates) ---
        self.logger.info(f"Found {len(valid_candidates)} valid candidates after screening.")
        for sym in valid_candidates:
            df = historical_data[sym]
            today = df.index.normalize().max()
            yday_df = df[df.index.date == (today.date() - pd.Timedelta(days=1))]
            yday_stats = self._profiler.calculate(yday_df)

            daily_agg = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
            atr_series = calculate_atr(daily_agg, period=self.atr_period)
            yday_atr = atr_series.iloc[-1]

            if yday_stats:
                self.value_areas[sym] = {
                    "VAH": yday_stats["value_area_high"],
                    "VAL": yday_stats["value_area_low"],
                    "POC": yday_stats["poc_price"],
                    "shape": yday_stats.get("shape", "D").upper(),
                    "ATR": yday_atr,
                }

    def _position_size(self, entry: float, stop: float) -> int:
        risk_per_share = max(entry - stop, self.tick_size)
        eq = self.ledger.get_total_equity(self.current_prices)
        by_risk = int((eq * self.risk_pct) / risk_per_share)
        by_liq  = int((eq * self.liquidity_cap_pct) / entry)
        return max(1, min(by_risk, by_liq))

    def on_bar(self, symbol: str, bar: pd.Series):
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
            return
        super().on_bar(symbol, bar)

        today = bar.name.date()
        now   = bar.name.time()

        if symbol not in self.todays_bars or self.todays_bars[symbol].iloc[-1].name.date() != today:
            self.todays_bars[symbol] = pd.DataFrame([bar])
        else:
            self.todays_bars[symbol] = pd.concat([self.todays_bars[symbol], pd.DataFrame([bar])])

        # --- VWAP Calculation ---
        if symbol not in self.vwap_calculators:
            self.vwap_calculators[symbol] = VWAPCalculator()
        calculator = self.vwap_calculators[symbol]
        calculator.update(bar)
        # We need to request bands for both the trigger and the gate sigma levels
        sigmas_to_calc = {self.trigger_vwap_sigma, self.vwap_gate_sigma}
        vwap_data = calculator.get_vwap_bands(sigmas=list(sigmas_to_calc))
        current_vwap = vwap_data.get('vwap')
        bands = vwap_data.get('bands', {})

        # --- Gating Conditions ---
        vwap_simple_gate_ok = True
        if self.use_vwap_filter and current_vwap:
            vwap_simple_gate_ok = bar['close'] > current_vwap

        relative_volume_ok = True
        if self.use_rel_vol_filter:
            hist_profiles = self.hist_cum_vol.get(symbol, {})
            relative_volume_ok = has_high_relative_volume(
                todays_intraday=self.todays_bars[symbol],
                hist_cum_vol_profiles=hist_profiles,
                bar_ts=bar.name,
                ratio_threshold=self.rel_vol_ratio
            )

        # --- Independent VWAP Standard Deviation Gate ---
        vwap_stdev_gate_ok = True
        if self.use_vwap_stdev_gate:
            lower_gate_band = bands.get(f'lower_{self.vwap_gate_sigma}s')
            if lower_gate_band is not None:
                vwap_stdev_gate_ok = bar['close'] < lower_gate_band
            else:
                vwap_stdev_gate_ok = False

        # --- Yesterday's Levels & Filters ---
        yday = today - timedelta(days=1)
        lv = self.value_areas.get(symbol, {}).get(yday)
        if not lv or pd.isna(lv["ATR"]):
            return

        if lv["shape"] not in self.allowed_shapes:
            return

        if symbol in self.last_trade_time and bar.name - self.last_trade_time[symbol] < self.cooldown_period:
            return

        # --- Calculate Flexible Entry Trigger ---
        entry_trigger_price = None
        lower_vwap_trigger_band = bands.get(f'lower_{self.trigger_vwap_sigma}s')

        if self.trigger_mode == 'val_only':
            entry_trigger_price = lv["VAL"]
        elif self.trigger_mode == 'vwap_only':
            if lower_vwap_trigger_band is not None:
                entry_trigger_price = lower_vwap_trigger_band
        elif self.trigger_mode == 'weighted_average':
            if lower_vwap_trigger_band is not None:
                entry_trigger_price = (lv["VAL"] * self.weight_val) + (lower_vwap_trigger_band * self.weight_vwap)
            else:
                entry_trigger_price = lv["VAL"]

        if entry_trigger_price is None:
            return

        # --- Dip Detection (based on flexible trigger) ---
        is_below_trigger = bar["close"] < entry_trigger_price - self.min_dip_ticks * self.tick_size

        if is_below_trigger and symbol not in self.dip_started:
            self.dip_started[symbol] = bar.name
            self.logger.info(
                "%s | %s | below entry trigger %.2f at %.2f", symbol, bar.name.time(), entry_trigger_price, bar['close']
            )
        elif not is_below_trigger and symbol in self.dip_started:
            self.logger.info(
                "%s | %s | back above trigger (dip aborted)", symbol, bar.name.time()
            )
            self.dip_started.pop(symbol, None)

        confirm_ok = True
        if self.confirm_minutes > 0:
            confirm_ok = (
                    symbol in self.dip_started
                    and (bar.name - self.dip_started[symbol]).seconds >= self.confirm_minutes * 60
            )

        # ==============================================================
        # ENTRY
        # ==============================================================
        if (
                self.entry_start_time <= now <= self.entry_end_time
                and is_below_trigger
                and confirm_ok
                and vwap_simple_gate_ok
                and vwap_stdev_gate_ok
                and relative_volume_ok
                and symbol not in self.ledger.open_positions
        ):
            stop_price = bar["close"] - self.atr_stop_mult * lv["ATR"]

            if stop_price >= bar["close"]:
                return

            risk = bar["close"] - stop_price
            level = lv["POC"] if self.use_poc_target else lv["VAH"]
            potential_reward = level - bar["close"]
            rr = potential_reward / risk if risk > 0 else 0

            if rr < self.min_rr:
                self.logger.debug(f"{symbol} skipped: R:R={rr:.2f} below min {self.min_rr}")
                return

            qty = self._position_size(bar["close"], stop_price)
            if qty == 0:
                return

            ## changing this to be true mean reversion: target_price = bar["close"] + (risk * self.min_rr)
            target_price = lv["POC"] if self.use_poc_target else lv["VAH"]

            if self.ledger.record_trade(
                    timestamp=bar.name,
                    symbol=symbol,
                    quantity=qty,
                    price=bar["close"],
                    order_type="BUY",
                    market_prices=self.current_prices,
            ):
                self.active_trades[symbol] = {"stop": stop_price, "target": target_price}
                self.last_trade_time[symbol] = bar.name
                self.logger.info(
                    f"{bar.name} | {symbol} | LONG {qty}@{bar['close']:.2f} "
                    f"TG:{target_price:.2f} SL:{stop_price:.2f}"
                )
        # ==============================================================
        # EXIT management
        # ==============================================================
        if symbol in self.ledger.open_positions and symbol in self.active_trades:
            pos   = self.ledger.open_positions[symbol]
            trade = self.active_trades[symbol]

            hit_stop   = bar["close"] <= trade["stop"]
            hit_target = bar["close"] >= trade["target"]
            eod        = now >= self.exit_time

            if hit_stop or hit_target or eod:
                reason = "SL" if hit_stop else "TP" if hit_target else "EOD"
                qty = pos["quantity"]

                self.ledger.record_trade(
                    timestamp=bar.name,
                    symbol=symbol,
                    quantity=qty,
                    price=bar["close"],
                    order_type="SELL",
                    market_prices=self.current_prices,
                    exit_reason=reason,
                )
                self.active_trades.pop(symbol, None)
                self.logger.info(
                    f"{bar.name} | {symbol} | EXIT {qty}@{bar['close']:.2f} ({reason})"
                )

    def on_session_end(self):
        """No end-of-day actions yet."""
        pass