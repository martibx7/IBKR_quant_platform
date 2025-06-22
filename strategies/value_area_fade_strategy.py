# strategies/value_area_fade_strategy.py
"""
Intraday Value-Area Fade – v3.0.2

Upgrades vs. v2.x
-----------------
* Vol-scaled stop & sizing  (ATR-based)
* Profile-shape gate        (default only after D-profiles)
* Gap-day auto-skip         (open gap ≥ N ATRs)
"""

from __future__ import annotations

from datetime import time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import time as perf_timer # LOGGING: For performance timing
import json # LOGGING: To format parameters

import pandas as pd

from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler, get_session_times
from analytics.indicators import calculate_atr
# --- NEW: Import the relative volume function ---
from analytics.relative_volume import has_high_relative_volume


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
        self.lookback_days   = params.get("lookback_days", 10)
        self.value_area_pct  = params.get("value_area_pct", 0.70)
        self.confirm_minutes = params.get("confirmation_window_minutes", 0)

        self.atr_stop_mult   = params.get("atr_stop_mult", 0.5)
        self.gap_skip_thresh = params.get("gap_skip_threshold", 1.0)
        self.allowed_shapes  = set(map(str.upper, params.get("allowed_shapes", ["D"])))

        self.min_dip_ticks   = params.get("min_dip_ticks", 3)
        self.min_rr          = params.get("min_reward_risk_ratio", 1.0)

        # --- NEW: VWAP filter parameter ---
        self.use_vwap_filter = params.get("use_vwap_filter", False)

        # --- NEW: Relative Volume filter parameters ---
        rel_vol_params = params.get("relative_volume_filter", {})
        self.use_rel_vol_filter = rel_vol_params.get("enable", False)
        self.rel_vol_ratio      = rel_vol_params.get("ratio_threshold", 2.0)

        # -------- risk / sizing -------------------------------------- #
        self.risk_pct          = params.get("risk_per_trade_pct", 0.01)
        self.liquidity_cap_pct = params.get("liquidity_cap_pct", 0.05)

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
        self.intraday_vwap_data: Dict[str, Dict] = {}
        self.hist_cum_vol: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
        self.todays_bars: Dict[str, pd.DataFrame] = {}

        # LOGGING: Log all parameters at startup for easy run analysis
        self.log_parameters()

    def log_parameters(self):
        """Helper to log all strategy parameters."""
        params_to_log = {
            "Symbols": f"{len(self.symbols)} loaded" + (f" from {self.symbol_whitelist}" if self.symbol_whitelist else ""),
            "Time Window": f"{self.entry_start_time.isoformat()} - {self.entry_end_time.isoformat()}",
            "Exit Time": self.exit_time.isoformat(),
            "Cooldown Period (minutes)": self.cooldown_period.total_seconds() / 60,
            "Lookback Days": self.lookback_days,
            "Value Area %": self.value_area_pct,
            "Confirmation Minutes": self.confirm_minutes,
            "ATR Stop Multiplier": self.atr_stop_mult,
            "Gap Skip Threshold (ATRs)": self.gap_skip_thresh,
            "Allowed Shapes": list(self.allowed_shapes),
            "Min Dip (ticks)": self.min_dip_ticks,
            "Min Reward/Risk Ratio": self.min_rr,
            "Risk Per Trade %": self.risk_pct,
            "Liquidity Cap %": self.liquidity_cap_pct,
            "Tick Size": self.tick_size,
            "Profile Engine": self.profile_type,
            "Use VWAP Filter": self.use_vwap_filter,
            "Use Relative Volume Filter": self.use_rel_vol_filter,
            "Relative Volume Ratio": self.rel_vol_ratio
        }
        # Pretty print the parameters to the log
        log_message = "Strategy Initialized with Parameters:\n"
        log_message += json.dumps(params_to_log, indent=2)
        self.logger.info(log_message)


    # ------------------------------------------------------------------ #
    def get_required_lookback(self) -> int:
        """Yesterday + N-day lookback."""
        return self.lookback_days + 1

    # ------------------------------------------------------------------ #
    # Session-prep                                                       #
    # ------------------------------------------------------------------ #
    def on_market_open(self, historical_data: Dict[str, pd.DataFrame]):
        # --- NEW: Reset daily data containers ---
        self.intraday_vwap_data = {}
        self.todays_bars = {}

        self.logger.info(
            "Starting pre-market value area calculation using %s…",
            self._profiler.__class__.__name__,
        )
        t_start = perf_timer.perf_counter()

        symbols_processed = 0
        for sym, df in historical_data.items():
            if sym not in self.symbols:
                continue

            t_sym_start = perf_timer.perf_counter()

            if df.empty:
                self.logger.warning("%s | Historical data is empty, skipping.", sym)
                continue

            # --- Get RTH session times ---
            session_start, session_end = get_session_times('regular')

            # --- NEW: Pre-calculate historical cumulative volume profiles ---
            if self.use_rel_vol_filter:
                self.hist_cum_vol[sym] = {}
                for d, day_df in df.groupby(df.index.date):
                    rth_day_df = day_df.between_time(session_start, session_end)
                    if not rth_day_df.empty:
                        self.hist_cum_vol[sym][d] = rth_day_df['volume'].cumsum()

            # --- Session-Aware ATR Calculation ---
            atr_session_df = df.between_time(session_start, session_end)
            daily = atr_session_df.resample("1D").agg(
                {"high": "max", "low": "min", "close": "last"}
            ).dropna()
            atr_series = calculate_atr(daily, period=self.atr_period)
            atr_by_date = pd.Series(atr_series.values, index=atr_series.index.date)

            # --- Session-Aware Profile Calculation ---
            for d, day_df in df.groupby(df.index.date):
                rth_day_df = day_df.between_time(session_start, session_end)
                rth_day_df = rth_day_df.copy()
                rth_day_df["close"]  = pd.to_numeric(rth_day_df["close"],  errors="coerce")
                rth_day_df["volume"] = pd.to_numeric(rth_day_df["volume"], errors="coerce")
                rth_day_df = rth_day_df.dropna(subset=["close", "volume"])

                if rth_day_df.empty:
                    continue

                stats = self._profiler.calculate(rth_day_df)

                if not stats:
                    sess_range = rth_day_df['high'].max() - rth_day_df['low'].min()
                    if sess_range > 0:
                        mid = (rth_day_df['high'].max() + rth_day_df['low'].min()) / 2
                        stats = {'poc_price': mid, 'value_area_high': mid + sess_range * 0.25, 'value_area_low':  mid - sess_range * 0.25, 'shape': 'T'}
                    else:
                        continue

                self.value_areas[sym][d] = {
                    "VAH":   stats["value_area_high"],
                    "VAL":   stats["value_area_low"],
                    "POC":   stats["poc_price"],
                    "VWAP":  (rth_day_df["close"] * rth_day_df["volume"]).sum() / max(rth_day_df["volume"].sum(), 1),
                    "shape": stats.get("shape", "D").upper(),
                    "ATR":   float(atr_by_date.get(d, float("nan"))),
                    "y_close": rth_day_df["close"].iloc[-1],
                }

                if d == (df.index.date[-1] - timedelta(days=1)):
                    self.logger.debug(
                        "%s | stored levels for %s  shape=%s ATR=%.2f",
                        sym, d, self.value_areas[sym][d]['shape'], self.value_areas[sym][d]['ATR']
                    )

            t_sym_end = perf_timer.perf_counter()
            self.logger.info(
                "%s | Profile calculations complete in %.4f seconds", sym, t_sym_end - t_sym_start
            )
            symbols_processed += 1

        t_end = perf_timer.perf_counter()
        self.logger.info(
            "Pre-market calculations for %d symbols finished in %.2f seconds.",
            symbols_processed,
            t_end - t_start,
            )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _position_size(self, entry: float, stop: float) -> int:
        risk_per_share = max(entry - stop, self.tick_size)
        eq = self.ledger.get_total_equity(self.current_prices)
        by_risk = int((eq * self.risk_pct) / risk_per_share)
        by_liq  = int((eq * self.liquidity_cap_pct) / entry)
        return max(1, min(by_risk, by_liq))

    # ------------------------------------------------------------------ #
    # Intraday logic                                                     #
    # ------------------------------------------------------------------ #
    def on_bar(self, symbol: str, bar: pd.Series):
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
            return
        super().on_bar(symbol, bar)

        today = bar.name.date()
        now   = bar.name.time()

        # --- Efficiently build up the current day's bars for checks ---
        if symbol not in self.todays_bars or self.todays_bars[symbol].iloc[-1].name.date() != today:
            self.todays_bars[symbol] = pd.DataFrame([bar])
        else:
            self.todays_bars[symbol] = pd.concat([self.todays_bars[symbol], pd.DataFrame([bar])])

        # --- Efficient Intraday VWAP Calculation ---
        if symbol not in self.intraday_vwap_data or self.intraday_vwap_data[symbol]['date'] != today:
            self.intraday_vwap_data[symbol] = {'tp_vol': 0, 'vol': 0, 'date': today}

        typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
        self.intraday_vwap_data[symbol]['tp_vol'] += typical_price * bar['volume']
        self.intraday_vwap_data[symbol]['vol'] += bar['volume']

        current_vwap = self.intraday_vwap_data[symbol]['tp_vol'] / self.intraday_vwap_data[symbol]['vol'] if self.intraday_vwap_data[symbol]['vol'] > 0 else 0
        vwap_ok = (not self.use_vwap_filter) or (bar['close'] > current_vwap)

        # --- Relative Volume Check ---
        relative_volume_ok = True
        if self.use_rel_vol_filter:
            hist_profiles = self.hist_cum_vol.get(symbol, {})
            relative_volume_ok = has_high_relative_volume(
                todays_intraday=self.todays_bars[symbol],
                hist_cum_vol_profiles=hist_profiles,
                bar_ts=bar.name,
                ratio_threshold=self.rel_vol_ratio
            )

        # --- Yesterday's Levels & Filters ---
        yday = today - timedelta(days=1)
        lv = self.value_areas.get(symbol, {}).get(yday)
        if not lv or pd.isna(lv["ATR"]):
            return

        if lv["shape"] not in self.allowed_shapes:
            return

        if symbol in self.last_trade_time and bar.name - self.last_trade_time[symbol] < self.cooldown_period:
            return

        # --- FIX: Re-insert the missing Dip Detection logic ---
        # ---- dip detection & confirmation
        below_val = bar["close"] < lv["VAL"] - self.min_dip_ticks * self.tick_size
        if below_val and symbol not in self.dip_started:
            self.dip_started[symbol] = bar.name
            self.logger.info(
                "%s | %s | below VAL %.2f at %.2f", symbol, bar.name.time(), lv['VAL'], bar['close']
            )
        elif not below_val and symbol in self.dip_started:
            self.logger.info(
                "%s | %s | back above VAL (dip aborted)", symbol, bar.name.time()
            )
            self.dip_started.pop(symbol, None)

        confirm_ok = True
        if self.confirm_minutes > 0:
            confirm_ok = (
                    symbol in self.dip_started
                    and (bar.name - self.dip_started[symbol]).seconds >= self.confirm_minutes * 60
            )
        # --- End of Fix ---

        # ==============================================================
        # ENTRY
        # ==============================================================
        if (
                self.entry_start_time <= now <= self.entry_end_time
                and below_val
                and confirm_ok
                and vwap_ok
                and relative_volume_ok
                and symbol not in self.ledger.open_positions
        ):
            stop_price = bar["close"] - self.atr_stop_mult * lv["ATR"]

            if stop_price >= bar["close"]:
                return

            risk = bar["close"] - stop_price

            # --- FIX: Calculate target_price directly from min_rr ---
            # This ensures your config setting is used correctly.
            target_price = bar["close"] + (risk * self.min_rr)

            qty = self._position_size(bar["close"], stop_price)
            if qty == 0:
                return

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

    # ------------------------------------------------------------------
    def on_session_end(self):
        """No end-of-day actions yet."""
        pass