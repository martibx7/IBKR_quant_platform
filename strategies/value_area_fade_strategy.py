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
from analytics.profiles import VolumeProfiler, MarketProfiler
from analytics.indicators import calculate_atr


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
        self.logger.info(
            "Starting pre-market value area calculation using %s…",
            self._profiler.__class__.__name__,
        )
        t_start = perf_timer.perf_counter()

        symbols_processed = 0
        for sym, df in historical_data.items():
            if self.symbol_whitelist and sym not in self.symbol_whitelist:
                continue

            t_sym_start = perf_timer.perf_counter()

            if df.empty:
                self.logger.warning("%s | Historical data is empty, skipping.", sym)
                continue

            daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
            atr_series = calculate_atr(daily, period=5)
            atr_by_date = pd.Series(atr_series.values, index=atr_series.index.date)

            for d, day_df in df.groupby(df.index.date):
                day_df["close"]  = pd.to_numeric(day_df["close"],  errors="coerce")
                day_df["volume"] = pd.to_numeric(day_df["volume"], errors="coerce")
                day_df = day_df.dropna(subset=["close", "volume"])
                if day_df.empty:
                    continue

                stats = self._profiler.calculate(day_df)
                if not stats:
                    self.logger.warning("%s | Profile calculation failed for date %s", sym, d)
                    # rescue: crude VAL/VAH = VWAP ± 0.5×(session range)
                    sess_range = day_df['high'].max() - day_df['low'].min()
                    if sess_range > 0:
                        mid = (day_df['high'].max() + day_df['low'].min()) / 2
                        stats = {
                            'poc_price': mid,
                            'value_area_high': mid + sess_range * 0.25,
                            'value_area_low':  mid - sess_range * 0.25,
                            'shape': 'T',
                        }
                    else:
                        continue

                self.value_areas[sym][d] = {
                    "VAH":   stats["value_area_high"],
                    "VAL":   stats["value_area_low"],
                    "POC":   stats["poc_price"],
                    "VWAP":  (day_df["close"] * day_df["volume"]).sum()
                             / max(day_df["volume"].sum(), 1),
                    "shape": stats.get("shape", "D").upper(),
                    "ATR":   float(atr_by_date.get(d, float("nan"))),
                    "y_close": day_df["close"].iloc[-1],
                }
                # ---------------- DEBUG: confirm we stored yesterday's levels ----------------
                if d == (df.index.date[-1] - timedelta(days=1)):   # only print for yesterday
                    # FIXED: Changed undefined variables 'shape' and 'atr_val' to their correct dictionary accessors
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

        # ---- skip-day enforcement
        if self.skip_day.get(symbol) == today:
            return

        # ---- gap check (once at open)
        if now == self.session_open_time and symbol not in self.skip_day:
            yday = today - timedelta(days=1)
            lv = self.value_areas.get(symbol, {}).get(yday)
            if lv and pd.notna(lv["ATR"]):
                if abs(bar["open"] - lv["y_close"]) / lv["ATR"] >= self.gap_skip_thresh:
                    self.skip_day[symbol] = today
                    self.logger.info("%s | REJECT: Opening gap is >= %.1f ATRs. Skipping day.", symbol, self.gap_skip_thresh)
                    return

        # ---- yesterday levels
        yday = today - timedelta(days=1)
        lv = self.value_areas.get(symbol, {}).get(yday)
        if not lv or pd.isna(lv["ATR"]):
            return # Cannot trade without yesterday's levels, this is a fundamental requirement.

        # DEBUG: show yesterday’s stats once per symbol per day
        if symbol not in getattr(self, "_dbg_printed", set()):
            self._dbg_printed = getattr(self, "_dbg_printed", set())
            self.logger.info(
                "%s | %s | VAL=%.2f VAH=%.2f POC=%.2f ATR=%.2f shape=%s",
                symbol, yday, lv['VAL'], lv['VAH'], lv['POC'], lv['ATR'], lv['shape']
            )
            self._dbg_printed.add(symbol)

        # BOTTLENECK 1: Is yesterday's profile shape allowed?
        if lv["shape"] not in self.allowed_shapes:
            # self.logger.info("%s | REJECT: Previous day shape '%s' not in allowed shapes %s", symbol, lv["shape"], self.allowed_shapes)
            return

        # BOTTLENECK 2: Is the symbol in a post-trade cooldown period?
        if symbol in self.last_trade_time and bar.name - self.last_trade_time[symbol] < self.cooldown_period:
            # self.logger.info("%s | REJECT: In cooldown period.", symbol)
            return

        # ---- dip detection & confirmation
        below_val = bar["close"] < lv["VAL"] - self.min_dip_ticks * self.tick_size
        if below_val and symbol not in self.dip_started:
            self.dip_started[symbol] = bar.name
            # DEBUG: first tick below VAL
            self.logger.info(
                "%s | %s | below VAL %.2f at %.2f", symbol, bar.name.time(), lv['VAL'], bar['close']
            )
        elif not below_val and symbol in self.dip_started:
            # DEBUG: dip aborted
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

        # ==============================================================
        # ENTRY
        # ==============================================================
        if (
                self.entry_start_time <= now <= self.entry_end_time
                and below_val
                and confirm_ok
                and symbol not in self.ledger.open_positions
        ):
            stop_price = bar["close"] - self.atr_stop_mult * lv["ATR"]

            # BOTTLENECK 3: Is the calculated stop price valid?
            if stop_price >= bar["close"]:
                self.logger.info(
                    "%s | %s | STOP_TOO_TIGHT close=%.2f stop=%.2f ATR=%.2f mult=%.2f",
                    symbol, bar.name.time(), bar['close'], stop_price, lv['ATR'], self.atr_stop_mult
                )
                return

            risk = bar["close"] - stop_price
            target_price = max(lv["POC"], bar["close"] + risk * 1.5)
            rr = (target_price - bar["close"]) / risk

            # BOTTLENECK 4: Is the Reward/Risk ratio sufficient?
            if rr < self.min_rr:
                self.logger.info(
                    "%s | %s | RR_TOO_LOW rr=%.2f target=%.2f stop=%.2f",
                    symbol, bar.name.time(), rr, target_price, stop_price
                )
                return

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