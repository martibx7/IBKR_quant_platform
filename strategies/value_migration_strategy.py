import pandas as pd
from strategies.base import BaseStrategy
from analytics.profiles import VolumeProfiler, MarketProfiler

class ValueMigrationStrategy(BaseStrategy):
    """
    Implements the Value Migration & Triple Confirmation Strategy.
    """
    def __init__(self, symbols: list[str], ledger, **kwargs):
        super().__init__(symbols, ledger, **kwargs)
        # --- Strategy Parameters ---
        self.risk_per_trade_pct = kwargs.get('risk_per_trade_pct', 0.01) # Risk 1% of equity

        # --- Strategy State ---
        self.prev_day_stats = {}
        self.active_trades = {}
        self.disqualified_today = set()

    def on_session_start(self, session_data: dict[str, pd.DataFrame]):
        # Reset state at the start of each new day
        self.active_trades = {}
        self.disqualified_today = set()

        # --- Stage 2: The Setup (Check Open vs Prior VAH) ---
        for symbol, df in session_data.items():
            if symbol in self.prev_day_stats:
                open_price = df.iloc[0]['Open']
                prev_vah = self.prev_day_stats[symbol]['vah']
                if open_price <= prev_vah:
                    # Disqualify the stock for the day if it opens below yesterday's value
                    self.disqualified_today.add(symbol)
                    print(f"[{df.index[0].date()}] {symbol} disqualified. Open ({open_price:.2f}) was not above prior VAH ({prev_vah:.2f}).")

    def scan_for_candidates(self, current_date: pd.Timestamp, historical_data: dict) -> list[str]:
        """
        Stage 1: The Scanner. Finds stocks that closed strong on the prior day.
        """
        candidates = []
        self.prev_day_stats = {}

        for symbol, df in historical_data.items():
            if df.empty:
                continue

            # Calculate Previous Day's Volume Profile
            profiler = VolumeProfiler(df, tick_size=0.01)
            if profiler.poc_price:
                self.prev_day_stats[symbol] = {'vah': profiler.vah}

                # Scanning Logic: Add stocks that closed in the top 25% of their daily range
                day_high = df['High'].max()
                day_low = df['Low'].min()
                day_close = df.iloc[-1]['Close']
                if day_high > day_low and day_close > (day_high - 0.25 * (day_high - day_low)):
                    candidates.append(symbol)

        return candidates

    def on_bar(self, current_bar_data: dict, session_bars: dict, market_prices: dict):
        """
        Stage 3 & 4: Manages entry triggers, exits, and risk.
        """
        for symbol, current_bar in current_bar_data.items():
            # ---- EXIT LOGIC ----
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                # Check stop-loss
                if current_bar['Close'] <= trade['stop_loss']:
                    print(f"[{current_bar.name}] STOP LOSS for {symbol} at {current_bar['Close']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], current_bar['Close'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                # Check for end-of-day exit (assuming last bar is EOD)
                elif len(session_bars[symbol]) == 390: # 390 bars in a standard session
                    print(f"[{current_bar.name}] End-of-day exit for {symbol} at {current_bar['Close']:.2f}")
                    self.ledger.record_trade(current_bar.name, symbol, trade['quantity'], current_bar['Close'], 'SELL', market_prices)
                    del self.active_trades[symbol]
                continue

            # ---- ENTRY LOGIC ----
            # Skip if disqualified, already in a trade, or not a valid candidate
            if symbol in self.disqualified_today or symbol in self.active_trades or symbol not in self.prev_day_stats:
                continue

            # --- Stage 3: The Triple Confirmation Trigger ---
            current_price = current_bar['Close']
            vwap = current_bar.get('VWAP')
            if vwap is None: continue

            # Condition 1: Pullback to VWAP
            pullback_to_vwap = current_bar['Low'] <= vwap and current_price > vwap

            if pullback_to_vwap:
                # Condition 2: Check Intraday TPO POC
                intraday_tpo_profiler = MarketProfiler(session_bars[symbol], tick_size=0.01)
                intraday_tpo_poc = intraday_tpo_profiler.poc_price
                if intraday_tpo_poc is None or current_price < intraday_tpo_poc:
                    continue # Price is not above the developing TPO POC

                # --- All conditions met, prepare and execute trade ---
                stop_loss_price = current_bar['Low']
                risk_per_share = current_price - stop_loss_price
                if risk_per_share <= 0: continue

                # Position Sizing
                risk_amount = self.ledger.get_total_equity(market_prices) * self.risk_per_trade_pct
                quantity = int(risk_amount / risk_per_share)
                if quantity == 0: continue

                # Execute
                print(f"[{current_bar.name}] ENTERING LONG {symbol} at {current_price:.2f}")
                self.ledger.record_trade(current_bar.name, symbol, quantity, current_price, 'BUY', market_prices)

                # Store active trade details
                self.active_trades[symbol] = {
                    'quantity': quantity,
                    'stop_loss': stop_loss_price
                }