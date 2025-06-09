import os
import pandas as pd
from strategies.base import BaseStrategy
from backtest.results import calculate_performance_metrics, print_performance_report, plot_equity_curve
from analytics.indicators import calculate_vwap
from analytics.profiles import get_session, VolumeProfiler, MarketProfiler

class BacktestEngine:
    def __init__(self, data_dir: str, strategy: BaseStrategy):
        self.data_dir = data_dir
        self.strategy = strategy
        self.all_data = self._load_all_data()
        self.master_timeline = self._create_master_timeline()

    def _load_all_data(self) -> dict[str, pd.DataFrame]:
        """
        Loads all CSV files, standardizing all column names and handling dates.
        """
        data = {}
        print(f"Loading data from: {self.data_dir}")
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                symbol = os.path.splitext(filename)[0]
                filepath = os.path.join(self.data_dir, filename)
                try:
                    df = pd.read_csv(filepath)

                    df.rename(columns={
                        'date': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True, errors='ignore')

                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    df.set_index('Date', inplace=True)

                    data[symbol] = df
                except Exception as e:
                    print(f"  - Error loading {filename}: {e}")
        return data

    def _create_master_timeline(self) -> list:
        """
        Creates a master index of all unique trading days.
        """
        all_dates = set()
        for df in self.all_data.values():
            all_dates.update(df.index.date)
        return sorted(list(all_dates))

    def run(self, start_date, end_date):
        """
        Runs the backtest for a specific date range.
        """
        if not self.all_data:
            print("No data loaded. Exiting backtest.")
            return

        print(f"\n--- Starting Portfolio Backtest ---")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        backtest_timeline = [
            date for date in self.master_timeline if start_date <= date <= end_date
        ]

        if not backtest_timeline:
            print("No trading days found in the specified date range.")
            return

        try:
            first_day_master_index = self.master_timeline.index(backtest_timeline[0])
            if first_day_master_index == 0:
                print("Error: Backtest start date is the first day of the dataset. Cannot get previous day's data.")
                return
            prev_trade_date = self.master_timeline[first_day_master_index - 1]
        except (ValueError, IndexError):
            print(f"Error: Could not find previous day for backtest start date.")
            return

        for trade_date in backtest_timeline:
            date_str = trade_date.strftime('%Y-%m-%d')
            print(f"\n--- Simulating Day: {date_str} ---")

            # Settle funds from previous day's sales at the start of the new day.
            self.strategy.ledger.settle_funds()
            print(f"Start of Day Settled Cash: ${self.strategy.ledger.cash:,.2f}")

            historical_data = {
                symbol: df[df.index.date == prev_trade_date]
                for symbol, df in self.all_data.items()
            }
            candidate_tickers = self.strategy.scan_for_candidates(trade_date, historical_data)

            if not candidate_tickers:
                print("No candidates found for today.")
                prev_trade_date = trade_date
                continue

            print(f"Today's Candidates: {candidate_tickers}")

            todays_market_data = {
                symbol: df[df.index.date == trade_date].copy()
                for symbol, df in self.all_data.items() if symbol in candidate_tickers
            }

            todays_market_data = {k: v for k, v in todays_market_data.items() if not v.empty}
            if not todays_market_data:
                print("No market data available for any candidates today.")
                prev_trade_date = trade_date
                continue

            self.strategy.on_session_start(todays_market_data)

            # Create a master timeline of all unique timestamps for the day
            all_timestamps = set()
            for df in todays_market_data.values():
                all_timestamps.update(df.index)

            sorted_timestamps = sorted(list(all_timestamps))

            # Loop through each timestamp of the day in chronological order
            for ts in sorted_timestamps:
                current_bar_data = {}
                for symbol, df in todays_market_data.items():
                    if ts in df.index:
                        current_bar_data[symbol] = df.loc[ts]

                if not current_bar_data:
                    continue

                session_bars = {
                    symbol: df.loc[:ts] for symbol, df in todays_market_data.items()
                    if symbol in current_bar_data
                }

                first_symbol = next(iter(current_bar_data), None)
                if not first_symbol: continue
                current_session = get_session(current_bar_data[first_symbol].name.tz_convert(self.strategy.timezone))


                analytics = {}
                for symbol, bars in session_bars.items():
                    session_specific_bars = bars[bars.index.to_series().apply(lambda dt: get_session(dt.tz_convert(self.strategy.timezone)) == current_session)]
                    if not session_specific_bars.empty:
                        analytics[symbol] = {
                            'vwap': calculate_vwap(session_specific_bars.copy())
                        }

                market_prices = {
                    pos: current_bar_data.get(pos, {}).get('Close', self.strategy.ledger.open_positions[pos]['entry_price'])
                    for pos in self.strategy.ledger.open_positions
                }
                market_prices.update({
                    sym: bar['Close'] for sym, bar in current_bar_data.items()
                })
                market_prices = {k: v for k, v in market_prices.items() if v is not None}

                self.strategy.on_bar(current_bar_data, session_bars, market_prices, analytics)

            self.strategy.on_session_end()
            prev_trade_date = trade_date

        print("\n--- Backtest Complete ---")
        metrics = calculate_performance_metrics(self.strategy.ledger)
        print_performance_report(metrics)
        plot_equity_curve(self.strategy.ledger)
