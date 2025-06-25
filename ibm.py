import pandas as pd
import yaml

# Load your config
cfg = yaml.safe_load(open("config.yaml"))
bt = cfg["backtest"]
print("lookback_days:", bt["lookback_days"])
print("poc_lookback_days:", bt["poc_lookback_days"])
print("atr_period:", bt["atr_period"])
print("computed lookback:", max(bt["lookback_days"], bt["poc_lookback_days"], bt["atr_period"]) + 1)
print()

# Peek at one symbol’s data
df = pd.read_feather("data/feather/IBM.feather")
print("IBM.feather unique dates:", df["date"].nunique())
print(df.head())