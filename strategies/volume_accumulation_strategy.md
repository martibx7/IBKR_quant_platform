````markdown
# Volume-Accumulation POC-Retest Strategy

## 1. Strategy Overview
A rule-based, multi-day consolidation breakout strategy that:
1. Identifies stocks trading in a tight range for **N** days.
2. Requires a **price breakout** above the consolidation high.
3. Confirms the breakout with a **volume surge** (today’s cumulative volume ≥ X × average consolidation volume).
4. Computes the **Point-of-Control (POC)** of the base volume profile.
5. Waits for price to **re-test the POC** before entering.
6. Manages risk with fixed **stop**, **profit target**, and **breakeven** rules.

---

## 2. Key Parameters

| Parameter                  | Description                                                                 | Default    |
|----------------------------|-----------------------------------------------------------------------------|------------|
| `consolidation_days`       | Number of prior trading days to form the base                                | 10         |
| `consolidation_range_pct`  | Maximum allowed (High – Low) / Low over base window                          | 7%         |
| `breakout_volume_ratio`    | Multiplier of average base volume required for today’s volume surge          | 1.5×       |
| `min_price`, `max_price`   | Price filter to avoid penny or ultra-expensive stocks                        | $2–$100    |
| `risk_per_trade_pct`       | Percent of total equity risked per trade                                     | 1%         |
| `profit_target_r`          | Reward-to-risk ratio for profit target                                       | 1.5R       |
| `breakeven_trigger_r`      | R multiple at which stop moves to breakeven                                  | 0.75R      |
| `eod_flat_time`            | Time to exit all positions to avoid overnight                                | 15:50      |

---

## 3. Screening & Base Formation

1. **Fetch history** for each candidate symbol from:
   `trade_date – (consolidation_days + buffer)` … `trade_date`
2. **Price filter**: require last close ∈ `[min_price, max_price]`.
3. **Data sufficiency**: ensure ≥ `consolidation_days` of prior bars exist.
4. **Tight range check**:
   ```text
   (max High – min Low) / min Low  ≤ consolidation_range_pct
````

5. Only symbols passing all above form the **consolidation base**.

---

## 4. Breakout Detection

1. **Price-breakout**

   * Compute:

     ```python
     consolidation_high = consolidation_data['High'].max()
     today_close       = last_day_data.iloc[-1]['Close']
     ```
   * Require:  `today_close > consolidation_high`.

2. **Volume-confirmation**

   * Calculate:

     ```python
     avg_base_vol = consolidation_data['Volume'].sum() / consolidation_days
     today_vol    = last_day_data['Volume'].sum()
     ```
   * Require:

     ```text
     today_vol ≥ breakout_volume_ratio × avg_base_vol
     ```

3. If both pass, mark symbol as a **breakout candidate**.

---

## 5. Point-of-Control (POC) & Retest

1. **Compute POC** over base bars using a volume-profile routine:

   * Identify the price level with the highest total traded volume.
2. **Watch for POC retest**:

   * When an intraday bar’s `Low ≤ POC`, record that bar’s `High` as `confirmation_high`.
   * Change state from **“watching”** → **“triggered”**.

---

## 6. Entry Execution

Once in **triggered** state, when a later bar’s `High > confirmation_high`:

1. **Entry price** = `confirmation_high`
2. **Stop-loss** = `consolidation_low` (min Low of base)
3. **Profit target** = `entry_price + R × profit_target_r`
4. **Position sizing**:

   ```python
   risk_per_share  = entry_price - stop_loss
   max_dollar_risk = equity * risk_per_trade_pct
   qty             = floor(max_dollar_risk / risk_per_share)
   ```
5. **Submit buy** and add to `active_trades`.

---

## 7. Trade Management & Exit Rules

For each `active_trade`, on each new bar:

1. **Breakeven move**: once `price ≥ entry_price + breakeven_trigger_r × R`, move stop to `entry_price`.
2. **Exit checks** (priority order):

   * **Stop-loss hit**: `bar.Low ≤ stop_loss` → exit at market.
   * **Profit target hit**: `bar.High ≥ profit_target` → exit at market.
   * **EOD flat**: `time ≥ eod_flat_time` → exit any remaining.
3. **Log exit** reason, P/L, and remove from active list.

---

## 8. Risk Controls & Rationale

* **Fixed risk per trade** keeps drawdowns predictable.
* **POC-retest entries** offer lower-risk setups vs. chasing highs.
* **Volume confirmation** filters out low-participation breakouts.
* **Timed exit** prevents overnight or weekend gap risk.

---

## 9. Implementation Tips

* Run `scan_for_candidates()` once **after market close**, or intra-day at a set time (e.g. 11 AM) but be consistent.
* Use minute bars (or higher resolution) for accurate retest detection.
* Maintain a **debug log** of filter pass/fail for continuous refinement.
* Backtest across varied symbols/sectors to validate parameter robustness.
  \`\`\`
