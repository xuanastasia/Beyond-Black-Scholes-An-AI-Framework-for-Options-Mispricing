# src/backtesting_framework.py
"""
Backtesting Framework: Cumulative Return Analysis
-------------------------------------------------
This script calculates cumulative returns for each ticker based on:
 - Mispricing signals from options (mispricing_pct)
 - Market sentiment from FinBERT (positive_ratio - negative_ratio)
It produces a CSV in outputs/ with cumulative return per ticker.
"""

import os
import pandas as pd
from datetime import datetime

# -------------------------
# 1) CONFIG
# -------------------------
MERGED_CSV = ""  # <-- You can leave blank to automatically pick the latest merged CSV
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# 2) HELPER: Find latest merged CSV
# -------------------------
def get_latest_merged_csv(prefix="merged_results"):
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No files starting with '{prefix}' found in outputs/")
    files.sort(reverse=True)
    return os.path.join(OUTPUT_DIR, files[0])

if not MERGED_CSV:
    MERGED_CSV = get_latest_merged_csv()

print(f"Loading merged file: {MERGED_CSV}")

# -------------------------
# 3) LOAD MERGED DATA
# -------------------------
df = pd.read_csv(MERGED_CSV)
print(f"Loaded {len(df)} rows.")

# -------------------------
# 4) CALCULATE STRATEGY SIGNALS
# -------------------------
# Mispricing influence: positive if underpriced, negative if overpriced
def mispricing_signal(row):
    if row['mispricing_pct'] > 0:
        return 1  # underpriced → buy
    elif row['mispricing_pct'] < 0:
        return -1  # overpriced → sell
    else:
        return 0  # fairly priced → hold

df['strategy_signal'] = df.apply(mispricing_signal, axis=1)

# Weight by sentiment (positive_ratio - negative_ratio)
df['weighted_signal'] = df['strategy_signal'] * (df['positive_ratio'] - df['negative_ratio'])

# -------------------------
# 5) BACKTEST CUMULATIVE RETURNS
# -------------------------
# Assume initial value = 1 per ticker
df['cumulative_return'] = 1 + df['weighted_signal']  # simplified: each signal adds/subtracts proportionally

# Ensure no negative cumulative returns for simplicity
df['cumulative_return'] = df['cumulative_return'].clip(lower=0)

# -------------------------
# 6) SAVE RESULTS
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"backtest_cumulative_{timestamp}.csv")
df.to_csv(output_file, index=False)

print("\nBacktest completed. Sample results:")
print(df[['ticker', 'mispricing_pct', 'positive_ratio', 'negative_ratio', 'strategy_signal', 'weighted_signal', 'cumulative_return']])
print(f"\nSaved backtest results to: {output_file}")

