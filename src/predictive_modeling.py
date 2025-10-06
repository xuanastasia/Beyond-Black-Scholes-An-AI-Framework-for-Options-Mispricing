# src/predictive_modeling.py
"""
Script 4: Predictive Modeling
-----------------------------
This script merges mispricing results with sentiment analysis and prepares
data for further ML or quantitative analysis.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------
# HELPER: get latest file with prefix
# ---------------------------
def get_latest_file(prefix):
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No files starting with '{prefix}' found in {OUTPUT_DIR}/")
    latest_file = max(files, key=lambda f: os.path.getmtime(OUTPUT_DIR / f))
    return OUTPUT_DIR / latest_file

# ---------------------------
# LOAD DATA
# ---------------------------
# Mispricing
mispricing_path = get_latest_file("mispricing_results")
mispricing_df = pd.read_csv(mispricing_path)
print(f"Loaded mispricing file: {mispricing_path}")

# Sentiment (raw)
sentiment_path = get_latest_file("sentiment_results")
sentiment_df = pd.read_csv(sentiment_path)
print(f"Loaded sentiment file: {sentiment_path}")

# ---------------------------
# PROCESS SENTIMENT
# ---------------------------
# Compute positive/negative ratios per ticker
summary = sentiment_df.groupby("ticker")["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
summary["positive_ratio"] = summary.get("positive", 0)
summary["negative_ratio"] = summary.get("negative", 0)
summary = summary.reset_index()

# ---------------------------
# MERGE DATA
# ---------------------------
merged_df = pd.merge(
    mispricing_df,
    summary[["ticker", "positive_ratio", "negative_ratio"]],
    on="ticker",
    how="left"
)

# ---------------------------
# QUICK CHECK / SUMMARY
# ---------------------------
print("\nMerged Data Sample:")
print(merged_df.head())

# Example: top mispriced options with sentiment
top_overpriced = merged_df.sort_values("mispricing_pct", ascending=False).head(5)
print("\nTop 5 Overpriced Options with Sentiment:")
print(top_overpriced[["ticker", "option_price", "market_price", "mispricing_pct", "status", "positive_ratio", "negative_ratio"]])

# ---------------------------
# SAVE MERGED RESULTS
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"merged_results_{timestamp}.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nMerged results saved to: {output_path}")
