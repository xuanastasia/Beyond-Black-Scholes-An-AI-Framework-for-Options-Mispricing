"""
Script 3: Live Financial News Sentiment Analysis using FinBERT
-------------------------------------------------------------
This script pulls *live market headlines* for selected tickers
from the Finnhub API, applies FinBERT sentiment analysis, and
produces an aggregated sentiment dashboard for decision support.
"""

import requests
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta
from pathlib import Path

# ========== CONFIGURATION ==========
API_KEY = "d3h79upr01qstnq80g9gd3h79upr01qstnq80ga0"   # Replace with your real key
TICKERS = ["TSLA", "AAPL", "NVDA", "AMZN", "JPM"]
DAYS_BACK = 7   # Lookback window for news (in days)

# ---------------------------
# Set OUTPUT_DIR to project root outputs folder
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== SETUP ==========
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ========== STEP 1: FETCH LIVE HEADLINES ==========
def fetch_finnhub_news(ticker, days_back=DAYS_BACK):
    """Fetch company news from Finnhub for the last N days."""
    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=days_back)
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"[Error] {ticker}: Failed to fetch news (status {response.status_code})")
        return []

    articles = response.json()
    news_list = []
    for article in articles:
        headline = article.get("headline", "")
        if headline:
            news_list.append({
                "ticker": ticker,
                "headline": headline,
                "datetime": article.get("datetime", ""),
                "source": article.get("source", ""),
                "url": article.get("url", "")
            })
    return news_list

# ========== STEP 2: AGGREGATE NEWS ==========
def collect_all_news(tickers):
    all_news = []
    for ticker in tickers:
        print(f"Fetching live headlines for {ticker}...")
        ticker_news = fetch_finnhub_news(ticker)
        all_news.extend(ticker_news)
    return pd.DataFrame(all_news)

# ========== STEP 3: APPLY FINBERT SENTIMENT ==========
def analyze_sentiment(df):
    """Apply FinBERT sentiment model to each headline."""
    print("\nRunning FinBERT sentiment analysis...")
    sentiments = []
    for text in df["headline"]:
        result = sentiment_analyzer(text)[0]
        sentiments.append(result["label"])
    df["sentiment"] = sentiments
    return df

# ========== STEP 4: SUMMARIZE RESULTS ==========
def summarize_sentiment(df):
    """Compute average sentiment distribution per stock."""
    summary = df.groupby(["ticker", "sentiment"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    for col in ["positive", "neutral", "negative"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["positive_ratio"] = summary["positive"] / summary["total"]
    summary["negative_ratio"] = summary["negative"] / summary["total"]
    summary = summary.sort_values("positive_ratio", ascending=False)
    return summary

# ========== STEP 5: MAIN PIPELINE ==========
def main():
    print("==================================================")
    print(" Live Sentiment Analysis using FinBERT")
    print("==================================================\n")

    news_df = collect_all_news(TICKERS)

    if news_df.empty:
        print("No news fetched. Please check API key or network connection.")
        return

    analyzed_df = analyze_sentiment(news_df)
    summary = summarize_sentiment(analyzed_df)

    print("\nSentiment Summary (Last 7 Days):")
    print("==================================================")
    print(summary[["positive_ratio", "negative_ratio"]].to_string(float_format="%.2f"))

    # Save output in project root outputs folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"sentiment_results_{timestamp}.csv"
    analyzed_df.to_csv(output_path, index=False)
    print(f"\nDetailed sentiment results saved to: {output_path}")

if __name__ == "__main__":
    main()
