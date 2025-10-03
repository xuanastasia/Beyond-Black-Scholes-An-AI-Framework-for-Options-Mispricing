# Beyond Black-Scholes: An AI Framework for Options Mispricing
ML framework combining Black-Scholes pricing and sentiment analysis to detect and predict options mispricing

:

# Problem Statement:

Options pricing models like Black-Scholes assume efficient markets and rational pricing. In practice, options are frequently mispriced due to sentiment shocks, news events, and behavioral biases. Traditional quant tools ignore sentiment, while NLP-based models ignore theoretical benchmarks. This creates inefficiencies that traders struggle to systematically identify and exploit.

# Solution
This project builds a machine learning framework that combines:
1. Theoretical pricing (Black-Scholes & Greeks) to benchmark “fair value”
2. Market sentiment analysis (FinBERT) to capture sentiment-driven deviations
3. Predictive modeling to forecast short-term (3-day) option price direction
4. Backtesting to validate performance and trading edge
5. The result is a tool that detects mispricing patterns caused by sentiment and predicts whether they will correct, providing a systematic edge over traditional models.

Key Features

- Black-Scholes & Greeks calculator for theoretical benchmarking
- Real-time mispricing detection against live market data
- News sentiment analysis using FinBERT (finance-specific BERT model)
- ML model to predict 3-day price direction using sentiment + mispricing features
- Backtesting framework with performance metrics (accuracy, Sharpe ratio, drawdowns, win rate)

Applications
- Quantitative finance research
- Options trading strategy development
- AI + NLP in financial markets

