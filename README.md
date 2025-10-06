# Beyond Black-Scholes: An AI Framework for Options Mispricing

# Problem Statement:

Options pricing models like Black-Scholes assume efficient markets and rational pricing. In practice, options are frequently mispriced due to sentiment shocks, news events, and behavioral biases. Traditional quant tools ignore sentiment, while NLP-based models ignore theoretical benchmarks. This creates inefficiencies that traders struggle to systematically identify and exploit.


# Predictive Options Trading Strategy

[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)


---

## Abstract

This project implements a predictive options trading strategy integrating **option mispricing detection** and **financial news sentiment analysis**. By combining the Black-Scholes theoretical model with FinBERT sentiment scoring, the system generates trading signals and evaluates performance through a backtesting framework.

---
## Repository Structure

project-root/
│
├── src/
│   ├── black_scholes_calculator.py        # Black-Scholes pricing models
│   ├── mispricing_detector.py             # Detect mispricing (Script 2)
│   ├── sentiment_analysis.py              # FinBERT live sentiment analysis (Script 3)
│   ├── predictive_modeling.py             # Merge data & generate strategy signals (Script 4)
│   └── backtesting_framework.py           # Backtest strategy & cumulative returns (Script 5)
│
├── outputs/                               # Generated CSV outputs
│   ├── mispricing_results_YYYYMMDD_HHMMSS.csv
│   ├── sentiment_results_YYYYMMDD_HHMMSS.csv
│   └── merged_results_YYYYMMDD_HHMMSS.csv
│
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies

---

##  Introduction

Options trading provides opportunities to profit from price discrepancies between theoretical option values and actual market prices. However, identifying profitable trades requires combining quantitative valuation models with qualitative insights, such as market sentiment.  

This project develops a **predictive options trading framework** that integrates:  

1. **Option Mispricing Detection** using the Black-Scholes model.  
2. **Financial News Sentiment Analysis** using FinBERT.  
3. **Signal Generation and Backtesting** to evaluate strategy performance.

The goal is to identify mispriced options and generate trading signals informed by both quantitative metrics and market sentiment, then evaluate potential returns through backtesting.

---

##  Methodology

###  Data Collection

- **Option Data**: Current stock prices, strikes, option prices, implied volatility, and historical performance (3-month returns).  
- **Sentiment Data**: Live market headlines for selected tickers (TSLA, AAPL, NVDA, AMZN, JPM) scraped via the Finnhub API.  
- **Merged Dataset**: Mispricing results combined with sentiment ratios to create actionable trading signals.

###  Mispricing Detection

- The **Black-Scholes formula** is used to calculate theoretical option prices.  
- Mispricing is defined as the percentage deviation between market price and theoretical price:  

Mispricing % = \frac{Market\ Price - Theoretical\ Price}{Theoretical\ Price} \times 100


- Options are labeled as:  
  - `Overpriced`: Market price > theoretical price  
  - `Underpriced`: Market price < theoretical price  
  - `Fairly Price`: Market price ≈ theoretical price

###  Sentiment Analysis

- Headlines are analyzed with **FinBERT** to compute sentiment ratios:  
  - `positive_ratio`  
  - `negative_ratio`  
  - `neutral_ratio` (implicitly calculated if needed)

- Sentiment informs strategy weighting: tickers with higher positive sentiment are more likely to be assigned long positions, and vice versa.

###  Strategy Signal and Weighted Score

- Each ticker receives a **strategy signal** based on combined mispricing and sentiment data:  
  - `1` = Long (buy call)  
  - `-1` = Short (sell call)  
  - `0` = No position  

- **Weighted Score**: Combines the magnitude of mispricing with sentiment strength to produce a proportional signal for backtesting.

###  Backtesting

- **Cumulative Returns** are computed per ticker using the weighted strategy signals.  
- Cumulative return \(> 1\) indicates net gain; \(< 1\) indicates net loss over the test period.

---

## 3. Results

### 3.1 Sample Merged Dataset

| Ticker | Sector         | Current Price | Mispricing % | Status       | Positive Ratio | Negative Ratio | Strategy Signal | Weighted Score | Cumulative Return |
|--------|----------------|---------------|--------------|--------------|----------------|----------------|----------------|----------------|------------------|
| TSLA   | Automotive     | 429.83        | 2.48         | Fairly Price | 0.266          | 0.256          | 1              | 0.01005        | 1.01005          |
| AAPL   | Consumer       | 258.02        | -0.69        | Fairly Price | 0.197          | 0.169          | -1             | -0.02732       | 0.97268          |
| NVDA   | Semiconductor  | 187.62        | 3.24         | Overpriced   | 0.276          | 0.164          | 1              | 0.112          | 1.112            |
| AMZN   | E-Commerce     | 219.51        | 7.62         | Overpriced   | 0.252          | 0.171          | 1              | 0.08095        | 1.08095          |
| JPM    | Financials     | 310.03        | -1.17        | Fairly Price | 0.246          | 0.175          | -1             | -0.07018       | 0.92983          |

---

##  Interpretation

- Cumulative returns indicate **overall strategy performance**:  
  - TSLA, NVDA, and AMZN have cumulative returns > 1, indicating profitable positions over the test period.  
  - AAPL and JPM have cumulative returns < 1, indicating slight losses.  
- The mixed performance highlights that the strategy captures opportunities in mispriced options but is influenced by market volatility and sentiment signals.  
- Overall, a positive weighted cumulative return for the portfolio suggests that the model has predictive value but may require **risk management and diversification** for real trading.

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/options-trading-strategy.git
cd options-trading-strategy


