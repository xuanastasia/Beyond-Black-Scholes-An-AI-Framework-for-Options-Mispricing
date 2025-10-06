# src/black_scholes_calculator.py
"""
Stock Options Black-Scholes Pricing Framework
Foundation for AI-driven mispricing and sentiment analysis
----------------------------------------------------------
This script benchmarks theoretical option prices and Greeks
for selected equities using the Black-Scholes model.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime


class StockOptionsBlackScholes:
    def __init__(self, risk_free_rate=0.05):
        """
        Initialize Black-Scholes model setup:
        - Define global parameters
        - Choose representative equities
        - Set risk-free rate and placeholders
        """
        self.r = risk_free_rate

        # Representative equities for analysis
        self.assets = {
            'TSLA': {
                'sector': 'Automotive / AI',
                'description': 'High sentiment-driven growth stock',
                'sentiment_profile': 'Highly reactive to news and social media'
            },
            'AAPL': {
                'sector': 'Consumer Tech',
                'description': 'Stable large-cap with balanced fundamentals',
                'sentiment_profile': 'Moderate sentiment sensitivity, heavy institutional flow'
            },
            'NVDA': {
                'sector': 'Semiconductors / AI',
                'description': 'Core AI momentum stock',
                'sentiment_profile': 'Driven by innovation cycles and AI narratives'
            },
            'AMZN': {
                'sector': 'E-Commerce / Cloud',
                'description': 'Diversified tech leader',
                'sentiment_profile': 'Moderate volatility; reacts to macro and earnings signals'
            },
            'JPM': {
                'sector': 'Financials',
                'description': 'Banking and investment services giant',
                'sentiment_profile': 'Lower sentiment bias, more fundamentals-driven'
            }
        }

        self.market_data = {}
        self.analysis_timestamp = datetime.now()

    def fetch_stock_data(self):
        """
        Fetch daily price data and calculate annualized volatility.
        """
        print("Fetching stock data for options pricing framework...\n")

        for ticker, info in self.assets.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo", interval="1d")

                current_price = hist['Close'].iloc[-1]
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                volatility = returns.std() * np.sqrt(252)

                # 3-month performance (approx. 63 trading days)
                price_3mo_ago = hist['Close'].iloc[-63] if len(hist) > 63 else hist['Close'].iloc[0]
                performance_3mo = (current_price - price_3mo_ago) / price_3mo_ago * 100

            except Exception as e:
                print(f"Error fetching {ticker}: {e}. Using fallback values.")
                fallback = self._get_fallback_data(ticker)
                current_price = fallback['price']
                volatility = fallback['vol']
                performance_3mo = fallback['perf']

            self.market_data[ticker] = {
                'price': current_price,
                'volatility': volatility,
                'performance_3mo': performance_3mo,
                'sector': info['sector'],
                'description': info['description'],
                'sentiment_profile': info['sentiment_profile']
            }

            print(f"{ticker}: ${current_price:.2f} | Vol: {volatility:.1%} | 3M Perf: {performance_3mo:+.1f}%")

        print("\nAll market data successfully loaded.\n")
        return self.market_data

    def _get_fallback_data(self, ticker):
        """
        Provide fallback values in case API fails or network unavailable.
        """
        fallback = {
            'TSLA': {'price': 260.0, 'vol': 0.55, 'perf': 10.0},
            'AAPL': {'price': 185.0, 'vol': 0.28, 'perf': 5.0},
            'NVDA': {'price': 450.0, 'vol': 0.48, 'perf': 15.0},
            'AMZN': {'price': 140.0, 'vol': 0.32, 'perf': 8.0},
            'JPM': {'price': 195.0, 'vol': 0.22, 'perf': 2.0}
        }
        return fallback.get(ticker, {'price': 100.0, 'vol': 0.30, 'perf': 0.0})

    def calculate_black_scholes(self, ticker, strike_price, time_to_expiry, option_type='call', dividend_yield=0):
        """
        Compute theoretical option price using the Black-Scholes model.
        """
        data = self.market_data.get(ticker)
        if not data:
            raise ValueError(f"No data found for {ticker}. Run fetch_stock_data() first.")

        S = data['price']
        K = strike_price
        T = time_to_expiry
        r = self.r
        sigma = data['volatility']
        q = dividend_yield

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return max(price, 0)

    def calculate_greeks(self, ticker, strike_price, time_to_expiry, option_type='call', dividend_yield=0):
        """
        Compute the five major Greeks for risk/sensitivity analysis.
        """
        data = self.market_data[ticker]
        S = data['price']
        K = strike_price
        T = time_to_expiry
        r = self.r
        sigma = data['volatility']
        q = dividend_yield

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        # Gamma and Vega
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01  # per 1% change in vol

        # Theta and Rho
        if option_type.lower() == 'call':
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        else:
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def comparative_analysis(self, strike_percentage=1.0, time_to_expiry=30 / 365, option_type='call'):
        """
        Compare option prices and Greeks for all selected stocks.
        """
        print(f"\nComparative Analysis: {option_type.upper()} options | {strike_percentage * 100:.0f}% strike | {time_to_expiry * 365:.0f} days to expiry")
        print("=" * 75)

        results = []
        for ticker in self.assets.keys():
            strike = self.market_data[ticker]['price'] * strike_percentage
            price = self.calculate_black_scholes(ticker, strike, time_to_expiry, option_type)
            greeks = self.calculate_greeks(ticker, strike, time_to_expiry, option_type)

            result = {
                'ticker': ticker,
                'sector': self.market_data[ticker]['sector'],
                'current_price': self.market_data[ticker]['price'],
                'strike': strike,
                'option_price': price,
                'volatility': self.market_data[ticker]['volatility'],
                'perf_3m(%)': self.market_data[ticker]['performance_3mo'],
                **greeks
            }
            results.append(result)

            print(f"{ticker}: Option ${price:.2f} | Delta {greeks['delta']:.3f} | Vega {greeks['vega']:.3f} | Vol {self.market_data[ticker]['volatility']:.1%}")

        return pd.DataFrame(results)

    def insights(self, df):
        """
        Generate basic sector-level volatility and sensitivity insights.
        """
        print("\nPreliminary Insights:")
        print("=" * 50)

        avg_vol = df.groupby("sector")["volatility"].mean().sort_values(ascending=False)
        print("* Average Volatility by Sector:")
        print(avg_vol, "\n")

        # Safely handle missing columns
        if "vega" in df.columns:
            high_sensitivity = df.sort_values("vega", ascending=False).head(3)
            print("* Top 3 Stocks Most Sensitive to Volatility (Vega):")
            print(high_sensitivity[["ticker", "vega", "volatility", "perf_3m(%)"]])
        else:
            print("No 'vega' column found in DataFrame. Check calculation consistency.")


def main():
    print("=" * 50)
    print("Stock Options Black-Scholes Model")
    print("   Foundation for Mispricing & Sentiment Analysis")
    print("=" * 50)

    model = StockOptionsBlackScholes(risk_free_rate=0.05)
    model.fetch_stock_data()
    df = model.comparative_analysis(option_type='call')
    model.insights(df)

    print(f"\nAnalysis Completed: {model.analysis_timestamp}")


if __name__ == "__main__":
    main()



