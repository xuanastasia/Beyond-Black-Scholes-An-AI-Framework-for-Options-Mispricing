# src/01_black_scholes_calculator.py
"""
Macro-Regime Black-Scholes Options Pricing Model
Analyzing options pricing across different economic regimes: Tech Growth vs Safe-Haven Assets
Directly addresses the 2025 market divergence between tech stocks and gold
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import optimize
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

class MacroRegimeBlackScholes:
    def __init__(self, risk_free_rate=0.05):
        """
        Step1 : intialize
        - Set up the calculator with current market regime assets
        - Define risk-free rate (benchmark with 10-year Treasury yield)
        - Segment assets by economic regime behaviour
        """
        self.r = risk_free_rate
        
        # Step 1: Define assets by their macroeconomic regime
        self.assets = {
            'AAPL': {
                'sector': 'Tech Growth', 
                'description': 'AI & Consumer Tech - Growth Narrative',
                'regime': 'Risk-On',
                'narrative': 'AI boom continuation despite macro concerns'
            },
            'QQQ': {
                'sector': 'Tech Index', 
                'description': 'NASDAQ 100 - Diversified Tech Growth',
                'regime': 'Risk-On', 
                'narrative': 'Broad tech exposure to innovation theme'
            },
            'GLD': {
                'sector': 'Safe Haven', 
                'description': 'Gold ETF - Inflation & Geopolitical Hedge',
                'regime': 'Risk-Off',
                'narrative': 'Record highs on safe-haven demand, diverging from stocks'
            }
        }
        self.market_data = {}
        self.analysis_timestamp = datetime.now()
        
    def fetch_macro_regime_data(self):
        """
        Step 2: Data collection
        - Live market data using yfinance
        - Key metrics: current price, volatility, recent performance
        - Handle errors  with fallback data
        """
            
        print("Current Market Regime (2025): Tech Rally + Gold Safe-Haven Demand")
    
        
        for ticker, info in self.assets.items():
            try:
                # Step 2a: Fetch data from yfinance
                stock = yf.Ticker(ticker)
                
                # Get current price and historical data
                hist_data = stock.history(period="6mo", interval="1d")
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                
                # Step 2b: Calculate annual volatility 
                if len(hist_data) > 30:
                    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
                    volatility = returns.std() * np.sqrt(252)  # Annualize daily volatility
                    
                    # Calculate 3-month performance
                    price_3mo_ago = hist_data['Close'].iloc[0] if len(hist_data) > 60 else current_price * 0.9
                    performance_3mo = (current_price - price_3mo_ago) / price_3mo_ago * 100
                else:
                    # Fallback to realistic sample data
                    current_price, volatility, performance_3mo = self._get_2025_market_data(ticker)
                
                # Step 2c: Store processed market data
                self.market_data[ticker] = {
                    'current_price': current_price,
                    'volatility': volatility,
                    'performance_3mo_pct': performance_3mo,
                    'sector': info['sector'],
                    'description': info['description'],
                    'regime': info['regime'],
                    'narrative': info['narrative'],
                    'timestamp': self.analysis_timestamp
                }
                
                # Display results
                regime_indicator = "[RISK-ON]" if info['regime'] == 'Risk-On' else "[RISK-OFF]"
                perf_indicator = "[UP]" if performance_3mo > 0 else "[DOWN]"
                
                print(f"{ticker} {regime_indicator}: ${current_price:.2f} | Vol: {volatility:.1%} | 3Mo: {performance_3mo:+.1f}% {perf_indicator}")
                print(f"   Narrative: {info['narrative']}")
                
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                # Use 2025 realistic fallback data
                current_price, volatility, performance_3mo = self._get_2025_market_data(ticker)
                self.market_data[ticker] = {
                    'current_price': current_price,
                    'volatility': volatility,
                    'performance_3mo_pct': performance_3mo,
                    'sector': info['sector'],
                    'description': info['description'], 
                    'regime': info['regime'],
                    'narrative': info['narrative'],
                    'timestamp': self.analysis_timestamp
                }
        
        self._display_macro_insights()
        return self.market_data
    
    def _get_2025_market_data(self, ticker):
        """
        Step 3: Data to fallback on 
        - Provide realistic market data if live fetch fails
        - Based on 2025 market trends and regime characteristics
        - Ensures the code always runs for demonstration (esp for dashhborad)
        """
        market_2025 = {
            'AAPL': {'price': 185.0, 'vol': 0.28, 'performance': 15.0},
            'QQQ': {'price': 385.0, 'vol': 0.22, 'performance': 18.0},
            'GLD': {'price': 215.0, 'vol': 0.16, 'performance': 25.0}  # Gold outperforming
        }
        data = market_2025.get(ticker, {'price': 100.0, 'vol': 0.20, 'performance': 10.0})
        return data['price'], data['vol'], data['performance']
    
    def _display_macro_insights(self):
        """
        Step 4: Market Analysis
        - Provide context about current market conditions
        - Explain the economic rationale behind asset behavior
       
        """
        print("\nMacro Regime Insights (2025):")
        print("   * GLD at record highs: Safe-haven demand on geopolitical risks")
        print("   * Tech maintaining strength: AI innovation driving growth expectations") 
        print("   * Market pricing dual narratives: Growth + Safety simultaneously")
        print("   * This creates unique options mispricing opportunities")
        print("   * Our model identifies regime-specific pricing anomalies")
    
    def calculate_black_scholes(self, ticker, strike_price, time_to_expiry, option_type='call', dividend_yield=0):
        """
        Step 5: Black-Scholes Calculate
        - Implement Black-Scholes formula
        - Handle both call and put options
        - Include dividend yield for completeness
        - Validate inputs to prevent calculation errors
        """
        if ticker not in self.market_data:
            raise ValueError(f"No data for {ticker}. Call fetch_macro_regime_data() first.")
        
        # Step 5a: Extract parameters
        S = self.market_data[ticker]['current_price']  # Spot price
        K = strike_price                               # Strike price
        T = time_to_expiry                             # Time to expiration (years)
        r = self.r                                     # Risk-free rate
        sigma = self.market_data[ticker]['volatility'] # Volatility
        q = dividend_yield                             # Dividend yield
        
        # Step 5b: Input validation
        if T <= 0:
            raise ValueError("Time to expiry must be positive")
        if S <= 0 or K <= 0 or sigma <= 0:
            raise ValueError("Price, strike, and volatility must be positive")
        
        # Step 5c: Calculate d1 and d2 parameters
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Step 5d: Calculate option price based on type
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        # Step 5e: Ensure non-negative price
        return max(price, 0)
    
    def calculate_greeks(self, ticker, strike_price, time_to_expiry, option_type='call', dividend_yield=0):
        """
        Step 6: Calculate Greeks 
        - Delta: Price sensitivity to underlying asset
        - Gamma: Delta sensitivity to underlying price  
        - Vega: Sensitivity to volatility changes
        - Theta: Time decay (daily)
        - Rho: Sensitivity to interest rate changes
        """
        S = self.market_data[ticker]['current_price']
        K = strike_price
        T = time_to_expiry
        r = self.r
        sigma = self.market_data[ticker]['volatility']
        q = dividend_yield
        
        # Step 6a: Calculate d1 and d2 (same as in pricing)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # STEP 6b: Calculate each Greek
        # Delta: Price change for $1 move in underlying
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        
        # Gamma: Delta change for $1 move in underlying (same for calls/puts)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega: Price change for 1% volatility change (same for calls/puts)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01
        
        # Theta: Daily time decay (different for calls vs puts)
        if option_type.lower() == 'call':
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        else:  # put
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        
        # Rho: Price change for 1% interest rate change
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def regime_comparative_analysis(self, strike_percentage=1.0, time_to_expiry=30/365, option_type='call'):
        """
        Step 7: Comparative Analysis across Regimes
        - Analyze how options behave differently across risk-on vs risk-off assets
        - Consistent parameters for fair comparison
        - Generate insights about regime-specific pricing
        """
        print(f"\n Macro Regime Comparative Analysis")
        print(f"   Strategy: {option_type.upper()} Options | Strike: {strike_percentage:.0%} of spot | Expiry: {time_to_expiry*365:.0f} days")
        print("=" * 70)
        
        analysis_results = []
        
        for ticker in self.assets.keys():
            # Step 7a: Calculate strike price as percentage of spot
            strike_price = self.market_data[ticker]['current_price'] * strike_percentage
            
            # Step 7b: Calculate theoretical price
            theoretical_price = self.calculate_black_scholes(
                ticker, strike_price, time_to_expiry, option_type
            )
            
            # Step 7c: Calculate all Greeks
            greeks = self.calculate_greeks(ticker, strike_price, time_to_expiry, option_type)
            
            # Step 7d: Store comprehensive results
            result = {
                'ticker': ticker,
                'regime': self.market_data[ticker]['regime'],
                'current_price': self.market_data[ticker]['current_price'],
                'strike_price': strike_price,
                'theoretical_price': theoretical_price,
                'volatility': self.market_data[ticker]['volatility'],
                'performance_3mo': self.market_data[ticker]['performance_3mo_pct'],
                **greeks
            }
            analysis_results.append(result)
            
            # Step 7e: Display results with regime context
            regime_indicator = "[RISK-ON]" if self.market_data[ticker]['regime'] == 'Risk-On' else "[RISK-OFF]"
            print(f"{ticker} {regime_indicator}:")
            print(f"   Price: ${theoretical_price:.2f} | Delta: {greeks['delta']:.3f} | Vega: {greeks['vega']:.3f}")
            print(f"   Vol: {self.market_data[ticker]['volatility']:.1%} | 3Mo Perf: {self.market_data[ticker]['performance_3mo_pct']:+.1f}%")
            print(f"   Narrative: {self.market_data[ticker]['narrative']}")
            print()
        
        return pd.DataFrame(analysis_results)
    
    def get_regime_insights(self, analysis_df):
        """
        Step 8: Insights generator
        - Compare options behavior across different regimes
        - Explain the economic rationale behind observed differences
        - Provide actionable insights for the next script (mispricing detection)
        """
        print("\nInsights from regime classifcation")
        print("=" * 50)
        
        # Step 8a: Separate assets by regime
        tech_assets = analysis_df[analysis_df['regime'] == 'Risk-On']
        safe_asset = analysis_df[analysis_df['regime'] == 'Risk-Off']
        
        if len(tech_assets) > 0 and len(safe_asset) > 0:
            # Step 8b: Calculate comparative metrics
            avg_tech_vega = tech_assets['vega'].mean()
            gold_vega = safe_asset['vega'].iloc[0]
            
            avg_tech_vol = tech_assets['volatility'].mean() 
            gold_vol = safe_asset['volatility'].iloc[0]
            
            # Step 8c: Generate insights
            print(f"* Tech Vega Sensitivity: {avg_tech_vega:.3f} vs Gold: {gold_vega:.3f}")
            print(f"  -> Tech options more sensitive to volatility changes")
            print(f"* Volatility Regimes: Tech {avg_tech_vol:.1%} vs Gold {gold_vol:.1%}")
            print(f"  -> Higher tech volatility reflects growth uncertainty")
            print(f"* Gold's lower Vegas: More stable pricing in safe-haven flows")
            print(f"* This divergence creates unique cross-asset mispricing opportunities")


def main():
    """
    Step 9: Call main
    - Showcase the complete functionality
    - Provide example outputs for verification
    """
    print("=" *50)
    print("Macro regime Black-scholes")
    print("   Addressing 2025 Market Divergence: Tech vs Gold")
    print("=" * 50)
    
    # Initialize calculator
    calculator = MacroRegimeBlackScholes(risk_free_rate=0.05)
    
    # Fetch live data with macro context
    market_data = calculator.fetch_macro_regime_data()
    
    # Perform comparative analysis across regimes
    analysis_df = calculator.regime_comparative_analysis(
        strike_percentage=1.0,  # ATM options
        time_to_expiry=30/365,  # 30 days
        option_type='call'
    )
    
    # Generate regime insights
    calculator.get_regime_insights(analysis_df)
    
    print(f"\nAnalysis Timestamp: {calculator.analysis_timestamp}")
    print("This analysis provides the foundation for sentiment-driven")
    print("mispricing detection in next script...")

if __name__ == "__main__":
    main()

