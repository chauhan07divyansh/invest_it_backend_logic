# Keep all other imports as-is
import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from textblob import TextBlob
import warnings
import requests

# Local application imports
import config
from hf_utils import query_hf_api

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedPositionTradingSystem:
    def __init__(self, data_provider=None, mda_processor=None):
        try:
            self.news_api_key = config.NEWS_API_KEY
            self.position_trading_params = config.POSITION_TRADING_PARAMS
            self._validate_trading_params()
            self.initialize_stock_database()

            # --- API CONFIGURATION CHECKS ---
            self.sentiment_api_url = config.HF_SENTIMENT_API_URL
            self.sentiment_api_available = bool(self.sentiment_api_url)
            self.model_type = "SBERT API" if self.sentiment_api_available else "TextBlob"

            self.mda_api_url = config.HF_MDA_API_URL
            self.mda_api_available = bool(self.mda_api_url)
            self.mda_available = self.mda_api_available

            # --- Store injected data provider ---
            self.data_provider = data_provider
            if data_provider:
                logger.info("✅ Data provider injected into PositionTradingSystem")
            else:
                logger.warning("⚠️ No data provider provided - data fetching will fail")

            # --- Store MDA processor ---
            self.mda_processor = mda_processor
            if mda_processor:
                logger.info("✅ MDA Processor injected into PositionTradingSystem")
            else:
                logger.warning("⚠️ No MDA Processor - will use sample data")

            # --- Session for potential future use (kept for now) ---
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            logger.info("✅ EnhancedPositionTradingSystem initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing EnhancedPositionTradingSystem: {e}")
            raise

    # ==============================================================================
    #  API CALLING HELPER METHODS
    # ==============================================================================

    def _analyze_sentiment_via_api(self, articles: list) -> tuple[list, list] | None:
        """Analyzes news sentiment by calling the remote SBERT Hugging Face API."""
        try:
            payload = {"inputs": articles}
            api_results = query_hf_api(self.sentiment_api_url, payload)

            if api_results is None:
                raise ValueError("API call to SBERT HF Space failed or returned no data.")

            # Handle potential nested list structure from HF API
            if isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
                api_results = api_results[0]
                
            # Safely extract labels and scores
            sentiments = [res.get('label', 'neutral').lower() for res in api_results]
            confidences = [res.get('score', 0.5) for res in api_results]
            return sentiments, confidences
            
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logging.error(
                f"Could not parse SBERT API response. Error: {e}. Response: {api_results if 'api_results' in locals() else 'unknown'}")
            return None

    def _analyze_mda_via_api(self, mda_texts: list) -> dict | None:
        """Analyzes MDA text by calling the remote MDA Hugging Face API."""
        try:
            payload = {"inputs": mda_texts}
            api_results = query_hf_api(self.mda_api_url, payload)

            if api_results is None:
                raise ValueError("API call to MDA HF Space failed or returned no data.")

            # Handle potential nested list structure from HF API
            if isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
                 api_results = api_results[0]
                 
            sentiments = [res.get('label') for res in api_results]
            confidences = [res.get('score') for res in api_results]

            sentiment_scores = []
            valid_confidences = []
            for sentiment, confidence in zip(sentiments, confidences):
                if confidence is None: continue # Skip if confidence is missing
                valid_confidences.append(confidence)
                sentiment_str = str(sentiment).lower()
                if sentiment_str in ['positive', 'very_positive', 'label_4', 'label_3']:
                    sentiment_scores.append(confidence)
                elif sentiment_str in ['negative', 'very_negative', 'label_0', 'label_1']:
                    sentiment_scores.append(-confidence)
                else: # Neutral or unrecognized
                    sentiment_scores.append(0)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            # Scale sentiment (-1 to 1) to score (0 to 100)
            mda_score = 50 + (avg_sentiment * 50)
            mda_score = max(0, min(100, mda_score)) # Clamp score between 0 and 100

            management_tone = "Neutral"
            if mda_score >= 70:
                management_tone = "Very Optimistic"
            elif mda_score >= 60:
                management_tone = "Optimistic"
            elif mda_score <= 40:
                management_tone = "Pessimistic"
            elif mda_score <= 30: # Adding a 'Very Pessimistic' category
                management_tone = "Very Pessimistic"


            return {
                'mda_score': mda_score,
                'management_tone': management_tone,
                'confidence': np.mean(valid_confidences) if valid_confidences else 0,
                'analysis_method': 'Remote PyTorch BERT MDA Model (API)',
            }
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logging.error(
                f"Could not parse MDA API response. Error: {e}. Response: {api_results if 'api_results' in locals() else 'unknown'}")
            return None

    # ==============================================================================
    #  CORE ANALYSIS METHODS
    # ==============================================================================

    def analyze_news_sentiment(self, symbol, num_articles=20):
        """Fetches news and analyzes sentiment using API or fallback."""
        try:
            # Attempt to fetch real news first
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Real news (NewsAPI)"

            # If no real news, use sample news
            if not articles:
                 articles = self.get_sample_news(symbol)
                 news_source = "Sample news"
                 logger.warning(f"Using sample news for {symbol}")

            if not articles: # Should not happen with sample news, but good check
                return [], [], [], "No Analysis", "No Source"

            # Try API first if available
            if self.sentiment_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    logger.info(f"Analyzed news for {symbol} using SBERT API.")
                    return sentiments, articles, confidences, "SBERT API", news_source

            # Fallback to TextBlob if API failed or isn't available
            logging.warning(f"Falling back to TextBlob for news sentiment for {symbol}.")
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob Fallback", news_source

        except Exception as e:
            logging.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error Source"


    def _validate_trading_params(self):
        """Validate position trading parameters from config."""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier', 'max_positions',
                               'fundamental_weight', 'technical_weight', 'sentiment_weight', 'mda_weight'] # Added weights

            missing = [p for p in required_params if p not in self.position_trading_params]
            if missing:
                raise ValueError(f"Missing required trading parameters: {', '.join(missing)}")

            for param in required_params:
                value = self.position_trading_params[param]
                if not isinstance(value, (int, float)) or value < 0: # Allow 0 for weights
                    raise ValueError(f"Invalid trading parameter {param}: {value}. Must be a non-negative number.")
                if param.endswith('_weight') and value > 1.0:
                     raise ValueError(f"Weight parameter {param} cannot exceed 1.0: {value}")

            # Specific checks
            if self.position_trading_params['min_holding_period'] >= self.position_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")
            if self.position_trading_params['risk_per_trade'] > 0.05:
                logger.warning("risk_per_trade exceeds 5%, which is high for position trading.")
            if self.position_trading_params['max_positions'] <= 0:
                 raise ValueError("max_positions must be greater than 0")


            # Validate weights sum to 1.0
            total_weight = sum(self.position_trading_params[p] for p in required_params if p.endswith('_weight'))
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Scoring weights do not sum to 1.0: {total_weight:.3f}")

            logger.info("✅ Position trading parameters validated successfully")
        except Exception as e:
            logger.error(f"Error validating trading parameters: {e}")
            raise


    def updated_analyze_mda_sentiment(self, symbol):
        """Gets MD&A analysis, preferably from injected processor/cache."""
        try:
            if self.mda_processor:
                analysis = self.mda_processor.get_mda_analysis(symbol)
                if analysis:
                    logger.info(f"✅ Retrieved cached MD&A for {symbol}")
                    return analysis
                else:
                    logger.warning(f"⚠️ MD&A not in cache for {symbol}, generating sample.")
                    return self.get_sample_mda_analysis(symbol) # Fallback to sample if not in cache
            else:
                 # If no processor injected, use sample data directly
                logger.warning("MDA Processor not available, using sample data.")
                return self.get_sample_mda_analysis(symbol)
        except Exception as e:
            logger.error(f"MD&A analysis error for {symbol}: {e}")
            return self.get_sample_mda_analysis(symbol) # Fallback on error


    def calculate_position_trading_score(self, data, sentiment_data, fundamentals, trends, market_analysis, sector,
                                         mda_analysis=None):
        """Calculate comprehensive position trading score."""
        try:
            # Get base weights
            weights = {
                'fundamental': self.position_trading_params['fundamental_weight'],
                'technical': self.position_trading_params['technical_weight'],
                'sentiment': self.position_trading_params['sentiment_weight'],
                'mda': self.position_trading_params['mda_weight']
            }

            # 1. Calculate individual base scores
            scores = {
                'fundamental': self.calculate_fundamental_score(fundamentals, sector),
                'technical': self.calculate_technical_score_position(data),
                'sentiment': self.calculate_sentiment_score(sentiment_data),
                'trend': trends.get('trend_score', 50),
                'sector': market_analysis.get('sector_score', 60),
                'mda': mda_analysis.get('mda_score', 50) if mda_analysis else 50
            }

            # 2. Contextual Sentiment Multiplier
            sector_sentiment_multipliers = {
                'Information Technology': 1.2, 'Consumer Goods': 1.1, 'Financial Services': 1.1,
                'Pharmaceuticals': 1.2, 'Power': 0.8, 'Oil & Gas': 1.0, 'Default': 1.0
            }
            sentiment_multiplier = sector_sentiment_multipliers.get(sector, sector_sentiment_multipliers['Default'])
            contextual_sentiment_score = scores['sentiment'] * sentiment_multiplier
            logger.info(f"Applying sentiment multiplier {sentiment_multiplier} for {sector}. Original: {scores['sentiment']:.1f}, Contextual: {contextual_sentiment_score:.1f}")


            # 3. Combine scores using weights
            base_score = (
                scores['fundamental'] * weights['fundamental'] +
                scores['technical'] * weights['technical'] +
                contextual_sentiment_score * weights['sentiment'] + # Use contextual score
                scores['mda'] * weights['mda']
            )

            # 4. Apply modifiers
            trend_modifier = scores['trend'] / 100
            sector_modifier = scores['sector'] / 100
            # Weighted average modifier: 70% base, 20% trend, 10% sector
            final_score = base_score * (0.7 + 0.2 * trend_modifier + 0.1 * sector_modifier)

            # Volatility Penalty
            if data is not None and not data.empty and 'Close' in data.columns:
                 try:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252)
                    if volatility > 0.6: final_score *= 0.6
                    elif volatility > 0.4: final_score *= 0.8
                 except Exception: pass # Ignore volatility calc errors


            # Dividend Bonus
            div_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('expected_div_yield', 0)
            if isinstance(div_yield, (int, float)) and div_yield > 0.02:
                final_score *= 1.1

            # Momentum Bonus
            if trends.get('momentum_1y', 0) > 0.15 and trends.get('momentum_6m', 0) > 0:
                final_score *= 1.05

            # MDA Tone Bonus/Penalty
            if mda_analysis:
                management_tone = mda_analysis.get('management_tone', 'Neutral')
                if management_tone == 'Very Optimistic': final_score *= 1.08
                elif management_tone == 'Optimistic': final_score *= 1.04
                elif management_tone == 'Pessimistic': final_score *= 0.92
                elif management_tone == 'Very Pessimistic': final_score *= 0.85 # Added penalty


            return min(100, max(0, final_score)) # Clamp final score

        except Exception as e:
            logger.error(f"Error calculating position trading score: {e}")
            return 0


    # ==============================================================================
    #  MAIN ANALYSIS FUNCTION (UPDATED)
    # ==============================================================================
    def analyze_position_trading_stock(self, symbol, period="5y"):
        """Comprehensive position trading analysis using the injected data provider."""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None
            
            logger.info(f"Starting position trading analysis for {symbol}")

            # --- USE INJECTED DATA PROVIDER ---
            if not self.data_provider:
                logger.error(f"❌ No data provider injected. Cannot analyze {symbol}.")
                return None

            stock_data = self.data_provider.get_stock_data(
                symbol,
                fetch_ohlcv=True,
                fetch_fundamentals=True,
                period=period
            )

            if not stock_data or stock_data.get('errors'):
                logger.error(f"Provider error for {symbol}: {stock_data.get('errors', 'Unknown data provider error')}")
                return None

            # 1. Unpack OHLCV data into DataFrame
            ohlcv_list = stock_data.get('ohlcv')
            if not ohlcv_list:
                 logger.error(f"No OHLCV data returned from provider for {symbol}")
                 return None
            try:
                 data = pd.DataFrame(ohlcv_list)
                 data['Date'] = pd.to_datetime(data['date'])
                 data = data.set_index('Date').drop(columns=['date'])

                 # ----> ADD THIS LINE <----
                 data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                 # ----> NOW COLUMNS ARE CORRECT <----

            except Exception as e:
                 logger.error(f"Could not process OHLCV data for {symbol}: {e}")
                 return None
            
            # 2. Unpack fundamentals directly
            fundamentals = stock_data.get('fundamentals', {})
            if not fundamentals:
                 logger.warning(f"No fundamental data returned from provider for {symbol}")
                 fundamentals = {} # Ensure it's a dict even if empty

            # 3. Unpack other info
            base_symbol = stock_data.get('symbol', symbol.split('.')[0].upper())
            final_symbol = f"{base_symbol}.{stock_data.get('exchange_used', 'NSE')}"
            company_name = stock_data.get('company_name', base_symbol)
            # --- END DATA PROVIDER INTEGRATION ---


            # Extract basic info from internal DB (can supplement provider data)
            stock_info = self.get_stock_info_from_db(base_symbol)
            sector = stock_info.get('sector', 'Unknown')
            market_cap_category = stock_info.get('market_cap', 'Unknown')
            # Add expected div yield from DB if not in fundamentals
            if 'expected_div_yield' not in fundamentals:
                 fundamentals['expected_div_yield'] = stock_info.get('div_yield', 0)


            # Current market data calculation (remains the same)
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100 if data['Close'].iloc[-2] != 0 else 0
                else: price_change, price_change_pct = 0, 0
            except Exception as e:
                logger.error(f"Error calculating price changes: {e}")
                current_price, price_change, price_change_pct = 0, 0, 0 # Fallback


            # --- ANALYSIS STEPS (using data/fundamentals from provider) ---
            
            # Long-term trend analysis
            try: trends = self.analyze_long_term_trends(data)
            except Exception as e: logger.error(f"Trend analysis error: {e}"); trends = {'trend_score': 50}

            # Market cycle and sector analysis
            try: market_analysis = self.analyze_market_cycles(base_symbol, data) # Use base_symbol
            except Exception as e: logger.error(f"Market analysis error: {e}"); market_analysis = {'sector_score': 60}

            # News sentiment analysis
            try: sentiment_results = self.analyze_news_sentiment(final_symbol) # Use final_symbol with exchange
            except Exception as e: logger.error(f"News sentiment error: {e}"); sentiment_results = ([], [], [], "Error", "Error")

            # MDA sentiment analysis
            try: mda_analysis = self.updated_analyze_mda_sentiment(base_symbol) # Use base_symbol
            except Exception as e: logger.error(f"MDA analysis error: {e}"); mda_analysis = self.get_sample_mda_analysis(base_symbol)

            # Risk metrics
            try: risk_metrics = self.calculate_risk_metrics(data)
            except Exception as e: logger.error(f"Risk metrics error: {e}"); risk_metrics = {'volatility': 0.3, 'atr': current_price * 0.02 if current_price else 0}

            # --- FINAL SCORE CALCULATION ---
            try:
                position_score = self.calculate_position_trading_score(
                    data, sentiment_results, fundamentals, trends, market_analysis, sector, mda_analysis
                )
            except Exception as e: logger.error(f"Score calculation error: {e}"); position_score = 0

            # --- TRADING PLAN GENERATION ---
            try:
                trading_plan = self.generate_position_trading_plan(
                    data, position_score, risk_metrics, fundamentals, trends
                )
            except Exception as e: logger.error(f"Trading plan error: {e}"); trading_plan = {'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed'}


            # --- Compile comprehensive results ---
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'market_cap_category': market_cap_category,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,

                'fundamentals': fundamentals, # Directly from provider
                'fundamental_score': self.calculate_fundamental_score(fundamentals, sector),

                'trends': trends,
                'trend_score': trends.get('trend_score', 50),

                'market_analysis': market_analysis,

                # Technical indicators (ensure data length before calculating)
                'rsi_30': self.calculate_rsi(data['Close'], period=30).iloc[-1] if len(data) >= 31 else None,
                'ma_50': trends.get('ma_50'),
                'ma_100': trends.get('ma_100'),
                'ma_200': trends.get('ma_200'),

                'sentiment': {
                    'scores': sentiment_results[0] if sentiment_results else [],
                    'articles': sentiment_results[1] if sentiment_results else [],
                    'confidence': sentiment_results[2] if sentiment_results else [],
                    'method': sentiment_results[3] if sentiment_results else "N/A",
                    'source': sentiment_results[4] if sentiment_results else "N/A",
                    'sentiment_summary': self.get_sentiment_summary(sentiment_results[0] if sentiment_results else [])
                },

                'mda_sentiment': mda_analysis,
                'risk_metrics': risk_metrics,
                'position_score': position_score,
                'trading_plan': trading_plan,

                'model_type': self.model_type,
                'mda_model_available': self.mda_api_available,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'Position Trading (Long-term with MDA)'
            }

            logger.info(f"✅ Successfully analyzed {symbol} with position score {position_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Critical error analyzing {symbol} for position trading: {e}")
            logger.error(traceback.format_exc())
            return None


    # ==============================================================================
    #  DATABASE INITIALIZATION (Kept as is - internal symbol info)
    # ==============================================================================
    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database with fundamental data structure"""
        # --- THIS METHOD REMAINS THE SAME ---
        try:
            self.indian_stocks = {
                 # NIFTY 50 Stocks with enhanced info
                 "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.003},
                 "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.025},
                 "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
                 "INFY": {"name": "Infosys", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.023},
                 "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.014},
                 "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                 "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.005},
                 "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services", "market_cap": "Large", "div_yield": 0.002},
                 "LT": {"name": "Larsen & Toubro", "sector": "Construction", "market_cap": "Large", "div_yield": 0.018},
                 "SBIN": {"name": "State Bank of India", "sector": "Banking", "market_cap": "Large", "div_yield": 0.035},
                 "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications", "market_cap": "Large", "div_yield": 0.008},
                 "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.008},
                 "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.012},
                 "TITAN": {"name": "Titan Company", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.005},
                 "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.008},
                 "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement", "market_cap": "Large", "div_yield": 0.005},
                 "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.008},
                 "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.025},
                 "AXISBANK": {"name": "Axis Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                 "WIPRO": {"name": "Wipro", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.015},
                 "NTPC": {"name": "NTPC", "sector": "Power", "market_cap": "Large", "div_yield": 0.045},
                 "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power", "market_cap": "Large", "div_yield": 0.038},
                 "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.055},
                 "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.032},
                 "TATASTEEL": {"name": "Tata Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.025},
                 "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate", "market_cap": "Large", "div_yield": 0.001},
                 "COALINDIA": {"name": "Coal India", "sector": "Mining", "market_cap": "Large", "div_yield": 0.065},
                 "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals", "market_cap": "Large", "div_yield": 0.008},
                 "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.012},
                 "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.022},
                 "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.018},
                 "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.025},
                 "GRASIM": {"name": "Grasim Industries", "sector": "Cement", "market_cap": "Large", "div_yield": 0.015},
                 "SHREECEM": {"name": "Shree Cement", "sector": "Cement", "market_cap": "Large", "div_yield": 0.003},
                 "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.005},
                 "UPL": {"name": "UPL Limited", "sector": "Chemicals", "market_cap": "Large", "div_yield": 0.012},
                 "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.035},
                 "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.005},
                 "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.008},
                 "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.012},
                 "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.008},
                 "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.015},
                 "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.042},
                 "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare", "market_cap": "Large", "div_yield": 0.002},
                 "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services", "market_cap": "Large", "div_yield": 0.008},
                 "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance", "market_cap": "Large", "div_yield": 0.012},
                 "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance", "market_cap": "Large", "div_yield": 0.008},
                 "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.015},
                 "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure", "market_cap": "Large", "div_yield": 0.012},
                 "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.008},
                 "ITC": {"name": "ITC Limited", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.055},

                 # Additional Mid & Small Cap Stocks
                 "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.012},
                 "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.008},
                 "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals", "market_cap": "Mid", "div_yield": 0.005},
                 "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.018},
                 "DABUR": {"name": "Dabur India", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.012},
                 "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals", "market_cap": "Mid", "div_yield": 0.008},
                 "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals", "market_cap": "Mid", "div_yield": 0.005},
                 "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile", "market_cap": "Mid", "div_yield": 0.012}, # Note: Name might change
                 "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile", "market_cap": "Mid", "div_yield": 0.008},
                 "MRF": {"name": "MRF Limited", "sector": "Automobile", "market_cap": "Mid", "div_yield": 0.015},
                 "DMART": {"name": "Avenue Supermarts", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.001},
                 "TRENT": {"name": "Trent Limited", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.002},
                 "PAGEIND": {"name": "Page Industries", "sector": "Textiles", "market_cap": "Mid", "div_yield": 0.003},
            }
            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")
            logger.info(f"Initialized internal stock database with {len(self.indian_stocks)} symbols")
        except Exception as e:
            logger.error(f"Error initializing stock database: {e}")
            self.indian_stocks = { # Minimal fallback
                 "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.003},
                 "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.025},
                 "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")


    # ==============================================================================
    #  REMOVED OBSOLETE METHODS
    # ==============================================================================
    # - get_indian_stock_data()  <- REMOVED
    # - analyze_fundamental_metrics() <- REMOVED


    # ==============================================================================
    #  CALCULATION METHODS (Kept as is - depend on DataFrame and fundamentals dict)
    # ==============================================================================
    def calculate_fundamental_score(self, fundamentals, sector):
        """Calculate fundamental score based on provided fundamentals dict."""
        # --- THIS METHOD REMAINS THE SAME ---
        # It now receives the 'fundamentals' dict directly from the data provider
        try:
            score = 0
            max_score = 100

            # P/E Ratio (check if not None and numeric)
            pe_ratio = fundamentals.get('pe_ratio')
            if isinstance(pe_ratio, (int, float)):
                if 8 < pe_ratio < 25: score += 15
                elif 5 < pe_ratio <= 8: score += 12
                elif 25 <= pe_ratio < 35: score += 8

            # PEG Ratio
            peg_ratio = fundamentals.get('peg_ratio')
            if isinstance(peg_ratio, (int, float)):
                 if 0.5 < peg_ratio < 1.0: score += 15
                 elif 1.0 <= peg_ratio < 1.5: score += 10
                 
            # Growth (handle None or non-numeric)
            rev_g = fundamentals.get('revenue_growth') or fundamentals.get('sales_growth_3y')
            earn_g = fundamentals.get('earnings_growth') or fundamentals.get('profit_growth_3y')

            if isinstance(rev_g, (int, float)):
                 rev_g_pct = rev_g / 100 if abs(rev_g) > 1 else rev_g # Assume > 1 means percentage
                 if rev_g_pct > 0.20: score += 10
                 elif rev_g_pct > 0.15: score += 8
                 elif rev_g_pct > 0.10: score += 5
                 
            if isinstance(earn_g, (int, float)):
                 earn_g_pct = earn_g / 100 if abs(earn_g) > 1 else earn_g
                 if earn_g_pct > 0.25: score += 10
                 elif earn_g_pct > 0.15: score += 8
                 elif earn_g_pct > 0.10: score += 5

            # ROE
            roe = fundamentals.get('roe')
            if isinstance(roe, (int, float)):
                 roe_pct = roe / 100 if abs(roe) > 1 else roe
                 if roe_pct > 0.20: score += 10
                 elif roe_pct > 0.15: score += 8
                 elif roe_pct > 0.12: score += 5

            # Debt to Equity
            debt_equity = fundamentals.get('debt_to_equity')
            if isinstance(debt_equity, (int, float)):
                 if debt_equity < 0.3: score += 10
                 elif debt_equity < 0.6: score += 8
                 elif debt_equity < 1.0: score += 4

            # Profitability (assuming these might be percentages directly)
            profit_margin = fundamentals.get('profit_margin')
            op_margin = fundamentals.get('operating_margin')
            if isinstance(profit_margin, (int, float)):
                 pm_pct = profit_margin / 100 if abs(profit_margin) > 1 else profit_margin
                 if pm_pct > 0.15: score += 5
                 elif pm_pct > 0.10: score += 3
                 
            if isinstance(op_margin, (int, float)):
                 opm_pct = op_margin / 100 if abs(op_margin) > 1 else op_margin
                 if opm_pct > 0.20: score += 5
                 elif opm_pct > 0.15: score += 3


            # Dividend Yield
            div_yield = fundamentals.get('dividend_yield') or fundamentals.get('expected_div_yield', 0)
            if isinstance(div_yield, (int, float)):
                 div_yield_pct = div_yield / 100 if abs(div_yield) > 1 else div_yield
                 if div_yield_pct > 0.03: score += 10
                 elif div_yield_pct > 0.015: score += 6
                 elif div_yield_pct > 0.005: score += 3

            # Financial Health
            current_ratio = fundamentals.get('current_ratio')
            pb_ratio = fundamentals.get('price_to_book') or fundamentals.get('priceToBook') # Add alias
            if isinstance(current_ratio, (int, float)):
                 if current_ratio > 1.5: score += 5
                 elif current_ratio > 1.2: score += 3
                 
            if isinstance(pb_ratio, (int, float)):
                 if pb_ratio < 2.0: score += 5
                 elif pb_ratio < 3.0: score += 3


            return min(score, max_score)
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0


    def analyze_long_term_trends(self, data):
        """Analyze long-term trends from OHLCV DataFrame."""
        # --- THIS METHOD REMAINS THE SAME ---
        try:
            if data is None or data.empty or len(data) < 200: # Need enough data for 200MA
                 logger.warning("Insufficient data for full trend analysis.")
                 # Provide default/partial values if data is too short
                 current_price = data['Close'].iloc[-1] if not data.empty else 0
                 return {'trend_score': 25, 'ma_50_slope': 0, 'ma_200_slope': 0, 'above_ma_200': False,
                         'momentum_6m': 0, 'momentum_1y': 0, 'ma_50': None, 'ma_100': None, 'ma_200': None}

            # Calculate MAs safely
            ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
            ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]
            current_price = data['Close'].iloc[-1]

            trend_score = 25 # Default to sideways/mixed
            # Check MAs only if they are valid numbers
            if not any(pd.isna([ma_50, ma_100, ma_200, current_price])):
                 if current_price > ma_50 > ma_100 > ma_200: trend_score = 100
                 elif current_price > ma_50 > ma_100: trend_score = 75
                 elif current_price > ma_100: trend_score = 50
                 elif current_price < ma_50 < ma_100 < ma_200: trend_score = 0


            # Slopes (handle potential NaN results if window is too short for slice)
            ma_50_series = self.safe_rolling_calculation(data['Close'], 50, 'mean')
            ma_200_series = self.safe_rolling_calculation(data['Close'], 200, 'mean')
            ma_50_slope = (ma_50_series.iloc[-1] - ma_50_series.iloc[-21]) / ma_50_series.iloc[-21] if len(ma_50_series) > 20 and ma_50_series.iloc[-21] !=0 else 0
            ma_200_slope = (ma_200_series.iloc[-1] - ma_200_series.iloc[-51]) / ma_200_series.iloc[-51] if len(ma_200_series) > 50 and ma_200_series.iloc[-51] !=0 else 0


            # Momentum
            momentum_6m = 0
            momentum_1y = 0
            if len(data) > 126:
                price_6m_ago = data['Close'].iloc[-126]
                if price_6m_ago != 0: momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            if len(data) > 252:
                price_1y_ago = data['Close'].iloc[-252]
                if price_1y_ago != 0: momentum_1y = (current_price - price_1y_ago) / price_1y_ago
                

            return {
                'trend_score': trend_score,
                'ma_50_slope': ma_50_slope,
                'ma_200_slope': ma_200_slope,
                'above_ma_200': current_price > ma_200 if not pd.isna(ma_200) else False,
                'momentum_6m': momentum_6m,
                'momentum_1y': momentum_1y,
                'ma_50': ma_50,
                'ma_100': ma_100,
                'ma_200': ma_200
            }
        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {e}")
            # Return safe defaults
            return {'trend_score': 50, 'ma_50_slope': 0, 'ma_200_slope': 0, 'above_ma_200': False,
                    'momentum_6m': 0, 'momentum_1y': 0, 'ma_50': None, 'ma_100': None, 'ma_200': None}


    def analyze_market_cycles(self, symbol, data):
        """Analyze market cycles and sector rotation."""
         # --- THIS METHOD REMAINS THE SAME ---
        try:
            # Use internal DB to get sector, as provider might not have it
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')

            sector_score = 60 # Default
            # Sector scores (simplified example)
            if sector in ['Banking', 'Financial Services']: sector_score = 65
            elif sector in ['Real Estate', 'Infrastructure']: sector_score = 55
            elif sector in ['Consumer Goods', 'Pharmaceuticals', 'Healthcare']: sector_score = 75
            elif sector in ['Automobile', 'Steel', 'Cement', 'Metals']: sector_score = 60
            elif sector in ['Information Technology']: sector_score = 70
            elif sector in ['Power', 'Utilities']: sector_score = 80
            elif sector in ['Oil & Gas', 'Mining']: sector_score = 55

            return {
                'sector_score': sector_score,
                'sector': sector,
                'cycle_stage': self.determine_market_cycle(data), # Helper uses data
                'sector_preference': self.get_sector_preference(sector) # Helper uses sector
            }
        except Exception as e:
            logger.error(f"Error in market cycle analysis: {e}")
            return {'sector_score': 60, 'sector': 'Unknown', 'cycle_stage': 'Unknown', 'sector_preference': 'Neutral'}

    def determine_market_cycle(self, data):
        """Determine current market cycle stage based on MA200."""
         # --- THIS HELPER REMAINS THE SAME ---
        try:
            if data is None or data.empty or len(data) < 200:
                 return "Unknown" # Not enough data

            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')
            if ma_200.isna().all(): return "Unknown" # MA calculation failed

            # Count days above MA200 in the last 120 days
            check_period = min(120, len(data))
            above_ma_200_days = (data['Close'].iloc[-check_period:] > ma_200.iloc[-check_period:]).sum()
            above_ma_200_pct = above_ma_200_days / check_period

            if above_ma_200_pct > 0.75: return "Bull Market"
            elif above_ma_200_pct < 0.25: return "Bear Market"
            else: return "Transitional"
        except Exception as e:
            logger.error(f"Error determining market cycle: {e}")
            return "Unknown"

    def get_sector_preference(self, sector):
        """Get sector preference for position trading."""
         # --- THIS HELPER REMAINS THE SAME ---
        high_preference = ['Consumer Goods', 'Information Technology', 'Healthcare', 'Pharmaceuticals', 'Power', 'Banking']
        medium_preference = ['Telecommunications', 'Oil & Gas', 'Chemicals', 'Cement', 'Financial Services'] # Added Fin Services
        if sector in high_preference: return 'High'
        elif sector in medium_preference: return 'Medium'
        else: return 'Low'


    def calculate_technical_score_position(self, data):
        """Calculate technical score optimized for position trading."""
         # --- THIS METHOD REMAINS THE SAME ---
        try:
            if data is None or data.empty or len(data) < 52: # Need enough for long MACD
                logger.warning("Insufficient data for full technical score.")
                return 25 # Low default score

            technical_score = 0
            current_price = data['Close'].iloc[-1]

            # RSI (30)
            if len(data) >= 31:
                rsi = self.calculate_rsi(data['Close'], period=30).iloc[-1]
                if not pd.isna(rsi):
                    if 40 <= rsi <= 60: technical_score += 30
                    elif 30 <= rsi < 40: technical_score += 25
                    elif 60 < rsi <= 70: technical_score += 20
                    elif rsi < 30: technical_score += 15
            
            # Moving Averages
            if len(data) >= 200:
                ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
                ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]
                if not any(pd.isna([ma_50, ma_100, ma_200, current_price])):
                    if current_price > ma_50 > ma_100 > ma_200: technical_score += 25
                    elif current_price > ma_50 > ma_100: technical_score += 20
                    elif current_price > ma_200: technical_score += 15
                    elif ma_50 > ma_100: technical_score += 10 # Golden cross potential
            
            # Volume Trend
            if 'Volume' in data.columns and len(data) >= 50:
                recent_vol = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                long_vol = self.safe_rolling_calculation(data['Volume'], 50, 'mean').iloc[-1]
                if not pd.isna(recent_vol) and not pd.isna(long_vol) and long_vol > 0:
                     volume_ratio = recent_vol / long_vol
                     if volume_ratio > 1.2: technical_score += 15
                     elif volume_ratio > 1.0: technical_score += 10
            
            # Long-term MACD (26, 52, 18)
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'], fast=26, slow=52, signal=18)
            if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) and not pd.isna(signal_line.iloc[-1]):
                 if macd_line.iloc[-1] > signal_line.iloc[-1]: technical_score += 15 # Bullish crossover/state
                 # Check rising histogram
                 if len(histogram) > 1 and not pd.isna(histogram.iloc[-1]) and not pd.isna(histogram.iloc[-2]):
                     if histogram.iloc[-1] > histogram.iloc[-2]: technical_score += 5
            
            # Support/Resistance (window=50)
            support, resistance = self.calculate_support_resistance(data, window=50)
            if support and resistance and not any(pd.isna([support, resistance, current_price])) and support > 0 and current_price > 0:
                 dist_support_pct = (current_price - support) / support
                 if 0.05 <= dist_support_pct <= 0.20: technical_score += 15 # Good entry zone
                 elif dist_support_pct > 0.20: technical_score += 10 # Well above
                 elif dist_support_pct < 0.05: technical_score += 8 # Very close


            return min(100, max(0, technical_score))
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0


    def calculate_sentiment_score(self, sentiment_data):
        """Calculate sentiment score from news analysis results."""
         # --- THIS METHOD REMAINS THE SAME ---
        try:
            if not sentiment_data or len(sentiment_data) < 4 or not sentiment_data[0]: # Check if sentiments list exists and is not empty
                 return 50 # Neutral default

            sentiments, _, confidences, _, _ = sentiment_data
            if not sentiments or not confidences or len(sentiments) != len(confidences):
                 logger.warning("Mismatched sentiments/confidences or empty lists.")
                 return 50

            sentiment_value = 0
            total_weight = 0
            for sentiment, confidence in zip(sentiments, confidences):
                 # Use confidence as weight, default to 0.5 if invalid
                 weight = float(confidence) if isinstance(confidence, (int, float)) and not pd.isna(confidence) else 0.5
                 
                 sentiment_str = str(sentiment).lower()
                 if 'positive' in sentiment_str: sentiment_value += weight
                 elif 'negative' in sentiment_str: sentiment_value -= weight
                 # Neutral adds 0
                 total_weight += weight

            if total_weight > 0:
                 normalized_sentiment = sentiment_value / total_weight
                 sentiment_score = 50 + (normalized_sentiment * 50) # Scale -1 to 1 -> 0 to 100
            else: sentiment_score = 50 # If no valid weights, return neutral

            return min(100, max(0, sentiment_score)) # Clamp score
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 50


    def generate_position_trading_plan(self, data, score, risk_metrics, fundamentals, trends):
        """Generate realistic position trading plan."""
        # --- THIS METHOD REMAINS THE SAME ---
        default_plan = { 'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed', 'entry_timing': 'Unknown',
                         'stop_loss': 0, 'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0}, 'support': 0,
                         'resistance': 0, 'holding_period': 'Unknown', 'trade_management_note': 'Unable to generate plan',
                         'stop_distance_pct': 0, 'upside_potential': 0, 'risk_reward_ratio': 0 }
        try:
            if data is None or data.empty: return default_plan
            current_price = data['Close'].iloc[-1]
            if current_price <= 0: return default_plan # Cannot plan with zero price

            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0: atr = current_price * 0.02

            # Entry signal based on score
            if score >= 80: entry_signal, entry_strategy, holding_period = "STRONG BUY", "Accumulate on dips; high conviction", "12-24 months"
            elif score >= 65: entry_signal, entry_strategy, holding_period = "BUY", "Enter gradually; good prospects", "9-18 months"
            elif score >= 40: entry_signal, entry_strategy, holding_period = "HOLD/WATCH", "Wait for better entry/confirmation", "Monitor"
            else: entry_signal, entry_strategy, holding_period = "AVOID", "Weaknesses detected", "Not recommended"


            # Stop Loss (wider, ATR-based, capped)
            stop_loss_distance = min(atr * 3.0, current_price * 0.10) # 3x ATR or 10% max
            stop_loss = max(current_price - stop_loss_distance, 0) # Ensure > 0

            # Support/Resistance (100-day window)
            support, resistance = self.calculate_support_resistance(data, window=100)
            support = support if support and not pd.isna(support) else current_price * 0.90
            resistance = resistance if resistance and not pd.isna(resistance) else current_price * 1.15

            # Adjust SL based on support
            if support > 0: stop_loss = max(stop_loss, support * 0.97) # Place SL just below support
            stop_loss = max(stop_loss, current_price * 0.85) # Absolute minimum 15% below entry


            # Realistic Targets (Risk/Reward based, capped)
            risk_per_share = current_price - stop_loss
            if risk_per_share <=0: risk_per_share = current_price * 0.1 # Fallback risk if SL is too close/invalid
            
            target_1 = current_price + (risk_per_share * 1.5)
            target_2 = current_price + (risk_per_share * 2.5)
            target_3 = current_price + (risk_per_share * 3.5)

            # Cap targets
            max_reasonable_target = current_price * 1.40 # 40% max gain target
            target_1 = min(target_1, current_price * 1.15, resistance if resistance > current_price else current_price * 1.15) # Cap near resistance
            target_2 = min(target_2, current_price * 1.25)
            target_3 = min(target_3, max_reasonable_target)

            # Ensure progressive targets
            target_1 = max(target_1, current_price * 1.05) # Min 5% gain for T1
            target_2 = max(target_2, target_1 * 1.05) # T2 > T1
            target_3 = max(target_3, target_2 * 1.05) # T3 > T2


            # Entry Timing
            ma_200 = trends.get('ma_200')
            entry_timing = "Wait for pullback or confirmation" # Default
            if ma_200 and not pd.isna(ma_200):
                if current_price > ma_200 * 1.02: entry_timing = "Buy on pullbacks or accumulate"
                elif current_price > ma_200 * 0.98: entry_timing = "Near long-term MA; watch for confirmation"
                else: entry_timing = "Wait for reclaim of 200-day MA"

            # Calculate final metrics
            stop_dist_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
            upside_pot = ((target_2 - current_price) / current_price) * 100 if current_price > 0 else 0
            rr_ratio = upside_pot / stop_dist_pct if stop_dist_pct > 0 else 0

            trade_management_note = ("Book partial profits at targets (e.g., 1/3 each). "
                                     "Trail stop loss to breakeven after Target 1. "
                                     "Review fundamentals quarterly.")

            return {
                'entry_signal': entry_signal, 'entry_strategy': entry_strategy, 'entry_timing': entry_timing,
                'stop_loss': round(stop_loss, 2),
                'targets': {'target_1': round(target_1, 2), 'target_2': round(target_2, 2), 'target_3': round(target_3, 2)},
                'support': round(support, 2), 'resistance': round(resistance, 2), 'holding_period': holding_period,
                'trade_management_note': trade_management_note, 'stop_distance_pct': round(stop_dist_pct, 2),
                'upside_potential': round(upside_pot, 2), 'risk_reward_ratio': round(rr_ratio, 2)
            }
        except Exception as e:
            logger.error(f"Error generating position trading plan: {e}")
            logger.error(traceback.format_exc())
            return default_plan


    # ==============================================================================
    #  HELPER METHODS (Technical Analysis, News, Utils - Kept as is)
    # ==============================================================================
    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations on pandas Series."""
        # --- THIS HELPER REMAINS THE SAME ---
        try:
            if data is None or data.empty or len(data) < window:
                # Return a series of NaNs with the same index if possible
                return pd.Series(np.nan, index=data.index if hasattr(data, 'index') else None)

            if operation == 'mean': return data.rolling(window=window, min_periods=max(1, window // 2)).mean() # Require at least half window
            elif operation == 'std': return data.rolling(window=window, min_periods=max(1, window // 2)).std()
            elif operation == 'min': return data.rolling(window=window, min_periods=max(1, window // 2)).min()
            elif operation == 'max': return data.rolling(window=window, min_periods=max(1, window // 2)).max()
            else:
                logger.error(f"Unknown rolling operation: {operation}")
                return pd.Series(np.nan, index=data.index)
        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {e}")
            return pd.Series(np.nan, index=data.index if hasattr(data, 'index') else None)


    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for a pandas Series."""
        # --- THIS HELPER REMAINS THE SAME ---
        try:
            if prices is None or prices.empty or len(prices) < period + 1:
                return pd.Series(50, index=prices.index if hasattr(prices, 'index') else None) # Default neutral

            delta = prices.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)

            # Use Exponential Moving Average (EMA) for RSI calculation - more standard
            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0, 1e-6) # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50) # Fill initial NaNs with neutral 50
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index if hasattr(prices, 'index') else None)


    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD for a pandas Series."""
        # --- THIS HELPER REMAINS THE SAME ---
        try:
            if prices is None or prices.empty or len(prices) < slow:
                 empty_series = pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)
                 return empty_series, empty_series, empty_series

            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            empty_series = pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)
            return empty_series, empty_series, empty_series


    def calculate_support_resistance(self, data, window=20):
        """Calculate simple support/resistance based on rolling min/max."""
        # --- THIS HELPER REMAINS THE SAME ---
        try:
            if data is None or data.empty or 'Low' not in data.columns or 'High' not in data.columns or len(data) < window:
                 return None, None # Cannot calculate

            # Use rolling min/max over the window
            support = self.safe_rolling_calculation(data['Low'], window, 'min').iloc[-1]
            resistance = self.safe_rolling_calculation(data['High'], window, 'max').iloc[-1]
            
            # Basic pivot point calculation as an alternative/supplement (using last available data)
            last_high = data['High'].iloc[-1]
            last_low = data['Low'].iloc[-1]
            last_close = data['Close'].iloc[-1]
            pivot = (last_high + last_low + last_close) / 3
            s1 = (2 * pivot) - last_high
            r1 = (2 * pivot) - last_low

            # Return the rolling S/R if valid, otherwise maybe use pivots? (Keeping rolling for now)
            return support, resistance

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return None, None


    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics from OHLCV DataFrame."""
        # --- THIS METHOD REMAINS THE SAME ---
        default_metrics = {'volatility': 0.3, 'var_95': -0.05, 'max_drawdown': -0.20, 'sharpe_ratio': 0.0, 'atr': 0, 'risk_level': 'HIGH'}
        try:
            if data is None or data.empty or 'Close' not in data.columns or len(data) < 22: # Need enough for ATR and std dev
                logger.warning("Insufficient data for full risk metrics.")
                last_close = data['Close'].iloc[-1] if not data.empty else 0
                default_metrics['atr'] = last_close * 0.02 if last_close else 0
                return default_metrics


            returns = data['Close'].pct_change().dropna()
            if returns.empty: return default_metrics

            # Volatility
            volatility = returns.std() * np.sqrt(252)
            volatility = volatility if not pd.isna(volatility) and volatility >= 0 else 0.3

            # VaR
            var_95 = np.percentile(returns, 5) if len(returns) > 20 else -0.05
            var_95 = var_95 if not pd.isna(var_95) else -0.05

            # Max Drawdown
            rolling_max = data['Close'].cummax()
            drawdown = (data['Close'] - rolling_max) / rolling_max.replace(0, 1) # Avoid div by zero
            max_drawdown = drawdown.min()
            max_drawdown = max_drawdown if not pd.isna(max_drawdown) else -0.20

            # Sharpe Ratio (simple version)
            risk_free_rate_annual = 0.06 # Assumed 6%
            mean_daily_return = returns.mean()
            std_daily_return = returns.std()
            sharpe_ratio = 0.0
            if std_daily_return > 0:
                 excess_return = mean_daily_return - (risk_free_rate_annual / 252)
                 sharpe_ratio = (excess_return / std_daily_return) * np.sqrt(252) # Annualized
            sharpe_ratio = sharpe_ratio if not pd.isna(sharpe_ratio) else 0.0


            # ATR (Average True Range) - standard 14 period
            if len(data) >= 15 and all(c in data.columns for c in ['High', 'Low', 'Close']):
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).dropna()
                atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1] # Use EMA for ATR
            else: atr = data['Close'].iloc[-1] * 0.02 # Fallback
            atr = atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.02


            # Risk Level
            if volatility > 0.40: risk_level = 'HIGH'
            elif volatility > 0.25: risk_level = 'MEDIUM'
            else: risk_level = 'LOW'

            return {'volatility': volatility, 'var_95': var_95, 'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio, 'atr': atr, 'risk_level': risk_level}
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            last_close = data['Close'].iloc[-1] if data is not None and not data.empty else 0
            default_metrics['atr'] = last_close * 0.02 if last_close else 0
            return default_metrics


    # News sentiment methods (kept as is)
    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob."""
        # --- THIS METHOD REMAINS THE SAME ---
        sentiments, confidences = [], []
        if not articles: return sentiments, confidences
        for article in articles:
             try:
                 if not article or not isinstance(article, str):
                     sentiments.append('neutral'); confidences.append(0.3); continue
                 blob = TextBlob(article)
                 polarity = blob.sentiment.polarity
                 confidence = abs(polarity) # Use polarity magnitude as confidence proxy
                 if polarity > 0.1: sentiments.append('positive'); confidences.append(min(confidence, 0.9)) # Cap confidence
                 elif polarity < -0.1: sentiments.append('negative'); confidences.append(min(confidence, 0.9))
                 else: sentiments.append('neutral'); confidences.append(max(0.3, 1 - confidence*2)) # Higher confidence for near-zero polarity
             except Exception as e:
                 logger.warning(f"TextBlob error: {e}"); sentiments.append('neutral'); confidences.append(0.3)
        return sentiments, confidences

    def get_sentiment_summary(self, sentiment_scores):
        """Get summary count of sentiment scores."""
         # --- THIS HELPER REMAINS THE SAME ---
        if not sentiment_scores: return {'positive': 0, 'negative': 0, 'neutral': 0}
        positive_count = sum(1 for s in sentiment_scores if 'positive' in str(s).lower())
        negative_count = sum(1 for s in sentiment_scores if 'negative' in str(s).lower())
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        return {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count}


    # Utility methods (kept as is)
    def get_all_stock_symbols(self):
        """Get all stock symbols from internal database."""
        # --- THIS METHOD REMAINS THE SAME ---
        try:
            if not hasattr(self, 'indian_stocks') or not self.indian_stocks:
                 raise ValueError("Stock database not initialized or empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {e}")
            return ["RELIANCE", "TCS", "HDFCBANK"] # Minimal fallback


    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database."""
        # --- THIS METHOD REMAINS THE SAME ---
        try:
            if not symbol: raise ValueError("Empty symbol provided")
            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol: raise ValueError("Invalid symbol format")
            # Default structure if symbol not found
            default = {"name": base_symbol, "sector": "Unknown", "market_cap": "Unknown", "div_yield": 0}
            if not hasattr(self, 'indian_stocks'): return default
            return self.indian_stocks.get(base_symbol, default)
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {"name": str(symbol), "sector": "Unknown", "market_cap": "Unknown", "div_yield": 0}

    def fetch_indian_news(self, symbol, num_articles=20):
        """Fetch news for Indian companies using NewsAPI."""
        # --- THIS METHOD REMAINS THE SAME (Consider moving to provider later) ---
        try:
            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not configured. Cannot fetch real news.")
                return None

            base_symbol = str(symbol).split('.')[0].upper()
            stock_info = self.get_stock_info_from_db(base_symbol)
            # Use company name for broader search, fallback to symbol
            query_term = stock_info.get("name", base_symbol) if stock_info.get("name") != base_symbol else base_symbol
            
            # Construct URL - focus on Indian market
            url = ("https://newsapi.org/v2/everything?"
                   f"q={query_term}+India+stock+market&" # More specific query
                   f"apiKey={self.news_api_key}&"
                   f"pageSize={num_articles}&"
                   "language=en&"
                   "sortBy=relevancy") # Sort by relevance might be better than publishedAt

            logger.info(f"Fetching news for '{query_term}' from NewsAPI")
            response = requests.get(url, timeout=10) # 10 second timeout

            if response.status_code == 200:
                data = response.json()
                articles = [a['title'] for a in data.get('articles', []) if a.get('title')]
                logger.info(f"Retrieved {len(articles)} news articles for {query_term}")
                return articles if articles else None
            else:
                logger.warning(f"NewsAPI error for {query_term}: Status {response.status_code}, Message: {response.text}")
                return None

        except requests.exceptions.Timeout: logger.warning(f"NewsAPI request timed out for {symbol}"); return None
        except requests.exceptions.RequestException as e: logger.warning(f"NewsAPI request failed for {symbol}: {e}"); return None
        except Exception as e: logger.error(f"Unexpected error fetching news for {symbol}: {e}"); return None


    # Portfolio generation methods (kept as is)
    def create_personalized_portfolio(self, risk_appetite, time_period_months, budget):
        """Create a personalized portfolio using risk-based position sizing."""
        # --- THIS METHOD REMAINS THE SAME ---
        try:
            min_score = 65 # Filter threshold
            symbols = self.get_all_stock_symbols()
            stock_results = []
            logger.info(f"Analyzing {len(symbols)} stocks for position portfolio...")
            # Analyze eligible stocks
            for i, symbol in enumerate(symbols):
                 logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})...")
                 result = self.analyze_position_trading_stock(symbol) # Uses the updated method now
                 if result and result.get('position_score', 0) >= min_score and \
                    result.get('trading_plan', {}).get('entry_signal') in ['BUY', 'STRONG BUY']:
                     stock_results.append(result)

            if not stock_results:
                 logger.warning("No stocks met the minimum criteria for the position portfolio.")
                 return {"portfolio": {}, "summary": {"error": "No suitable stocks found."}}


            # Sort by score and select top N based on config
            max_positions = self.position_trading_params.get('max_positions', 10) # Get max positions from params
            sorted_stocks = sorted(stock_results, key=lambda x: x['position_score'], reverse=True)
            selected_stocks = sorted_stocks[:max_positions]
            logger.info(f"Selected top {len(selected_stocks)} stocks based on score and max_positions.")


            portfolio = self.calculate_position_sizes(selected_stocks, budget)
            if not portfolio:
                 logger.error("Failed to calculate position sizes.")
                 return {"portfolio": {}, "summary": {"error": "Could not allocate budget based on risk parameters."}}


            summary = self.generate_portfolio_summary(portfolio, time_period_months, budget) # Pass budget to summary

            return {'portfolio': portfolio, 'summary': summary,
                    'risk_profile': risk_appetite, 'time_period_months': time_period_months, 'budget': budget}

        except Exception as e:
            logger.error(f"Error creating personalized position portfolio: {e}", exc_info=True)
            return {"portfolio": {}, "summary": {"error": f"Internal error: {e}"}}


    def calculate_position_sizes(self, selected_stocks, total_capital):
        """Calculate position sizes based on fixed risk percentage."""
         # --- THIS METHOD REMAINS THE SAME ---
        portfolio = {}
        risk_per_trade_pct = self.position_trading_params['risk_per_trade'] # e.g., 0.01 for 1%
        capital_at_risk_per_trade = total_capital * risk_per_trade_pct
        max_total_risk = total_capital * self.position_trading_params['max_portfolio_risk']

        total_allocated = 0
        current_total_risk = 0

        logger.info(f"Calculating position sizes. Risk per trade: {risk_per_trade_pct*100:.1f}%, Capital at risk/trade: {capital_at_risk_per_trade:.2f}")

        for stock_data in selected_stocks:
            try:
                current_price = stock_data.get('current_price')
                stop_loss = stock_data.get('trading_plan', {}).get('stop_loss')

                # Strict validation for sizing
                if not isinstance(current_price, (int, float)) or current_price <= 0 or \
                   not isinstance(stop_loss, (int, float)) or stop_loss <= 0 or \
                   current_price <= stop_loss:
                    logger.warning(f"Skipping {stock_data.get('symbol')}: Invalid price ({current_price}) or stop loss ({stop_loss}).")
                    continue

                risk_per_share = current_price - stop_loss
                num_shares = int(capital_at_risk_per_trade / risk_per_share)

                if num_shares == 0:
                     logger.warning(f"Skipping {stock_data.get('symbol')}: Cannot afford even one share with risk {risk_per_share:.2f} per share.")
                     continue

                investment_amount = num_shares * current_price
                trade_risk = num_shares * risk_per_share # Actual capital risked on this trade

                # Check budget and total portfolio risk constraints
                if total_allocated + investment_amount > total_capital:
                     logger.info(f"Stopping allocation for {stock_data.get('symbol')}: Exceeds total budget.")
                     break # Stop adding stocks if budget exceeded
                if current_total_risk + trade_risk > max_total_risk:
                     logger.info(f"Stopping allocation for {stock_data.get('symbol')}: Exceeds max portfolio risk.")
                     break # Stop adding stocks if max portfolio risk exceeded


                total_allocated += investment_amount
                current_total_risk += trade_risk
                symbol = stock_data.get('symbol', 'Unknown')

                portfolio[symbol] = {
                    'company_name': stock_data.get('company_name', 'Unknown'),
                    'sector': stock_data.get('sector', 'Unknown'),
                    'score': round(stock_data.get('position_score', 0), 2),
                    'num_shares': num_shares,
                    'entry_price': round(current_price, 2), # Add entry price for clarity
                    'investment_amount': round(investment_amount, 2),
                    'stop_loss': round(stop_loss, 2),
                    'trade_risk_amount': round(trade_risk, 2), # Add risk amount
                    'targets': stock_data.get('trading_plan', {}).get('targets')
                }
                logger.info(f"Allocated {investment_amount:.2f} to {symbol} ({num_shares} shares). Trade risk: {trade_risk:.2f}")

            except Exception as e:
                logger.error(f"Error sizing position for {stock_data.get('symbol', 'N/A')}: {e}")
                continue
                
        logger.info(f"Total allocated: {total_allocated:.2f}, Total risk: {current_total_risk:.2f}")
        return portfolio


    def generate_portfolio_summary(self, portfolio, time_period_months, budget):
        """Generate a summary of the created portfolio."""
         # --- THIS METHOD REMAINS THE SAME ---
        if not portfolio: return {"error": "Portfolio is empty."} # Handle empty portfolio case

        total_investment = sum(stock['investment_amount'] for stock in portfolio.values())
        total_risk = sum(stock['trade_risk_amount'] for stock in portfolio.values()) # Sum actual trade risk
        
        num_stocks = len(portfolio)
        avg_score = sum(stock['score'] for stock in portfolio.values()) / num_stocks if num_stocks > 0 else 0

        # Sector allocation %
        sector_allocation = {}
        for stock in portfolio.values():
            sector = stock.get('sector', 'Unknown')
            sector_allocation[sector] = sector_allocation.get(sector, 0) + stock['investment_amount']
            
        sector_allocation_pct = {s: (a / total_investment * 100) for s, a in sector_allocation.items()} if total_investment > 0 else {}


        # Simplified expected return projection
        # Base annual return assumption for a score of 100
        base_annual_return_perfect_score = 0.20 # Assume 20% annual for A+ score
        # Scale based on average score
        projected_annual_return = base_annual_return_perfect_score * (avg_score / 100)
        # Adjust for time period (simple linear scaling - adjust if needed)
        time_years = time_period_months / 12
        total_expected_return_pct = projected_annual_return * time_years * 100


        return {
            'total_budget': budget,
            'total_investment': round(total_investment, 2),
            'remaining_cash': round(budget - total_investment, 2),
            'total_portfolio_risk': round(total_risk, 2),
            'total_portfolio_risk_pct': round((total_risk / budget) * 100, 2) if budget > 0 else 0,
            'number_of_stocks': num_stocks,
            'average_score': round(avg_score, 2),
            'sector_allocation_pct': {s: round(p, 1) for s, p in sector_allocation_pct.items()},
            'projected_return_pct': round(total_expected_return_pct, 2), # Renamed for clarity
            'recommended_holding_period': f"{time_period_months} months"
        }


    # Sample data methods (kept as is)
    def get_sample_mda_analysis(self, symbol):
        """Generate sample MDA analysis."""
         # --- THIS HELPER REMAINS THE SAME ---
        try:
            base_score = 50 + (hash(symbol) % 31) - 15 # More variability around 50
            management_tone = "Neutral"
            if base_score >= 70: management_tone = "Very Optimistic"
            elif base_score >= 60: management_tone = "Optimistic"
            elif base_score <= 40: management_tone = "Pessimistic"
            elif base_score <= 30: management_tone = "Very Pessimistic"

            return {'mda_score': base_score, 'management_tone': management_tone, 'confidence': 0.75 + (hash(symbol)%10)/100, # Slight variation
                    'analysis_method': 'Sample MDA Analysis (Fallback)'}
        except Exception as e:
            logger.error(f"Error generating sample MDA: {e}")
            return {'mda_score': 50, 'management_tone': 'Neutral', 'analysis_method': 'Error'}


    def get_sample_news(self, symbol):
        """Generate sample news articles."""
         # --- THIS HELPER REMAINS THE SAME ---
        try:
             base_symbol = str(symbol).split('.')[0]
             stock_info = self.get_stock_info_from_db(base_symbol)
             company_name = stock_info.get("name", base_symbol)
             positive_news = [f"{company_name} reports strong earnings", f"Analysts upgrade {company_name}", f"{company_name} announces expansion"]
             negative_news = [f"Concerns over {company_name}'s debt levels", f"{company_name} faces regulatory hurdles", f"Sector outlook weakens for {company_name}"]
             neutral_news = [f"{company_name} maintains market share", f"Management change at {company_name}", f"General market update affects {company_name}"]
             # Mix them based on hash for some variety
             idx = hash(symbol) % 3
             if idx == 0: return positive_news * 2 + negative_news + neutral_news * 2 # Mostly positive
             elif idx == 1: return negative_news * 2 + positive_news + neutral_news * 2 # Mostly negative
             else: return neutral_news * 3 + positive_news + negative_news # Mostly neutral
        except Exception as e:
             logger.error(f"Error generating sample news: {e}")
             return [f"News item for {symbol}"]


# Configure logging if run directly (useful for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Simple Test Example (Requires data_providers.py and config.py) ---
    # try:
    #     # You would need to initialize a StockDataProvider here for a real test
    #     # from data_providers import StockDataProvider
    #     # from symbol_mapper import SymbolMapper
    #     # mapper = SymbolMapper()
    #     # provider = StockDataProvider(fyers_app_id=config.FYERS_APP_ID, fyers_access_token=config.FYERS_ACCESS_TOKEN, symbol_mapper=mapper, redis_url=config.REDIS_URL)
        
    #     # Mock provider for basic test without credentials
    #     class MockDataProvider:
    #          def get_stock_data(*args, **kwargs):
    #              # Return minimal data structure
    #              sample_ohlcv = [{'date': (datetime.now()-timedelta(days=i)).strftime('%Y-%m-%d'), 'open': 100+i, 'high':105+i, 'low':95+i, 'close':102+i, 'volume':10000} for i in range(252*5)]
    #              sample_ohlcv.reverse()
    #              return {'symbol': 'RELIANCE', 'company_name': 'Reliance Industries', 'exchange_used': 'NSE',
    #                      'ohlcv': sample_ohlcv,
    #                      'fundamentals': {'pe_ratio': 25.0, 'dividend_yield': 0.01, 'roe': 0.15, 'debt_to_equity': 0.5},
    #                      'errors': []}
                     
    #     mock_provider = MockDataProvider()
        
    #     position_system = EnhancedPositionTradingSystem(data_provider=mock_provider)
    #     analysis = position_system.analyze_position_trading_stock("RELIANCE")
        
    #     if analysis:
    #         print("\n--- Analysis Result ---")
    #         # print(json.dumps(analysis, indent=2)) # Requires custom JSON encoder for pandas/numpy types
    #         print(f"Symbol: {analysis.get('symbol')}")
    #         print(f"Score: {analysis.get('position_score')}")
    #         print(f"Signal: {analysis.get('trading_plan', {}).get('entry_signal')}")
    #         print(f"Stop Loss: {analysis.get('trading_plan', {}).get('stop_loss')}")
    #         print(f"Target 2: {analysis.get('trading_plan', {}).get('targets', {}).get('target_2')}")
    #     else:
    #         print("Analysis failed.")

    # except ImportError:
    #      print("Could not import dependencies for test.")
    # except Exception as e:
    #      print(f"An error occurred during test: {e}")
    #      traceback.print_exc()
