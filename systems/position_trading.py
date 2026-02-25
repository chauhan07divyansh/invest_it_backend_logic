import os
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from textblob import TextBlob
import warnings
import requests
import json
import concurrent.futures

# Local application imports
import config
from hf_utils import query_hf_api
from position_trading_system_db import StockDBMixin

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedPositionTradingSystem(StockDBMixin):
    def __init__(self, data_provider=None, mda_processor=None, redis_client=None):
        try:
            self.event_registry_api_key = getattr(config, "EVENT_REGISTRY_API_KEY", None)
            self.event_registry_endpoint = getattr(
                config,
                "EVENT_REGISTRY_ENDPOINT",
                "https://eventregistry.org/api/v1/article/getArticles"
            )

            self.position_trading_params = config.POSITION_TRADING_PARAMS
            self._validate_trading_params()

            # Initialize Expanded Database (500+ Stocks)
            self._init_stock_db()

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

            # --- Store MDA processor (MDAProcessor from mda_processor.py) ---
            self.mda_processor = mda_processor
            if mda_processor:
                logger.info("✅ MDA Processor injected into PositionTradingSystem")
            else:
                logger.warning("⚠️ No MDA Processor - will use sample data")

            # Redis Cache
            self.redis_client = redis_client
            self.cache_ttl = 86400  # 24 hours

            # --- Session for HTTP requests ---
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

    def _analyze_sentiment_via_api(self, articles: list):
        """Analyze sentiment via SBERT HuggingFace API."""
        try:
            payload = {"inputs": articles}
            api_response = query_hf_api(self.sentiment_api_url, payload)

            if not api_response or "results" not in api_response:
                raise ValueError("Invalid SBERT API response")

            results = api_response.get("results") or api_response
            if not isinstance(results, list):
                raise ValueError("Unexpected SBERT API response format")

            sentiments = []
            confidences = []

            for r in results:
                sentiments.append(str(r.get("label", "neutral")).lower())
                confidences.append(float(r.get("confidence", 0.5)))

            return sentiments, confidences

        except Exception as e:
            logger.error(f"SBERT API parsing failed: {e}")
            return None

    def _analyze_mda_via_api(self, mda_texts: list) -> dict | None:
        """
        Legacy: Analyzes MDA text by calling the remote MDA HuggingFace API.
        Used only when MDAProcessor is not injected.
        """
        try:
            payload = {"inputs": mda_texts}
            api_results = query_hf_api(self.mda_api_url, payload)

            if api_results is None:
                raise ValueError("API call to MDA HF Space failed or returned no data.")

            if isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
                api_results = api_results[0]

            sentiments = [res.get('label') for res in api_results]
            confidences = [res.get('score') for res in api_results]

            sentiment_scores = []
            valid_confidences = []
            for sentiment, confidence in zip(sentiments, confidences):
                if confidence is None:
                    continue
                valid_confidences.append(confidence)
                sentiment_str = str(sentiment).lower()
                if sentiment_str in ['positive', 'very_positive', 'label_4', 'label_3']:
                    sentiment_scores.append(confidence)
                elif sentiment_str in ['negative', 'very_negative', 'label_0', 'label_1']:
                    sentiment_scores.append(-confidence)
                else:
                    sentiment_scores.append(0)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            mda_score = 50 + (avg_sentiment * 50)
            mda_score = max(0, min(100, mda_score))

            if mda_score >= 70:
                management_tone = "Very Optimistic"
            elif mda_score >= 60:
                management_tone = "Optimistic"
            elif mda_score <= 30:
                management_tone = "Very Pessimistic"
            elif mda_score <= 40:
                management_tone = "Pessimistic"
            else:
                management_tone = "Neutral"

            return {
                'mda_score': mda_score,
                'management_tone': management_tone,
                'confidence': np.mean(valid_confidences) if valid_confidences else 0,
                'analysis_method': 'Remote PyTorch BERT MDA Model (API)',
            }
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logging.error(
                f"Could not parse MDA API response. Error: {e}. "
                f"Response: {api_results if 'api_results' in locals() else 'unknown'}"
            )
            return None

    # ==============================================================================
    #  CORE ANALYSIS METHODS
    # ==============================================================================

    def analyze_news_sentiment(self, symbol, num_articles=20):
        """Fetches news and analyzes sentiment using API or TextBlob fallback."""
        try:
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Event Registry (newsapi.ai)"

            if not articles:
                articles = self.get_sample_news(symbol)
                news_source = "Sample news"
                logger.warning(f"Using sample news for {symbol}")

            if not articles:
                return [], [], [], "No Analysis", "No Source"

            if self.sentiment_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    logger.info(f"Analyzed news for {symbol} using SBERT API.")
                    return sentiments, articles, confidences, "SBERT API", news_source

            logging.warning(f"Falling back to TextBlob for news sentiment for {symbol}.")
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob Fallback", news_source

        except Exception as e:
            logging.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error Source"

    def _validate_trading_params(self):
        """Validate position trading parameters from config."""
        try:
            required_params = [
                'min_holding_period', 'max_holding_period', 'risk_per_trade',
                'max_portfolio_risk', 'profit_target_multiplier', 'max_positions',
                'fundamental_weight', 'technical_weight', 'sentiment_weight', 'mda_weight'
            ]

            missing = [p for p in required_params if p not in self.position_trading_params]
            if missing:
                raise ValueError(f"Missing required trading parameters: {', '.join(missing)}")

            for param in required_params:
                value = self.position_trading_params[param]
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"Invalid trading parameter {param}: {value}. Must be a non-negative number."
                    )
                if param.endswith('_weight') and value > 1.0:
                    raise ValueError(f"Weight parameter {param} cannot exceed 1.0: {value}")

            if self.position_trading_params['min_holding_period'] >= self.position_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")
            if self.position_trading_params['risk_per_trade'] > 0.05:
                logger.warning("risk_per_trade exceeds 5%, which is high for position trading.")
            if self.position_trading_params['max_positions'] <= 0:
                raise ValueError("max_positions must be greater than 0")

            total_weight = sum(
                self.position_trading_params[p] for p in required_params if p.endswith('_weight')
            )
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Scoring weights do not sum to 1.0: {total_weight:.3f}")

            logger.info("✅ Position trading parameters validated successfully")
        except Exception as e:
            logger.error(f"Error validating trading parameters: {e}")
            raise

    def updated_analyze_mda_sentiment(self, symbol):
        """
        Gets MD&A analysis. Priority order:
        1. Injected MDAProcessor (PDF-extracted, high quality)
        2. Legacy HF API (if configured)
        3. Sample data fallback
        """
        try:
            # Priority 1: Injected MDAProcessor (mda_processor.py)
            if self.mda_processor:
                analysis = self.mda_processor.get_mda_analysis(symbol)
                if analysis:
                    logger.info(f"✅ Retrieved MDA analysis for {symbol} from MDAProcessor")
                    return analysis
                else:
                    logger.warning(f"⚠️ MDAProcessor has no entry for {symbol}. Check if PDF was ingested.")

            # Priority 2: Legacy HF API
            if self.mda_api_available:
                logger.info(f"Attempting legacy HF MDA API for {symbol}")
                sample_texts = self.get_sample_mda_texts(symbol)
                if sample_texts:
                    api_result = self._analyze_mda_via_api(sample_texts)
                    if api_result:
                        return api_result

            # Priority 3: Sample fallback
            logger.warning(f"Using sample MDA data for {symbol}")
            return self.get_sample_mda_analysis(symbol)

        except Exception as e:
            logger.error(f"MD&A analysis error for {symbol}: {e}")
            return self.get_sample_mda_analysis(symbol)

    def calculate_position_trading_score(
        self,
        data,
        sentiment_data,
        fundamentals,
        trends,
        market_analysis,
        sector,
        mda_analysis=None
    ):
        """
        Calculate final Position Trading score (0–100).
        Optimized for long-term holding (6–24 months).
        """
        try:
            # ── 1. WEIGHTS ──────────────────────────────────────────────────
            weights = {
                "fundamental": self.position_trading_params["fundamental_weight"],
                "technical":   self.position_trading_params["technical_weight"],
                "sentiment":   self.position_trading_params["sentiment_weight"],
                "mda":         self.position_trading_params["mda_weight"],
            }

            # ── 2. BASE SCORES ───────────────────────────────────────────────
            fundamental_score = self.calculate_fundamental_score(fundamentals, sector)
            technical_score   = self.calculate_technical_score_position(data)
            sentiment_score   = self.calculate_sentiment_score(sentiment_data)
            mda_score         = mda_analysis.get("mda_score", 50) if mda_analysis else 50

            # ── 3. CONTEXTUAL SENTIMENT ADJUSTMENT ──────────────────────────
            sentiment_multiplier = self._get_sentiment_bias_from_db(sector)
            sentiment_score = min(100, sentiment_score * sentiment_multiplier)

            # ── 4. WEIGHTED BASE SCORE ───────────────────────────────────────
            base_score = (
                fundamental_score * weights["fundamental"]
                + technical_score * weights["technical"]
                + sentiment_score * weights["sentiment"]
                + mda_score       * weights["mda"]
            )

            # ── 5. TREND & MARKET MODIFIERS ──────────────────────────────────
            trend_score  = trends.get("trend_score", 50) / 100
            sector_score = market_analysis.get("sector_score", 60) / 100
            modifier     = 0.75 + (0.15 * trend_score) + (0.10 * sector_score)
            final_score  = base_score * modifier

            # ── 6. RISK & VOLATILITY PENALTY ─────────────────────────────────
            if data is not None and not data.empty and "Close" in data.columns:
                try:
                    volatility = data["Close"].pct_change().std() * (252 ** 0.5)
                    if volatility > 0.55:
                        final_score *= 0.65
                    elif volatility > 0.40:
                        final_score *= 0.80
                except Exception:
                    pass

            # ── 7. FUNDAMENTAL BONUS SIGNALS ─────────────────────────────────
            dividend_yield = fundamentals.get("dividend_yield") or fundamentals.get("expected_div_yield", 0)
            if isinstance(dividend_yield, (int, float)) and dividend_yield > 0.02:
                final_score *= 1.05

            if trends.get("momentum_1y", 0) > 0.20 and trends.get("momentum_6m", 0) > 0:
                final_score *= 1.04

            # ── 8. MANAGEMENT TONE IMPACT (MDA) ──────────────────────────────
            if mda_analysis:
                tone = mda_analysis.get("management_tone", "Neutral")
                if tone == "Very Optimistic":
                    final_score *= 1.08
                elif tone == "Optimistic":
                    final_score *= 1.04
                elif tone == "Pessimistic":
                    final_score *= 0.92
                elif tone == "Very Pessimistic":
                    final_score *= 0.85

            # ── 9. MDA QUALITY BONUS (from PDF extractor) ────────────────────
            if mda_analysis:
                fls_count        = mda_analysis.get("fls_count", 0)
                uncertainty_den  = mda_analysis.get("avg_uncertainty_density", 0)
                extraction_ok    = mda_analysis.get("extraction_status", "") == "success"

                # Reward rich forward-looking statements (up to +3%)
                if fls_count >= 10 and extraction_ok:
                    final_score *= 1.03
                elif fls_count >= 5 and extraction_ok:
                    final_score *= 1.01

                # Penalise extreme uncertainty density (>0.05 is very high)
                if uncertainty_den > 0.05:
                    final_score *= 0.97

            # ── 10. FINAL CLAMP ───────────────────────────────────────────────
            return round(min(100, max(0, final_score)), 2)

        except Exception as e:
            logger.error(f"Position score calculation failed: {e}")
            return 0

    # ==============================================================================
    #  MAIN ANALYSIS FUNCTION
    # ==============================================================================

    def analyze_position_trading_stock(self, symbol, period="5y"):
        """Comprehensive position trading analysis using the injected data provider."""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting position trading analysis for {symbol}")

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
                logger.error(
                    f"Provider error for {symbol}: "
                    f"{stock_data.get('errors', 'Unknown data provider error')}"
                )
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
                data.rename(
                    columns={
                        'open': 'Open', 'high': 'High',
                        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                    },
                    inplace=True
                )
            except Exception as e:
                logger.error(f"Could not process OHLCV data for {symbol}: {e}")
                return None

            # 2. Unpack fundamentals
            fundamentals = stock_data.get('fundamentals', {})
            if not fundamentals:
                logger.warning(f"No fundamental data returned from provider for {symbol}")
                fundamentals = {}

            # 3. Unpack other info
            base_symbol  = stock_data.get('symbol', symbol.split('.')[0].upper())
            final_symbol = f"{base_symbol}.{stock_data.get('exchange_used', 'NSE')}"
            company_name = stock_data.get('company_name', base_symbol)

            # Supplement from internal DB
            stock_info         = self.get_stock_info_from_db(base_symbol)
            sector             = stock_info.get('sector', 'Unknown')
            market_cap_category = stock_info.get('market_cap', 'Unknown')
            if 'expected_div_yield' not in fundamentals:
                fundamentals['expected_div_yield'] = stock_info.get('div_yield', 0)

            # Current market data
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change     = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (
                        (price_change / data['Close'].iloc[-2]) * 100
                        if data['Close'].iloc[-2] != 0 else 0
                    )
                else:
                    price_change, price_change_pct = 0, 0
            except Exception as e:
                logger.error(f"Error calculating price changes: {e}")
                current_price, price_change, price_change_pct = 0, 0, 0

            # ── ANALYSIS STEPS ─────────────────────────────────────────────────

            try:
                trends = self.analyze_long_term_trends(data)
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                trends = {'trend_score': 50}

            try:
                market_analysis = self.analyze_market_cycles(base_symbol, data)
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                market_analysis = {'sector_score': 60}

            try:
                sentiment_results = self.analyze_news_sentiment(final_symbol)
            except Exception as e:
                logger.error(f"News sentiment error: {e}")
                sentiment_results = ([], [], [], "Error", "Error")

            try:
                mda_analysis = self.updated_analyze_mda_sentiment(base_symbol)
            except Exception as e:
                logger.error(f"MDA analysis error: {e}")
                mda_analysis = self.get_sample_mda_analysis(base_symbol)

            try:
                risk_metrics = self.calculate_risk_metrics(data)
            except Exception as e:
                logger.error(f"Risk metrics error: {e}")
                risk_metrics = {
                    'volatility': 0.3,
                    'atr': current_price * 0.02 if current_price else 0
                }

            try:
                position_score = self.calculate_position_trading_score(
                    data, sentiment_results, fundamentals,
                    trends, market_analysis, sector, mda_analysis
                )
            except Exception as e:
                logger.error(f"Score calculation error: {e}")
                position_score = 0

            try:
                trading_plan = self.generate_position_trading_plan(
                    data, position_score, risk_metrics, fundamentals, trends
                )
            except Exception as e:
                logger.error(f"Trading plan error: {e}")
                trading_plan = {'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed'}

            # ── Compile result ──────────────────────────────────────────────────
            result = {
                'symbol':              final_symbol,
                'company_name':        company_name,
                'sector':              sector,
                'market_cap_category': market_cap_category,
                'current_price':       current_price,
                'price_change':        price_change,
                'price_change_pct':    price_change_pct,

                'fundamentals':        fundamentals,
                'fundamental_score':   self.calculate_fundamental_score(fundamentals, sector),

                'trends':              trends,
                'trend_score':         trends.get('trend_score', 50),

                'market_analysis':     market_analysis,

                'rsi_30': (
                    self.calculate_rsi(data['Close'], period=30).iloc[-1]
                    if len(data) >= 31 else None
                ),
                'ma_50':  trends.get('ma_50'),
                'ma_100': trends.get('ma_100'),
                'ma_200': trends.get('ma_200'),

                'sentiment': {
                    'scores':            sentiment_results[0] if sentiment_results else [],
                    'articles':          sentiment_results[1] if sentiment_results else [],
                    'confidence':        sentiment_results[2] if sentiment_results else [],
                    'method':            sentiment_results[3] if sentiment_results else "N/A",
                    'source':            sentiment_results[4] if sentiment_results else "N/A",
                    'sentiment_summary': self.get_sentiment_summary(
                        sentiment_results[0] if sentiment_results else []
                    ),
                },

                'mda_sentiment':  mda_analysis,
                'risk_metrics':   risk_metrics,
                'position_score': position_score,
                'trading_plan':   trading_plan,

                'model_type':           self.model_type,
                'mda_model_available':  self.mda_api_available,
                'analysis_date':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type':        'Position Trading (Long-term with MDA)',
            }

            logger.info(f"✅ Successfully analyzed {symbol} with position score {position_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Critical error analyzing {symbol} for position trading: {e}")
            logger.error(traceback.format_exc())
            return None

    # ==============================================================================
    #  PARALLEL ANALYSIS
    # ==============================================================================

    def analyze_stocks_parallel(self, symbols: List[str], max_workers: int = 5) -> List[Dict]:
        """Analyze multiple stocks in parallel (IO-intensive: fetches fundamentals)."""
        try:
            logger.info(
                f"🚀 Starting parallel position analysis of {len(symbols)} stocks "
                f"with {max_workers} workers"
            )
            start_time = time.time()

            results      = []
            failed_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_position_trading_stock, symbol): symbol
                    for symbol in symbols
                }

                completed = 0
                total     = len(symbols)

                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol    = future_to_symbol[future]
                    completed += 1
                    try:
                        result = future.result(timeout=60)
                        if result and result.get('position_score', 0) > 0:
                            results.append(result)
                            logger.info(
                                f"✅ [{completed}/{total}] {symbol}: "
                                f"Score={result['position_score']:.0f}"
                            )
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ [{completed}/{total}] {symbol}: {e}")

            results.sort(key=lambda x: x.get('position_score', 0), reverse=True)

            elapsed = time.time() - start_time
            logger.info(
                f"✅ Parallel analysis complete: {len(results)} successful in {elapsed:.1f}s"
            )
            return results

        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            return []

            

            # Merge all stocks
            self.indian_stocks = {}
            for symbol, info in nifty_50.items():
                sector = info.get('sector', 'Default')
                self.indian_stocks[symbol] = {
                    'name': info.get('name', symbol), 'sector': sector,
                    'market_cap': 'Large',
                    'div_yield': sector_div_yields.get(sector, sector_div_yields['Default'])
                }
            for symbol, info in nifty_next_50.items():
                sector = info.get('sector', 'Default')
                self.indian_stocks[symbol] = {
                    'name': info.get('name', symbol), 'sector': sector,
                    'market_cap': 'Large',
                    'div_yield': sector_div_yields.get(sector, sector_div_yields['Default'])
                }
            for symbol, info in nifty_midcap_100.items():
                sector = info.get('sector', 'Default')
                self.indian_stocks[symbol] = {
                    'name': info.get('name', symbol), 'sector': sector,
                    'market_cap': 'Mid',
                    'div_yield': sector_div_yields.get(sector, sector_div_yields['Default'])
                }
            for symbol, info in smallcap_stocks.items():
                sector = info.get('sector', 'Default')
                self.indian_stocks[symbol] = {
                    'name': info.get('name', symbol), 'sector': sector,
                    'market_cap': 'Small',
                    'div_yield': sector_div_yields.get(sector, sector_div_yields['Default'])
                }

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

            large_cap_count = sum(1 for s in self.indian_stocks.values() if s['market_cap'] == 'Large')
            mid_cap_count   = sum(1 for s in self.indian_stocks.values() if s['market_cap'] == 'Mid')
            small_cap_count = sum(1 for s in self.indian_stocks.values() if s['market_cap'] == 'Small')

            logger.info(f"✅ Initialized internal stock database with {len(self.indian_stocks)} symbols")
            logger.info(f"   - Large Cap: {large_cap_count} stocks (Nifty 50 + Next 50)")
            logger.info(f"   - Mid Cap: {mid_cap_count} stocks (Midcap 100)")
            logger.info(f"   - Small Cap: {small_cap_count} stocks")

        except Exception as e:
            logger.error(f"Error initializing stock database: {e}")
            self.indian_stocks = {
                "RELIANCE": {"name": "Reliance Industries",       "sector": "Oil & Gas",              "market_cap": "Large", "div_yield": 0.045},
                "TCS":      {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.020},
                "HDFCBANK": {"name": "HDFC Bank",                 "sector": "Banking",                "market_cap": "Large", "div_yield": 0.010},
                "INFY":     {"name": "Infosys",                   "sector": "Information Technology", "market_cap": "Large", "div_yield": 0.020},
                "ICICIBANK":{"name": "ICICI Bank",                "sector": "Banking",                "market_cap": "Large", "div_yield": 0.010},
            }
            logger.warning(f"⚠️ Using fallback database with {len(self.indian_stocks)} stocks")

    # ==============================================================================
    #  CALCULATION METHODS
    # ==============================================================================

    def calculate_fundamental_score(self, fundamentals, sector):
        """Calculate fundamental score (0–100) based on provided fundamentals dict."""
        try:
            score     = 0
            max_score = 100

            def get_value(key, default=None):
                val = fundamentals.get(key, default)
                if val is None or not isinstance(val, (int, float)):
                    return default
                return val

            # P/E Ratio (20 pts)
            pe_ratio = get_value('pe_ratio')
            if pe_ratio is not None:
                if 10 <= pe_ratio <= 25:   score += 20
                elif 8 <= pe_ratio < 10 or 25 < pe_ratio <= 35: score += 15
                elif 5 <= pe_ratio < 8  or 35 < pe_ratio <= 45: score += 10
                elif pe_ratio > 0:          score += 5

            # PEG Ratio (15 pts)
            peg_ratio = get_value('peg_ratio')
            if peg_ratio is not None and peg_ratio > 0:
                if 0.5 <= peg_ratio <= 1.0:   score += 15
                elif 1.0 < peg_ratio <= 1.5:  score += 12
                elif 1.5 < peg_ratio <= 2.0:  score += 8
                elif peg_ratio < 0.5 or (2.0 < peg_ratio <= 3.0): score += 5

            # Revenue Growth (15 pts)
            rev_growth = get_value('revenue_growth') or get_value('sales_growth_3y')
            if rev_growth is not None:
                if rev_growth >= 20:   score += 15
                elif rev_growth >= 15: score += 12
                elif rev_growth >= 10: score += 9
                elif rev_growth >= 5:  score += 6
                elif rev_growth >= 0:  score += 3

            # Earnings Growth (15 pts)
            earn_growth = get_value('earnings_growth') or get_value('profit_growth_3y')
            if earn_growth is not None:
                if earn_growth >= 25:   score += 15
                elif earn_growth >= 20: score += 12
                elif earn_growth >= 15: score += 9
                elif earn_growth >= 10: score += 6
                elif earn_growth >= 0:  score += 3

            # ROE (12 pts)
            roe = get_value('roe')
            if roe is not None:
                if roe >= 20:   score += 12
                elif roe >= 15: score += 10
                elif roe >= 12: score += 7
                elif roe >= 8:  score += 4
                elif roe > 0:   score += 2

            # Debt to Equity (10 pts)
            debt_equity = get_value('debt_to_equity')
            if debt_equity is not None:
                if debt_equity < 0.3:   score += 10
                elif debt_equity < 0.5: score += 8
                elif debt_equity < 0.8: score += 6
                elif debt_equity < 1.2: score += 4
                elif debt_equity < 2.0: score += 2

            # Profit Margin (8 pts)
            profit_margin = get_value('profit_margin')
            if profit_margin is not None:
                if profit_margin >= 15:  score += 8
                elif profit_margin >= 10: score += 6
                elif profit_margin >= 7:  score += 4
                elif profit_margin > 0:   score += 2

            # Operating Margin (7 pts)
            op_margin = get_value('operating_margin')
            if op_margin is not None:
                if op_margin >= 20:   score += 7
                elif op_margin >= 15: score += 5
                elif op_margin >= 10: score += 3
                elif op_margin > 0:   score += 1

            # Dividend Yield (10 pts)
            div_yield = get_value('dividend_yield')
            if div_yield is not None and div_yield > 0:
                if div_yield >= 3.0:   score += 10
                elif div_yield >= 2.0: score += 8
                elif div_yield >= 1.5: score += 6
                elif div_yield >= 1.0: score += 4
                elif div_yield > 0:    score += 2
            else:
                expected_div = get_value('expected_div_yield', 0)
                if expected_div > 0:
                    div_yield_pct = expected_div * 100
                    if div_yield_pct >= 3.0:   score += 10
                    elif div_yield_pct >= 2.0: score += 7
                    elif div_yield_pct >= 1.0: score += 5
                    elif div_yield_pct > 0:    score += 2

            # Current Ratio (5 pts)
            current_ratio = get_value('current_ratio')
            if current_ratio is not None:
                if current_ratio >= 2.0:   score += 5
                elif current_ratio >= 1.5: score += 4
                elif current_ratio >= 1.2: score += 3
                elif current_ratio >= 1.0: score += 2

            # Price to Book (3 pts)
            pb_ratio = get_value('price_to_book')
            if pb_ratio is not None and pb_ratio > 0:
                if pb_ratio < 1.5:   score += 3
                elif pb_ratio < 2.5: score += 2
                elif pb_ratio < 4.0: score += 1

            return min(score, max_score)

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0

    def analyze_long_term_trends(self, data):
        """Analyze long-term trends from OHLCV DataFrame."""
        try:
            if data is None or data.empty or len(data) < 200:
                logger.warning("Insufficient data for full trend analysis.")
                return {
                    'trend_score': 25, 'ma_50_slope': 0, 'ma_200_slope': 0,
                    'above_ma_200': False, 'momentum_6m': 0, 'momentum_1y': 0,
                    'ma_50': None, 'ma_100': None, 'ma_200': None
                }

            ma_50  = self.safe_rolling_calculation(data['Close'], 50,  'mean').iloc[-1]
            ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]
            current_price = data['Close'].iloc[-1]

            trend_score = 25
            if not any(pd.isna([ma_50, ma_100, ma_200, current_price])):
                if current_price > ma_50 > ma_100 > ma_200:   trend_score = 100
                elif current_price > ma_50 > ma_100:           trend_score = 75
                elif current_price > ma_100:                   trend_score = 50
                elif current_price < ma_50 < ma_100 < ma_200: trend_score = 0

            ma_50_series  = self.safe_rolling_calculation(data['Close'], 50,  'mean')
            ma_200_series = self.safe_rolling_calculation(data['Close'], 200, 'mean')

            ma_50_slope = (
                (ma_50_series.iloc[-1] - ma_50_series.iloc[-21]) / ma_50_series.iloc[-21]
                if len(ma_50_series) > 20 and ma_50_series.iloc[-21] != 0 else 0
            )
            ma_200_slope = (
                (ma_200_series.iloc[-1] - ma_200_series.iloc[-51]) / ma_200_series.iloc[-51]
                if len(ma_200_series) > 50 and ma_200_series.iloc[-51] != 0 else 0
            )

            momentum_6m = 0
            momentum_1y = 0
            if len(data) > 126:
                price_6m_ago = data['Close'].iloc[-126]
                if price_6m_ago != 0:
                    momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            if len(data) > 252:
                price_1y_ago = data['Close'].iloc[-252]
                if price_1y_ago != 0:
                    momentum_1y = (current_price - price_1y_ago) / price_1y_ago

            return {
                'trend_score':   trend_score,
                'ma_50_slope':   ma_50_slope,
                'ma_200_slope':  ma_200_slope,
                'above_ma_200':  current_price > ma_200 if not pd.isna(ma_200) else False,
                'momentum_6m':   momentum_6m,
                'momentum_1y':   momentum_1y,
                'ma_50':         ma_50,
                'ma_100':        ma_100,
                'ma_200':        ma_200,
            }
        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {e}")
            return {
                'trend_score': 50, 'ma_50_slope': 0, 'ma_200_slope': 0,
                'above_ma_200': False, 'momentum_6m': 0, 'momentum_1y': 0,
                'ma_50': None, 'ma_100': None, 'ma_200': None
            }

    def analyze_market_cycles(self, symbol, data):
        """Analyze market cycles and sector rotation."""
        try:
            stock_info   = self.get_stock_info_from_db(symbol)
            sector       = stock_info.get('sector', 'Unknown')
            sector_score = self._get_sector_score_from_db(sector)

            return {
                'sector_score':      sector_score,
                'sector':            sector,
                'cycle_stage':       self.determine_market_cycle(data),
                'sector_preference': self.get_sector_preference(sector),
            }
        except Exception as e:
            logger.error(f"Error in market cycle analysis: {e}")
            return {'sector_score': 60, 'sector': 'Unknown', 'cycle_stage': 'Unknown', 'sector_preference': 'Neutral'}

    def determine_market_cycle(self, data):
        """Determine current market cycle stage based on MA200."""
        try:
            if data is None or data.empty or len(data) < 200:
                return "Unknown"

            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')
            if ma_200.isna().all():
                return "Unknown"

            check_period      = min(120, len(data))
            above_ma_200_days = (data['Close'].iloc[-check_period:] > ma_200.iloc[-check_period:]).sum()
            above_ma_200_pct  = above_ma_200_days / check_period

            if above_ma_200_pct > 0.75:   return "Bull Market"
            elif above_ma_200_pct < 0.25: return "Bear Market"
            else:                          return "Transitional"
        except Exception as e:
            logger.error(f"Error determining market cycle: {e}")
            return "Unknown"

    def get_sector_preference(self, sector):
        """Get sector preference for position trading."""
        high_preference   = ['Consumer Goods', 'Information Technology', 'Healthcare',
                             'Pharmaceuticals', 'Power', 'Banking']
        medium_preference = ['Telecommunications', 'Oil & Gas', 'Chemicals',
                             'Cement', 'Financial Services']
        if sector in high_preference:   return 'High'
        elif sector in medium_preference: return 'Medium'
        else:                            return 'Low'

    def calculate_technical_score_position(self, data):
        """Calculate technical score optimized for position trading."""
        try:
            if data is None or data.empty or len(data) < 52:
                logger.warning("Insufficient data for full technical score.")
                return 25

            technical_score = 0
            current_price   = data['Close'].iloc[-1]

            # RSI(30) — 30 pts
            if len(data) >= 31:
                rsi = self.calculate_rsi(data['Close'], period=30).iloc[-1]
                if not pd.isna(rsi):
                    if 40 <= rsi <= 60:   technical_score += 30
                    elif 30 <= rsi < 40:  technical_score += 25
                    elif 60 < rsi <= 70:  technical_score += 20
                    elif rsi < 30:        technical_score += 15

            # Moving Averages — 25 pts
            if len(data) >= 200:
                ma_50  = self.safe_rolling_calculation(data['Close'], 50,  'mean').iloc[-1]
                ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
                ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]
                if not any(pd.isna([ma_50, ma_100, ma_200, current_price])):
                    if current_price > ma_50 > ma_100 > ma_200: technical_score += 25
                    elif current_price > ma_50 > ma_100:        technical_score += 20
                    elif current_price > ma_200:                 technical_score += 15
                    elif ma_50 > ma_100:                         technical_score += 10

            # Volume Trend — 15 pts
            if 'Volume' in data.columns and len(data) >= 50:
                recent_vol = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                long_vol   = self.safe_rolling_calculation(data['Volume'], 50, 'mean').iloc[-1]
                if not pd.isna(recent_vol) and not pd.isna(long_vol) and long_vol > 0:
                    volume_ratio = recent_vol / long_vol
                    if volume_ratio > 1.2:   technical_score += 15
                    elif volume_ratio > 1.0: technical_score += 10

            # Long-term MACD (26, 52, 18) — 20 pts
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'], fast=26, slow=52, signal=18)
            if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) and not pd.isna(signal_line.iloc[-1]):
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    technical_score += 15
                if len(histogram) > 1 and not pd.isna(histogram.iloc[-1]) and not pd.isna(histogram.iloc[-2]):
                    if histogram.iloc[-1] > histogram.iloc[-2]:
                        technical_score += 5

            # Support/Resistance — 15 pts
            support, resistance = self.calculate_support_resistance(data, window=50)
            if (support and resistance
                    and not any(pd.isna([support, resistance, current_price]))
                    and support > 0 and current_price > 0):
                dist_support_pct = (current_price - support) / support
                if 0.05 <= dist_support_pct <= 0.20: technical_score += 15
                elif dist_support_pct > 0.20:         technical_score += 10
                elif dist_support_pct < 0.05:         technical_score += 8

            return min(100, max(0, technical_score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0

    def calculate_sentiment_score(self, sentiment_data):
        """Calculate sentiment score from news analysis results."""
        try:
            if not sentiment_data or len(sentiment_data) != 5 or not sentiment_data[0]:
                return 50

            sentiments, _, confidences, _, _ = sentiment_data

            if len(sentiments) != len(confidences):
                logger.warning("Sentiment/confidence length mismatch")

            sentiment_value = 0.0
            total_weight    = 0.0

            for sentiment, confidence in zip(sentiments, confidences):
                weight = (
                    float(confidence)
                    if isinstance(confidence, (int, float)) and not pd.isna(confidence)
                    else 0.5
                )
                sentiment_str = str(sentiment).lower()
                if 'positive' in sentiment_str:
                    sentiment_value += weight
                elif 'negative' in sentiment_str:
                    sentiment_value -= weight
                total_weight += weight

            if total_weight > 0:
                normalized_sentiment = sentiment_value / total_weight
                sentiment_score = 50 + (normalized_sentiment * 50)
            else:
                sentiment_score = 50

            return min(100, max(0, sentiment_score))

        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 50

    def generate_position_trading_plan(self, data, score, risk_metrics, fundamentals, trends):
        """Generate realistic position trading plan."""
        default_plan = {
            'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed',
            'entry_timing': 'Unknown', 'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'support': 0, 'resistance': 0, 'holding_period': 'Unknown',
            'trade_management_note': 'Unable to generate plan',
            'stop_distance_pct': 0, 'upside_potential': 0, 'risk_reward_ratio': 0,
        }
        try:
            if data is None or data.empty:
                return default_plan
            current_price = data['Close'].iloc[-1]
            if current_price <= 0:
                return default_plan

            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02

            if score >= 80:
                entry_signal, entry_strategy, holding_period = (
                    "STRONG BUY", "Accumulate on dips; high conviction", "12-24 months"
                )
            elif score >= 65:
                entry_signal, entry_strategy, holding_period = (
                    "BUY", "Enter gradually; good prospects", "9-18 months"
                )
            elif score >= 40:
                entry_signal, entry_strategy, holding_period = (
                    "HOLD/WATCH", "Wait for better entry/confirmation", "Monitor"
                )
            else:
                entry_signal, entry_strategy, holding_period = (
                    "AVOID", "Weaknesses detected", "Not recommended"
                )

            # Stop Loss
            stop_loss_distance = min(atr * 3.0, current_price * 0.10)
            stop_loss = max(current_price - stop_loss_distance, 0)

            # Support/Resistance
            support, resistance = self.calculate_support_resistance(data, window=100)
            support    = support    if support    and not pd.isna(support)    else current_price * 0.90
            resistance = resistance if resistance and not pd.isna(resistance) else current_price * 1.15

            if support > 0:
                stop_loss = max(stop_loss, support * 0.97)
            stop_loss = max(stop_loss, current_price * 0.85)

            # Targets
            risk_per_share = current_price - stop_loss
            if risk_per_share <= 0:
                risk_per_share = current_price * 0.1

            target_1 = current_price + (risk_per_share * 1.5)
            target_2 = current_price + (risk_per_share * 2.5)
            target_3 = current_price + (risk_per_share * 3.5)

            target_1 = min(target_1, current_price * 1.15, resistance if resistance > current_price else current_price * 1.15)
            target_2 = min(target_2, current_price * 1.25)
            target_3 = min(target_3, current_price * 1.40)

            target_1 = max(target_1, current_price * 1.05)
            target_2 = max(target_2, target_1 * 1.05)
            target_3 = max(target_3, target_2 * 1.05)

            # Entry Timing
            ma_200       = trends.get('ma_200')
            entry_timing = "Wait for pullback or confirmation"
            if ma_200 and not pd.isna(ma_200):
                if current_price > ma_200 * 1.02:
                    entry_timing = "Buy on pullbacks or accumulate"
                elif current_price > ma_200 * 0.98:
                    entry_timing = "Near long-term MA; watch for confirmation"
                else:
                    entry_timing = "Wait for reclaim of 200-day MA"

            stop_dist_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
            upside_pot    = ((target_2 - current_price) / current_price) * 100 if current_price > 0 else 0
            rr_ratio      = upside_pot / stop_dist_pct if stop_dist_pct > 0 else 0

            return {
                'entry_signal':    entry_signal,
                'entry_strategy':  entry_strategy,
                'entry_timing':    entry_timing,
                'stop_loss':       round(stop_loss, 2),
                'targets': {
                    'target_1': round(target_1, 2),
                    'target_2': round(target_2, 2),
                    'target_3': round(target_3, 2),
                },
                'support':          round(support, 2),
                'resistance':       round(resistance, 2),
                'holding_period':   holding_period,
                'trade_management_note': (
                    "Book partial profits at targets (e.g., 1/3 each). "
                    "Trail stop loss to breakeven after Target 1. "
                    "Review fundamentals quarterly."
                ),
                'stop_distance_pct':  round(stop_dist_pct, 2),
                'upside_potential':   round(upside_pot, 2),
                'risk_reward_ratio':  round(rr_ratio, 2),
            }
        except Exception as e:
            logger.error(f"Error generating position trading plan: {e}")
            logger.error(traceback.format_exc())
            return default_plan

    # ==============================================================================
    #  TECHNICAL ANALYSIS HELPERS
    # ==============================================================================

    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations on a pandas Series."""
        try:
            if data is None or data.empty or len(data) < window:
                return pd.Series(np.nan, index=data.index if hasattr(data, 'index') else None)

            min_p = max(1, window // 2)
            if operation == 'mean':   return data.rolling(window=window, min_periods=min_p).mean()
            elif operation == 'std':  return data.rolling(window=window, min_periods=min_p).std()
            elif operation == 'min':  return data.rolling(window=window, min_periods=min_p).min()
            elif operation == 'max':  return data.rolling(window=window, min_periods=min_p).max()
            else:
                logger.error(f"Unknown rolling operation: {operation}")
                return pd.Series(np.nan, index=data.index)
        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {e}")
            return pd.Series(np.nan, index=data.index if hasattr(data, 'index') else None)

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI (EMA-based) for a pandas Series."""
        try:
            if prices is None or prices.empty or len(prices) < period + 1:
                return pd.Series(50, index=prices.index if hasattr(prices, 'index') else None)

            delta    = prices.diff()
            gain     = delta.where(delta > 0, 0).fillna(0)
            loss     = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
            rs       = avg_gain / avg_loss.replace(0, 1e-6)
            rsi      = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index if hasattr(prices, 'index') else None)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD for a pandas Series."""
        try:
            if prices is None or prices.empty or len(prices) < slow:
                empty = pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)
                return empty, empty, empty

            exp1        = prices.ewm(span=fast,   adjust=False).mean()
            exp2        = prices.ewm(span=slow,   adjust=False).mean()
            macd_line   = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram   = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            empty = pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)
            return empty, empty, empty

    def calculate_support_resistance(self, data, window=20):
        """Calculate support/resistance using rolling min/max."""
        try:
            if (data is None or data.empty
                    or 'Low' not in data.columns
                    or 'High' not in data.columns
                    or len(data) < window):
                return None, None

            support    = self.safe_rolling_calculation(data['Low'],  window, 'min').iloc[-1]
            resistance = self.safe_rolling_calculation(data['High'], window, 'max').iloc[-1]
            return support, resistance

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return None, None

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics from OHLCV DataFrame."""
        default_metrics = {
            'volatility': 0.3, 'var_95': -0.05, 'max_drawdown': -0.20,
            'sharpe_ratio': 0.0, 'atr': 0, 'risk_level': 'HIGH'
        }
        try:
            if data is None or data.empty or 'Close' not in data.columns or len(data) < 22:
                logger.warning("Insufficient data for full risk metrics.")
                last_close = data['Close'].iloc[-1] if not data.empty else 0
                default_metrics['atr'] = last_close * 0.02 if last_close else 0
                return default_metrics

            returns = data['Close'].pct_change().dropna()
            if returns.empty:
                return default_metrics

            # Volatility
            volatility = returns.std() * np.sqrt(252)
            volatility = volatility if not pd.isna(volatility) and volatility >= 0 else 0.3

            # VaR 95
            var_95 = np.percentile(returns, 5) if len(returns) > 20 else -0.05
            var_95 = var_95 if not pd.isna(var_95) else -0.05

            # Max Drawdown
            rolling_max  = data['Close'].cummax()
            drawdown     = (data['Close'] - rolling_max) / rolling_max.replace(0, 1)
            max_drawdown = drawdown.min()
            max_drawdown = max_drawdown if not pd.isna(max_drawdown) else -0.20

            # Sharpe Ratio
            risk_free_rate_annual = 0.06
            mean_daily_return     = returns.mean()
            std_daily_return      = returns.std()
            sharpe_ratio          = 0.0
            if std_daily_return > 0:
                excess_return = mean_daily_return - (risk_free_rate_annual / 252)
                sharpe_ratio  = (excess_return / std_daily_return) * np.sqrt(252)
            sharpe_ratio = sharpe_ratio if not pd.isna(sharpe_ratio) else 0.0

            # ATR (EMA-14)
            if len(data) >= 15 and all(c in data.columns for c in ['High', 'Low', 'Close']):
                high_low    = data['High'] - data['Low']
                high_close  = np.abs(data['High'] - data['Close'].shift())
                low_close   = np.abs(data['Low']  - data['Close'].shift())
                tr          = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).dropna()
                atr         = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
            else:
                atr = data['Close'].iloc[-1] * 0.02
            atr = atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.02

            if volatility > 0.40:     risk_level = 'HIGH'
            elif volatility > 0.25:   risk_level = 'MEDIUM'
            else:                     risk_level = 'LOW'

            return {
                'volatility': volatility, 'var_95': var_95,
                'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe_ratio,
                'atr': atr, 'risk_level': risk_level,
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            last_close = data['Close'].iloc[-1] if data is not None and not data.empty else 0
            default_metrics['atr'] = last_close * 0.02 if last_close else 0
            return default_metrics

    # ==============================================================================
    #  NEWS SENTIMENT HELPERS
    # ==============================================================================

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob."""
        sentiments, confidences = [], []
        if not articles:
            return sentiments, confidences
        for article in articles:
            try:
                if not article or not isinstance(article, str):
                    sentiments.append('neutral')
                    confidences.append(0.3)
                    continue
                blob       = TextBlob(article)
                polarity   = blob.sentiment.polarity
                confidence = abs(polarity)
                if polarity > 0.1:
                    sentiments.append('positive')
                    confidences.append(min(confidence, 0.9))
                elif polarity < -0.1:
                    sentiments.append('negative')
                    confidences.append(min(confidence, 0.9))
                else:
                    sentiments.append('neutral')
                    confidences.append(max(0.3, 1 - confidence * 2))
            except Exception as e:
                logger.warning(f"TextBlob error: {e}")
                sentiments.append('neutral')
                confidences.append(0.3)
        return sentiments, confidences

    def get_sentiment_summary(self, sentiment_scores):
        """Get summary count of sentiment scores."""
        if not sentiment_scores:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        positive_count = sum(1 for s in sentiment_scores if 'positive' in str(s).lower())
        negative_count = sum(1 for s in sentiment_scores if 'negative' in str(s).lower())
        neutral_count  = len(sentiment_scores) - positive_count - negative_count
        return {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count}

    def fetch_indian_news(self, symbol, num_articles=20):
        """Fetch full news articles using newsapi.ai (Event Registry)."""
        try:
            if not self.event_registry_api_key:
                logger.warning("EVENT_REGISTRY_API_KEY not configured.")
                return None

            base_symbol  = str(symbol).split('.')[0].upper()
            stock_info   = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            payload = {
                "action":           "getArticles",
                "keyword":          company_name,
                "keywordOper":      "and",
                "lang":             ["eng"],
                "articlesPage":     1,
                "articlesCount":    num_articles,
                "articlesSortBy":   "date",
                "articlesSortByAsc": False,
                "dataType":         ["news"],
                "includeArticleBody": True,
                "apiKey":           self.event_registry_api_key,
            }

            response = requests.post(self.event_registry_endpoint, json=payload, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Event Registry error {response.status_code}: {response.text}")
                return None

            data    = response.json()
            results = data.get("articles", {}).get("results", [])

            articles = []
            for item in results:
                body  = item.get("body")
                title = item.get("title")
                if body and len(body.split()) >= 120:
                    articles.append(body)
                elif title:
                    articles.append(title)

            if not articles:
                logger.warning(f"No usable articles found for {company_name}")
                return None

            avg_len = sum(len(a) for a in articles) // len(articles)
            logger.info(
                f"Event Registry news fetched for {company_name}: "
                f"{len(articles)} articles | avg_chars={avg_len}"
            )
            return articles

        except requests.exceptions.Timeout:
            logger.warning(f"Event Registry request timed out for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Event Registry fetch failed for {symbol}: {e}")
            return None

    
    # ==============================================================================
    #  PORTFOLIO GENERATION
    # ==============================================================================

    def create_personalized_portfolio(self, risk_appetite, time_period_months, budget):
        """Create a personalized portfolio using risk-based position sizing."""
        try:
            min_score = 65
            symbols   = self.get_all_stock_symbols()
            stock_results = []

            logger.info(f"Analyzing {len(symbols)} stocks for position portfolio...")
            for i, symbol in enumerate(symbols):
                logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})...")
                result = self.analyze_position_trading_stock(symbol)
                if (result
                        and result.get('position_score', 0) >= min_score
                        and result.get('trading_plan', {}).get('entry_signal') in ['BUY', 'STRONG BUY']):
                    stock_results.append(result)

            if not stock_results:
                logger.warning("No stocks met the minimum criteria for the position portfolio.")
                return {"portfolio": {}, "summary": {"error": "No suitable stocks found."}}

            max_positions  = self.position_trading_params.get('max_positions', 10)
            sorted_stocks  = sorted(stock_results, key=lambda x: x['position_score'], reverse=True)
            selected_stocks = sorted_stocks[:max_positions]
            logger.info(f"Selected top {len(selected_stocks)} stocks based on score and max_positions.")

            portfolio = self.calculate_position_sizes(selected_stocks, budget)
            if not portfolio:
                logger.error("Failed to calculate position sizes.")
                return {"portfolio": {}, "summary": {"error": "Could not allocate budget based on risk parameters."}}

            summary = self.generate_portfolio_summary(portfolio, time_period_months, budget)

            return {
                'portfolio':          portfolio,
                'summary':            summary,
                'risk_profile':       risk_appetite,
                'time_period_months': time_period_months,
                'budget':             budget,
            }
        except Exception as e:
            logger.error(f"Error creating personalized position portfolio: {e}", exc_info=True)
            return {"portfolio": {}, "summary": {"error": f"Internal error: {e}"}}

    def calculate_position_sizes(self, selected_stocks, total_capital):
        """Calculate position sizes based on fixed risk percentage."""
        portfolio              = {}
        risk_per_trade_pct     = self.position_trading_params['risk_per_trade']
        capital_at_risk_per_trade = total_capital * risk_per_trade_pct
        max_total_risk         = total_capital * self.position_trading_params['max_portfolio_risk']
        total_allocated        = 0
        current_total_risk     = 0

        logger.info(
            f"Calculating position sizes. Risk per trade: {risk_per_trade_pct*100:.1f}%, "
            f"Capital at risk/trade: {capital_at_risk_per_trade:.2f}"
        )

        for stock_data in selected_stocks:
            try:
                current_price = stock_data.get('current_price')
                stop_loss     = stock_data.get('trading_plan', {}).get('stop_loss')

                if (not isinstance(current_price, (int, float)) or current_price <= 0
                        or not isinstance(stop_loss, (int, float)) or stop_loss <= 0
                        or current_price <= stop_loss):
                    logger.warning(
                        f"Skipping {stock_data.get('symbol')}: "
                        f"Invalid price ({current_price}) or stop loss ({stop_loss})."
                    )
                    continue

                risk_per_share    = current_price - stop_loss
                num_shares        = int(capital_at_risk_per_trade / risk_per_share)

                if num_shares == 0:
                    logger.warning(
                        f"Skipping {stock_data.get('symbol')}: Cannot afford even one share "
                        f"with risk {risk_per_share:.2f} per share."
                    )
                    continue

                investment_amount = num_shares * current_price
                trade_risk        = num_shares * risk_per_share

                if total_allocated + investment_amount > total_capital:
                    logger.info(f"Stopping allocation for {stock_data.get('symbol')}: Exceeds total budget.")
                    break
                if current_total_risk + trade_risk > max_total_risk:
                    logger.info(f"Stopping allocation for {stock_data.get('symbol')}: Exceeds max portfolio risk.")
                    break

                total_allocated    += investment_amount
                current_total_risk += trade_risk
                symbol              = stock_data.get('symbol', 'Unknown')

                portfolio[symbol] = {
                    'company_name':      stock_data.get('company_name', 'Unknown'),
                    'sector':            stock_data.get('sector', 'Unknown'),
                    'score':             round(stock_data.get('position_score', 0), 2),
                    'num_shares':        num_shares,
                    'entry_price':       round(current_price, 2),
                    'investment_amount': round(investment_amount, 2),
                    'stop_loss':         round(stop_loss, 2),
                    'trade_risk_amount': round(trade_risk, 2),
                    'targets':           stock_data.get('trading_plan', {}).get('targets'),
                }
                logger.info(
                    f"Allocated {investment_amount:.2f} to {symbol} "
                    f"({num_shares} shares). Trade risk: {trade_risk:.2f}"
                )

            except Exception as e:
                logger.error(f"Error sizing position for {stock_data.get('symbol', 'N/A')}: {e}")
                continue

        logger.info(f"Total allocated: {total_allocated:.2f}, Total risk: {current_total_risk:.2f}")
        return portfolio

    def generate_portfolio_summary(self, portfolio, time_period_months, budget):
        """Generate a summary of the created portfolio."""
        if not portfolio:
            return {"error": "Portfolio is empty."}

        total_investment = sum(stock['investment_amount'] for stock in portfolio.values())
        total_risk       = sum(stock['trade_risk_amount'] for stock in portfolio.values())
        num_stocks       = len(portfolio)
        avg_score        = (
            sum(stock['score'] for stock in portfolio.values()) / num_stocks
            if num_stocks > 0 else 0
        )

        sector_allocation = {}
        for stock in portfolio.values():
            sector = stock.get('sector', 'Unknown')
            sector_allocation[sector] = sector_allocation.get(sector, 0) + stock['investment_amount']

        sector_allocation_pct = (
            {s: (a / total_investment * 100) for s, a in sector_allocation.items()}
            if total_investment > 0 else {}
        )

        base_annual_return    = 0.20
        projected_annual      = base_annual_return * (avg_score / 100)
        time_years            = time_period_months / 12
        total_expected_return = projected_annual * time_years * 100

        return {
            'total_budget':              budget,
            'total_investment':          round(total_investment, 2),
            'remaining_cash':            round(budget - total_investment, 2),
            'total_portfolio_risk':      round(total_risk, 2),
            'total_portfolio_risk_pct':  round((total_risk / budget) * 100, 2) if budget > 0 else 0,
            'number_of_stocks':          num_stocks,
            'average_score':             round(avg_score, 2),
            'sector_allocation_pct':     {s: round(p, 1) for s, p in sector_allocation_pct.items()},
            'projected_return_pct':      round(total_expected_return, 2),
            'recommended_holding_period': f"{time_period_months} months",
        }

    # ==============================================================================
    #  SAMPLE / FALLBACK DATA
    # ==============================================================================

    def get_sample_mda_analysis(self, symbol):
        """Generate deterministic sample MDA analysis as last-resort fallback."""
        try:
            base_score      = 50 + (hash(symbol) % 31) - 15
            management_tone = "Neutral"
            if base_score >= 70:     management_tone = "Very Optimistic"
            elif base_score >= 60:   management_tone = "Optimistic"
            elif base_score <= 40:   management_tone = "Pessimistic"
            elif base_score <= 30:   management_tone = "Very Pessimistic"

            return {
                'mda_score':       base_score,
                'management_tone': management_tone,
                'confidence':      0.75 + (hash(symbol) % 10) / 100,
                'analysis_method': 'Sample MDA Analysis (Fallback)',
                'fls_count':       0,
                'avg_uncertainty_density': 0.0,
                'sections_found':  [],
                'extraction_status': 'fallback',
            }
        except Exception as e:
            logger.error(f"Error generating sample MDA: {e}")
            return {
                'mda_score': 50, 'management_tone': 'Neutral',
                'analysis_method': 'Error', 'fls_count': 0,
                'avg_uncertainty_density': 0.0, 'sections_found': [],
                'extraction_status': 'error',
            }

    def get_sample_mda_texts(self, symbol):
        """Return minimal sample MDA text strings for legacy HF API calls."""
        stock_info   = self.get_stock_info_from_db(symbol)
        company_name = stock_info.get("name", symbol)
        return [
            f"{company_name} demonstrated strong revenue growth this fiscal year, "
            f"driven by expansion in key markets and improved operational efficiency.",
            f"Management expects continued momentum in the coming quarters, "
            f"with strategic investments in technology and distribution channels.",
        ]

    def get_sample_news(self, symbol):
        """Generate sample news articles as last-resort fallback."""
        try:
            base_symbol  = str(symbol).split('.')[0]
            stock_info   = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            positive_news = [
                f"{company_name} reports strong earnings",
                f"Analysts upgrade {company_name}",
                f"{company_name} announces expansion",
            ]
            negative_news = [
                f"Concerns over {company_name}'s debt levels",
                f"{company_name} faces regulatory hurdles",
                f"Sector outlook weakens for {company_name}",
            ]
            neutral_news = [
                f"{company_name} maintains market share",
                f"Management change at {company_name}",
                f"General market update affects {company_name}",
            ]

            idx = hash(symbol) % 3
            if idx == 0:   return positive_news * 2 + negative_news + neutral_news * 2
            elif idx == 1: return negative_news * 2 + positive_news + neutral_news * 2
            else:          return neutral_news * 3 + positive_news + negative_news
        except Exception as e:
            logger.error(f"Error generating sample news: {e}")
            return [f"News item for {symbol}"]


# Configure logging if run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    print("EnhancedPositionTradingSystem module loaded.")

