import os
import re
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from textblob import TextBlob
import warnings
import requests
from eodhd import EodHdApi # Import the new library

# Local application imports
import config
from hf_utils import query_hf_api

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedPositionTradingSystem:
    def __init__(self):
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

            # --- EODHD API Setup ---
            self.eodhd_api_key = os.getenv("EODHD_API_KEY")
            if not self.eodhd_api_key:
                logger.warning("⚠️ EODHD_API_KEY not set. Data fetching will fail.")

            # --- Session for ImprovedMDAExtractor ---
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            logger.info("✅ EnhancedPositionTradingSystem initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing EnhancedPositionTradingSystem: {e}")
            raise

    # ==============================================================================
    #  NEW: API CALLING HELPER METHODS
    # ==============================================================================

    def _analyze_sentiment_via_api(self, articles: list) -> tuple[list, list] | None:
        """Analyzes news sentiment by calling the remote SBERT Hugging Face API."""
        try:
            payload = {"inputs": articles}
            api_results = query_hf_api(self.sentiment_api_url, payload)

            if api_results is None:
                raise ValueError("API call to SBERT HF Space failed or returned no data.")

            if isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
                api_results = api_results[0]

            sentiments = [res.get('label', 'neutral').lower() for res in api_results]
            confidences = [res.get('score', 0.5) for res in api_results]
            return sentiments, confidences
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logging.error(f"Could not parse SBERT API response. Error: {e}. Response: {api_results if 'api_results' in locals() else 'unknown'}")
            return None

    def _analyze_mda_via_api(self, mda_texts: list) -> dict | None:
        """Analyzes MDA text by calling the remote MDA Hugging Face API."""
        try:
            payload = {"inputs": mda_texts}
            api_results = query_hf_api(self.mda_api_url, payload)

            if api_results is None:
                raise ValueError("API call to MDA HF Space failed or returned no data.")
            
            # This logic assumes your API returns a list of sentiment dictionaries.
            # You must adapt this to your actual API output format.
            sentiments = [res.get('label') for res in api_results]
            confidences = [res.get('score') for res in api_results]
            
            sentiment_scores = []
            for sentiment, confidence in zip(sentiments, confidences):
                if str(sentiment).lower() in ['positive', 'very_positive', 'label_4', 'label_3']:
                    sentiment_scores.append(confidence)
                elif str(sentiment).lower() in ['negative', 'very_negative', 'label_0', 'label_1']:
                    sentiment_scores.append(-confidence)
                else:
                    sentiment_scores.append(0)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            mda_score = 50 + (avg_sentiment * 50)
            mda_score = max(0, min(100, mda_score))
            
            management_tone = "Neutral"
            if mda_score >= 70: management_tone = "Very Optimistic"
            elif mda_score >= 60: management_tone = "Optimistic"
            elif mda_score <= 40: management_tone = "Pessimistic"

            return {
                'mda_score': mda_score,
                'management_tone': management_tone,
                'confidence': np.mean(confidences) if confidences else 0,
                'analysis_method': 'Remote PyTorch BERT MDA Model (API)',
            }
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logging.error(f"Could not parse MDA API response. Error: {e}. Response: {api_results if 'api_results' in locals() else 'unknown'}")
            return None

    # ==============================================================================
    #  UPDATED: CORE ANALYSIS METHODS
    # ==============================================================================

    def analyze_news_sentiment(self, symbol, num_articles=20):
        """Main sentiment analysis function updated to use the API."""
        try:
            articles = self.fetch_indian_news(symbol, num_articles) or self.get_sample_news(symbol)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"

            if not articles:
                return [], [], [], "No Analysis", "No Source"

            if self.sentiment_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    return sentiments, articles, confidences, "SBERT API", news_source
            
            logging.warning(f"Falling back to TextBlob for news sentiment for {symbol}.")
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob Fallback", news_source
        except Exception as e:
            logging.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error"

    def updated_analyze_mda_sentiment(self, symbol):
        """Updated to prioritize API calls for real MD&A extraction and analysis."""
        try:
            if not self.mda_api_available:
                logger.warning("MDA API URL not configured. Using sample analysis as fallback.")
                return self.get_sample_mda_analysis(symbol)

            # extractor = self.ImprovedMDAExtractor()
            # mda_texts = extractor.get_mda_text(symbol, max_reports=3)
            mda_texts = [] # Temporarily disable for speed, use sample data

            if not mda_texts:
                logger.warning(f"No real MDA text found for {symbol}, using sample analysis.")
                return self.get_sample_mda_analysis(symbol)
            
            logger.info(f"Sending {len(mda_texts)} MD&A texts to the API for analysis.")
            api_result = self._analyze_mda_via_api(mda_texts)

            if api_result:
                return api_result
            
            logger.warning(f"MDA API call failed for {symbol}. Using sample analysis as fallback.")
            return self.get_sample_mda_analysis(symbol)
            
        except Exception as e:
            logger.error(f"An unexpected error in MDA sentiment analysis for {symbol}: {e}")
            return self.get_sample_mda_analysis(symbol)
            
    def _validate_trading_params(self):
        """Validate position trading parameters"""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier', 'max_positions']

            for param in required_params:
                if param not in self.position_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")

                value = self.position_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")

            # Additional validation
            if self.position_trading_params['min_holding_period'] >= self.position_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")

            if self.position_trading_params['risk_per_trade'] > 0.05:  # 5% max risk per trade
                raise ValueError("risk_per_trade cannot exceed 5% for position trading")

            # Validate weights sum to 1.0
            total_weight = (self.position_trading_params['fundamental_weight'] +
                            self.position_trading_params['technical_weight'] +
                            self.position_trading_params['sentiment_weight'] +
                            self.position_trading_params['mda_weight'])

            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
                logger.warning(f"Scoring weights don't sum to 1.0: {total_weight:.3f}")

            logger.info("Position trading parameters validated successfully")

        except Exception as e:
            logger.error(f"Error validating trading parameters: {str(e)}")
            raise

    class ImprovedMDAExtractor:
        """Enhanced MD&A text extractor for Indian companies"""

        def __init__(self):
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })

        def get_mda_text(self, symbol: str, max_reports: int = 3) -> List[str]:
            """
            Extract real MD&A text from multiple sources
            Returns a list of MD&A text sections
            """
            try:
                symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
                logger.info(f"Extracting MD&A text for {symbol}")

                mda_texts = []

                # Try multiple extraction methods
                methods = [
                    self._extract_from_bse_announcements,
                    self._extract_from_nse_reports,
                    self._extract_from_company_website,
                    self._extract_from_annual_reports,
                    self._extract_from_yahoo_finance_filings
                ]

                for method in methods:
                    try:
                        texts = method(symbol)
                        if texts:
                            mda_texts.extend(texts)
                            logger.info(f"Successfully extracted {len(texts)} MD&A sections using {method.__name__}")

                            # If we have enough content, break
                            if len(mda_texts) >= max_reports:
                                break

                    except Exception as e:
                        logger.warning(f"Method {method.__name__} failed for {symbol}: {str(e)}")
                        continue

                    # Rate limiting
                    time.sleep(1)

                # Clean and filter the extracted texts
                cleaned_texts = self._clean_and_validate_mda_texts(mda_texts)

                if cleaned_texts:
                    logger.info(f"Successfully extracted {len(cleaned_texts)} MD&A texts for {symbol}")
                    return cleaned_texts[:max_reports]  # Return up to max_reports
                else:
                    logger.warning(f"No valid MD&A text found for {symbol}")
                    return []

            except Exception as e:
                logger.error(f"Error extracting MD&A text for {symbol}: {str(e)}")
                return []

        def _extract_from_bse_announcements(self, symbol: str) -> List[str]:
            """Extract MD&A from BSE announcements"""
            try:
                # BSE announcement URL pattern
                bse_url = f"https://www.bseindia.com/stock-share-price/{symbol}/announcements/"

                response = self.session.get(bse_url, timeout=10)
                if response.status_code != 200:
                    return []

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report links or MD&A related announcements
                announcements = soup.find_all('a', href=re.compile(r'annual|report|mda|management.*discussion',
                                                                   re.IGNORECASE))

                mda_texts = []
                for link in announcements[:3]:  # Check first 3 relevant links
                    try:
                        href = link.get('href')
                        if href and not href.startswith('http'):
                            href = 'https://www.bseindia.com' + href

                        doc_response = self.session.get(href, timeout=10)
                        if doc_response.status_code == 200:
                            # Extract text from PDF or HTML
                            text = self._extract_text_from_response(doc_response)
                            mda_section = self._extract_mda_section(text)
                            if mda_section:
                                mda_texts.append(mda_section)

                    except Exception as e:
                        logger.debug(f"Error processing BSE link {href}: {str(e)}")
                        continue

                return mda_texts

            except Exception as e:
                logger.error(f"Error extracting from BSE for {symbol}: {str(e)}")
                return []

        def _extract_from_nse_reports(self, symbol: str) -> List[str]:
            """Extract MD&A from NSE corporate reports"""
            try:
                # NSE doesn't have a direct API, but we can try their corporate section
                nse_search_url = f"https://www.nseindia.com/companies-listing/corporate-filings-company-wise"

                # This would require more complex scraping with session management
                # For now, we'll implement a basic version

                return []  # Placeholder - NSE requires complex session handling

            except Exception as e:
                logger.error(f"Error extracting from NSE for {symbol}: {str(e)}")
                return []

        def _extract_from_company_website(self, symbol: str) -> List[str]:
            """Try to extract MD&A from company's official website"""
            try:
                # Get company website from yfinance
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info

                website = info.get('website', '')
                if not website:
                    return []

                # Look for investor relations section
                ir_urls = [
                    f"{website}/investor-relations",
                    f"{website}/investors",
                    f"{website}/annual-reports",
                    f"{website}/financial-reports"
                ]

                mda_texts = []
                for url in ir_urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Look for annual report links
                            report_links = soup.find_all('a', href=re.compile(r'annual.*report|financial.*report',
                                                                              re.IGNORECASE))

                            for link in report_links[:2]:  # Check first 2 reports
                                try:
                                    href = link.get('href')
                                    if href and not href.startswith('http'):
                                        href = website + href

                                    doc_response = self.session.get(href, timeout=15)
                                    if doc_response.status_code == 200:
                                        text = self._extract_text_from_response(doc_response)
                                        mda_section = self._extract_mda_section(text)
                                        if mda_section:
                                            mda_texts.append(mda_section)

                                except Exception as e:
                                    logger.debug(f"Error processing company website link: {str(e)}")
                                    continue

                    except Exception as e:
                        logger.debug(f"Error accessing {url}: {str(e)}")
                        continue

                return mda_texts

            except Exception as e:
                logger.error(f"Error extracting from company website for {symbol}: {str(e)}")
                return []

        def _extract_from_annual_reports(self, symbol: str) -> List[str]:
            """Extract MD&A from publicly available annual reports"""
            try:
                # Search for annual reports using Google Search API or web scraping
                search_query = f"{symbol} annual report filetype:pdf site:bseindia.com OR site:nseindia.com"

                # This is a simplified version - in practice, you'd use Google Search API
                # or implement more sophisticated web scraping

                return []  # Placeholder

            except Exception as e:
                logger.error(f"Error extracting from annual reports for {symbol}: {str(e)}")
                return []

        def _extract_from_yahoo_finance_filings(self, symbol: str) -> List[str]:
            """Extract MD&A information from Yahoo Finance filings data"""
            try:
                ticker = yf.Ticker(f"{symbol}.NS")

                # Get recent financial data and news
                info = ticker.info
                news = ticker.news

                # Extract relevant information from company description and recent news
                mda_like_texts = []

                # Company description often contains management perspective
                if 'longBusinessSummary' in info and info['longBusinessSummary']:
                    business_summary = info['longBusinessSummary']
                    if len(business_summary) > 200:  # Only if substantial content
                        mda_like_texts.append(business_summary)

                # Recent news articles that might contain management quotes
                for article in news[:5]:  # Check recent 5 articles
                    try:
                        if 'summary' in article and article['summary']:
                            summary = article['summary']
                            # Look for management-related content
                            if any(keyword in summary.lower() for keyword in
                                   ['management', 'ceo', 'outlook', 'strategy', 'expects', 'guidance']):
                                mda_like_texts.append(summary)
                    except Exception:
                        continue

                return mda_like_texts

            except Exception as e:
                logger.error(f"Error extracting from Yahoo Finance for {symbol}: {str(e)}")
                return []

        def _extract_text_from_response(self, response) -> str:
            """Extract text from HTTP response (HTML or PDF)"""
            try:
                content_type = response.headers.get('content-type', '').lower()

                if 'application/pdf' in content_type:
                    # Extract text from PDF
                    return self._extract_text_from_pdf(response.content)
                else:
                    # Extract text from HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    return soup.get_text()

            except Exception as e:
                logger.error(f"Error extracting text from response: {str(e)}")
                return ""

        def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
            """Extract text from PDF content"""
            try:
                import PyPDF2
                from io import BytesIO

                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                text = ""

                for page in pdf_reader.pages:
                    text += page.extract_text()

                return text

            except ImportError:
                logger.warning("PyPDF2 not available for PDF text extraction")
                return ""
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
                return ""

        def _extract_mda_section(self, full_text: str) -> Optional[str]:
            """Extract MD&A section from full document text"""
            try:
                if not full_text or len(full_text) < 100:
                    return None

                # Common MD&A section headers in Indian reports
                mda_patterns = [
                    r"management.*discussion.*and.*analysis",
                    r"management.*discussion",
                    r"directors.*report",
                    r"management.*analysis",
                    r"business.*outlook",
                    r"management.*commentary",
                    r"operational.*review",
                    r"management.*perspective"
                ]

                text_lower = full_text.lower()

                for pattern in mda_patterns:
                    matches = list(re.finditer(pattern, text_lower))

                    if matches:
                        # Find the start of MD&A section
                        start_pos = matches[0].start()

                        # Find the end (look for next major section or end of document)
                        end_patterns = [
                            r"financial.*statements",
                            r"notes.*to.*accounts",
                            r"auditor.*report",
                            r"corporate.*governance",
                            r"annexure",
                            r"schedule"
                        ]

                        end_pos = len(full_text)
                        for end_pattern in end_patterns:
                            end_matches = list(re.finditer(end_pattern, text_lower[start_pos:]))
                            if end_matches:
                                end_pos = start_pos + end_matches[0].start()
                                break

                        # Extract the section
                        mda_section = full_text[start_pos:end_pos]

                        # Clean and validate
                        if len(mda_section) > 500:  # Minimum length for meaningful MD&A
                            return self._clean_extracted_text(mda_section)

                # If no specific MD&A section found, look for management-related content
                management_content = self._extract_management_content(full_text)
                if management_content and len(management_content) > 300:
                    return management_content

                return None

            except Exception as e:
                logger.error(f"Error extracting MD&A section: {str(e)}")
                return None

        def _extract_management_content(self, text: str) -> Optional[str]:
            """Extract management-related content from text"""
            try:
                sentences = text.split('.')
                management_sentences = []

                management_keywords = [
                    'management', 'strategy', 'outlook', 'expects', 'believes',
                    'anticipates', 'guidance', 'performance', 'operations',
                    'future', 'growth', 'investment', 'market', 'business'
                ]

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 50:  # Minimum sentence length
                        sentence_lower = sentence.lower()
                        keyword_count = sum(1 for keyword in management_keywords if keyword in sentence_lower)

                        if keyword_count >= 2:  # At least 2 management-related keywords
                            management_sentences.append(sentence)

                if len(management_sentences) >= 5:  # At least 5 relevant sentences
                    return '. '.join(management_sentences[:20])  # Limit to 20 sentences

                return None

            except Exception as e:
                logger.error(f"Error extracting management content: {str(e)}")
                return None

        def _clean_extracted_text(self, text: str) -> str:
            """Clean and format extracted MD&A text"""
            try:
                # Remove excessive whitespace
                text = re.sub(r'\s+', ' ', text)

                # Remove page numbers and headers/footers
                text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
                text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)

                # Remove common report artifacts
                artifacts = [
                    r'annual report \d{4}',
                    r'page \d+',
                    r'www\.\w+\.com',
                    r'tel:?\s*\+?\d+[\d\s\-\(\)]+',
                    r'email:\s*\S+@\S+',
                ]

                for artifact in artifacts:
                    text = re.sub(artifact, '', text, flags=re.IGNORECASE)

                # Clean up spacing
                text = re.sub(r'\s+', ' ', text).strip()

                return text

            except Exception as e:
                logger.error(f"Error cleaning extracted text: {str(e)}")
                return text

        def _clean_and_validate_mda_texts(self, mda_texts: List[str]) -> List[str]:
            """Clean and validate extracted MD&A texts"""
            try:
                cleaned_texts = []

                for text in mda_texts:
                    if not text or not isinstance(text, str):
                        continue

                    # Clean the text
                    cleaned_text = self._clean_extracted_text(text)

                    # Validate quality
                    if self._validate_mda_text_quality(cleaned_text):
                        cleaned_texts.append(cleaned_text)

                # Remove duplicates (texts that are too similar)
                unique_texts = self._remove_similar_texts(cleaned_texts)

                return unique_texts

            except Exception as e:
                logger.error(f"Error cleaning and validating MD&A texts: {str(e)}")
                return mda_texts  # Return original if cleaning fails

        def _validate_mda_text_quality(self, text: str) -> bool:
            """Validate if the extracted text is meaningful MD&A content"""
            try:
                if not text or len(text) < 200:
                    return False

                # Check for minimum management-related keywords
                management_keywords = [
                    'management', 'performance', 'business', 'operations',
                    'growth', 'strategy', 'market', 'revenue', 'profit',
                    'outlook', 'expects', 'believes', 'future'
                ]

                text_lower = text.lower()
                keyword_count = sum(1 for keyword in management_keywords if keyword in text_lower)

                # Require at least 3 management-related keywords
                if keyword_count < 3:
                    return False

                # Check for reasonable sentence structure
                sentences = text.split('.')
                valid_sentences = [s for s in sentences if len(s.strip()) > 20]

                if len(valid_sentences) < 5:
                    return False

                return True

            except Exception as e:
                logger.error(f"Error validating MD&A text quality: {str(e)}")
                return False

        def _remove_similar_texts(self, texts: List[str]) -> List[str]:
            """Remove texts that are too similar to each other"""
            try:
                if len(texts) <= 1:
                    return texts

                unique_texts = []

                for text in texts:
                    is_unique = True

                    for existing_text in unique_texts:
                        # Simple similarity check based on common words
                        similarity = self._calculate_text_similarity(text, existing_text)
                        if similarity > 0.7:  # 70% similarity threshold
                            is_unique = False
                            break

                    if is_unique:
                        unique_texts.append(text)

                return unique_texts

            except Exception as e:
                logger.error(f"Error removing similar texts: {str(e)}")
                return texts

        def _calculate_text_similarity(self, text1: str, text2: str) -> float:
            """Calculate similarity between two texts"""
            try:
                # Simple word-based similarity
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())

                if not words1 and not words2:
                    return 1.0

                if not words1 or not words2:
                    return 0.0

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                return intersection / union if union > 0 else 0.0

            except Exception as e:
                logger.error(f"Error calculating text similarity: {str(e)}")
                return 0.0

    # Update the analyze_mda_sentiment method in your main class
    def updated_analyze_mda_sentiment(self, symbol):
        """Updated to prioritize API calls for real MD&A extraction and analysis."""
        try:
            if not self.mda_api_available:
                logger.warning("MDA API URL not configured. Using sample analysis as fallback.")
                return self.get_sample_mda_analysis(symbol)

            extractor = self.ImprovedMDAExtractor()
            mda_texts = extractor.get_mda_text(symbol, max_reports=3)

            if not mda_texts:
                logger.warning(f"No real MDA text found for {symbol}, using sample analysis.")
                return self.get_sample_mda_analysis(symbol)

            logger.info(f"Sending {len(mda_texts)} MD&A texts to the API for analysis.")
            api_result = self._analyze_mda_via_api(mda_texts)

            if api_result:
                return api_result

            logger.warning(f"MDA API call failed for {symbol}. Using sample analysis as fallback.")
            return self.get_sample_mda_analysis(symbol)

        except Exception as e:
            logger.error(f"An unexpected error in MDA sentiment analysis for {symbol}: {e}")
            return self.get_sample_mda_analysis(symbol)

    def _get_alternative_management_content(self, symbol):
        """
        Get alternative management content from earnings calls, press releases, etc.
        """
        try:
            alternative_texts = []

            # Method 1: Try to get recent earnings call transcripts
            try:
                # This would require integration with services like:
                # - AlphaVantage (has earnings call transcripts)
                # - Financial news APIs
                # - Company press releases

                # For now, we'll try to get management quotes from recent news
                ticker = yf.Ticker(f"{symbol}.NS")
                news = ticker.news

                management_quotes = []
                for article in news[:10]:  # Check recent 10 articles
                    try:
                        if 'summary' in article and article['summary']:
                            summary = article['summary']
                            # Look for quoted management statements
                            if any(keyword in summary.lower() for keyword in [
                                'ceo said', 'management said', 'according to', 'stated',
                                'commented', 'believes', 'expects', 'outlook'
                            ]):
                                management_quotes.append(summary)
                    except Exception:
                        continue

                if management_quotes:
                    alternative_texts.extend(management_quotes)

            except Exception as e:
                logger.debug(f"Error getting earnings content: {e}")

            return alternative_texts if len(alternative_texts) >= 2 else []

        except Exception as e:
            logger.error(f"Error getting alternative management content: {e}")
            return []

    def calculate_position_trading_score(self, data, sentiment_data, fundamentals, trends, market_analysis, sector,
                                         mda_analysis=None):
        """
        Calculate comprehensive position trading score with contextual sentiment
        and other dynamic modifiers.
        """
        try:
            # Get base weights for position trading (fundamentals-heavy)
            fundamental_weight = self.position_trading_params['fundamental_weight']
            technical_weight = self.position_trading_params['technical_weight']
            sentiment_weight = self.position_trading_params['sentiment_weight']
            mda_weight = self.position_trading_params['mda_weight']

            # 1. Calculate all individual base scores
            fundamental_score = self.calculate_fundamental_score(fundamentals, sector)
            technical_score = self.calculate_technical_score_position(data)
            sentiment_score = self.calculate_sentiment_score(sentiment_data)
            trend_score = trends.get('trend_score', 50)
            sector_score = market_analysis.get('sector_score', 60)
            mda_score = mda_analysis.get('mda_score', 50) if mda_analysis and isinstance(mda_analysis, dict) else 50

            # 2. MODIFICATION: Apply the contextual sentiment multiplier based on the sector
            sector_sentiment_multipliers = {
                'Information Technology': 1.2,  # News is highly impactful
                'Consumer Goods': 1.1,
                'Financial Services': 1.1,
                'Pharmaceuticals': 1.2,  # Regulatory news is critical
                'Power': 0.8,  # More stable, less news-driven
                'Oil & Gas': 1.0,
                'Default': 1.0
            }
            sentiment_multiplier = sector_sentiment_multipliers.get(sector, sector_sentiment_multipliers['Default'])
            contextual_sentiment_score = sentiment_score * sentiment_multiplier
            logger.info(f"Applying sentiment multiplier of {sentiment_multiplier} for {sector} sector.")

            # 3. Combine scores using the CONTEXTUAL sentiment score
            base_score = (
                    fundamental_score * fundamental_weight +
                    technical_score * technical_weight +
                    contextual_sentiment_score * sentiment_weight +  # <-- Use the adjusted score here
                    mda_score * mda_weight
            )

            # 4. Apply trend, sector, and other specific modifiers to the final score
            trend_modifier = trend_score / 100
            sector_modifier = sector_score / 100
            final_score = base_score * (0.7 + 0.2 * trend_modifier + 0.1 * sector_modifier)

            # Penalize high volatility stocks
            if data is not None and not data.empty and 'Close' in data.columns:
                volatility = data['Close'].pct_change().std() * np.sqrt(252)
                if volatility > 0.4:
                    final_score *= 0.8
                elif volatility > 0.6:
                    final_score *= 0.6

            # Bonus for dividend-paying stocks
            div_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('expected_div_yield', 0)
            if div_yield and div_yield > 0.02:
                final_score *= 1.1

            # Bonus for consistent long-term performance
            if trends.get('momentum_1y', 0) > 0.15 and trends.get('momentum_6m', 0) > 0:
                final_score *= 1.05

            # MDA sentiment bonus/penalty
            if mda_analysis:
                management_tone = mda_analysis.get('management_tone', 'Neutral')
                if management_tone == 'Very Optimistic':
                    final_score *= 1.08
                elif management_tone == 'Optimistic':
                    final_score *= 1.04
                elif management_tone == 'Pessimistic':
                    final_score *= 0.92

            return min(100, max(0, final_score))

        except Exception as e:
            logger.error(f"Error calculating position trading score: {str(e)}")
            return 0

    def analyze_position_trading_stock(self, symbol, period="5y"):
        """Comprehensive position trading analysis for a single stock with MDA sentiment"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting position trading analysis for {symbol}")

            # Get extended stock data (5 years for position trading)
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Extract basic information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)
            market_cap_category = stock_info.get('market_cap', 'Unknown')

            # Current market data
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                else:
                    price_change = 0
                    price_change_pct = 0
            except Exception as e:
                logger.error(f"Error calculating price changes: {str(e)}")
                return None

            # Fundamental analysis (key for position trading)
            try:
                fundamentals = self.analyze_fundamental_metrics(final_symbol, info)
            except Exception as e:
                logger.error(f"Error in fundamental analysis: {str(e)}")
                fundamentals = {}

            # Long-term trend analysis
            try:
                trends = self.analyze_long_term_trends(data)
            except Exception as e:
                logger.error(f"Error in trend analysis: {str(e)}")
                trends = {'trend_score': 50}

            # Market cycle and sector analysis
            try:
                market_analysis = self.analyze_market_cycles(final_symbol, data)
            except Exception as e:
                logger.error(f"Error in market analysis: {str(e)}")
                market_analysis = {'sector_score': 60}

            # News sentiment analysis using your trained model
            try:
                sentiment_results = self.analyze_news_sentiment(final_symbol)
            except Exception as e:
                logger.error(f"Error in news sentiment analysis: {str(e)}")
                sentiment_results = ([], [], [], "Error", "Error")

            # MDA sentiment analysis
            try:
                # *** FIX 1: Corrected method name ***
                mda_analysis = self.updated_analyze_mda_sentiment(final_symbol)
                logger.info(
                    f"MDA analysis for {symbol}: Score={mda_analysis.get('mda_score', 0):.1f}, Tone={mda_analysis.get('management_tone', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error in MDA sentiment analysis: {str(e)}")
                mda_analysis = self.get_sample_mda_analysis(final_symbol)

            # Risk metrics
            try:
                risk_metrics = self.calculate_risk_metrics(data)
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {str(e)}")
                risk_metrics = {'volatility': 0.3, 'atr': current_price * 0.02}

            # Position trading score (fundamental-heavy with MDA sentiment)
            try:
                position_score = self.calculate_position_trading_score(
                    data, sentiment_results, fundamentals, trends, market_analysis, sector, mda_analysis
                )
            except Exception as e:
                logger.error(f"Error calculating position score: {str(e)}")
                position_score = 0

            # Position trading plan
            try:
                trading_plan = self.generate_position_trading_plan(
                    data, position_score, risk_metrics, fundamentals, trends
                )
            except Exception as e:
                logger.error(f"Error generating trading plan: {str(e)}")
                trading_plan = {'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed'}

            # Compile comprehensive results
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'market_cap_category': market_cap_category,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,

                # Fundamental metrics (key for position trading)
                'fundamentals': fundamentals,
                'fundamental_score': self.calculate_fundamental_score(fundamentals, sector),

                # Long-term trends
                'trends': trends,
                'trend_score': trends.get('trend_score', 50),

                # Market and sector analysis
                'market_analysis': market_analysis,

                # Technical indicators (with longer periods)
                'rsi_30': self.calculate_rsi(data['Close'], period=30).iloc[-1] if len(data) >= 30 else None,
                'ma_50': trends.get('ma_50'),
                'ma_100': trends.get('ma_100'),
                'ma_200': trends.get('ma_200'),

                # News sentiment analysis
                'sentiment': {
                    'scores': sentiment_results[0] if sentiment_results else [],
                    'articles': sentiment_results[1] if sentiment_results else [],
                    'confidence': sentiment_results[2] if sentiment_results else [],
                    'method': sentiment_results[3] if sentiment_results else "Error",
                    'source': sentiment_results[4] if sentiment_results else "Error",
                    'sentiment_summary': self.get_sentiment_summary(sentiment_results[0]) if sentiment_results and
                                                                                             sentiment_results[0] else {
                        'positive': 0, 'negative': 0, 'neutral': 0}
                },

                # MDA sentiment analysis
                'mda_sentiment': mda_analysis,

                # Risk metrics
                'risk_metrics': risk_metrics,

                # Position trading score and plan
                'position_score': position_score,
                'trading_plan': trading_plan,

                # Model information
                'model_type': self.model_type,
                'mda_model_available': self.mda_available,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'Position Trading (Long-term with MDA)'
            }

            logger.info(f"Successfully analyzed {symbol} with position score {position_score}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol} for position trading: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    # Include all the existing methods from your original code
    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database with fundamental data structure"""
        try:
            self.indian_stocks = {
                # NIFTY 50 Stocks with enhanced info
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large",
                             "div_yield": 0.003},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large",
                        "div_yield": 0.025},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
                "INFY": {"name": "Infosys", "sector": "Information Technology", "market_cap": "Large",
                         "div_yield": 0.023},
                "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.014},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking", "market_cap": "Large",
                              "div_yield": 0.005},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services", "market_cap": "Large",
                               "div_yield": 0.002},
                "LT": {"name": "Larsen & Toubro", "sector": "Construction", "market_cap": "Large", "div_yield": 0.018},
                "SBIN": {"name": "State Bank of India", "sector": "Banking", "market_cap": "Large", "div_yield": 0.035},
                "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications", "market_cap": "Large",
                               "div_yield": 0.008},
                "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.008},
                "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.012},
                "TITAN": {"name": "Titan Company", "sector": "Consumer Goods", "market_cap": "Large",
                          "div_yield": 0.005},
                "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals", "market_cap": "Large",
                              "div_yield": 0.008},
                "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement", "market_cap": "Large",
                               "div_yield": 0.005},
                "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods", "market_cap": "Large",
                              "div_yield": 0.008},
                "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology", "market_cap": "Large",
                            "div_yield": 0.025},
                "AXISBANK": {"name": "Axis Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                "WIPRO": {"name": "Wipro", "sector": "Information Technology", "market_cap": "Large",
                          "div_yield": 0.015},
                "NTPC": {"name": "NTPC", "sector": "Power", "market_cap": "Large", "div_yield": 0.045},
                "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power", "market_cap": "Large",
                              "div_yield": 0.038},
                "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas", "market_cap": "Large",
                         "div_yield": 0.055},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology", "market_cap": "Large",
                          "div_yield": 0.032},
                "TATASTEEL": {"name": "Tata Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.025},
                "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate", "market_cap": "Large",
                             "div_yield": 0.001},
                "COALINDIA": {"name": "Coal India", "sector": "Mining", "market_cap": "Large", "div_yield": 0.065},
                "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals", "market_cap": "Large",
                             "div_yield": 0.008},
                "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.012},
                "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.022},
                "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile", "market_cap": "Large",
                        "div_yield": 0.018},
                "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile", "market_cap": "Large",
                               "div_yield": 0.025},
                "GRASIM": {"name": "Grasim Industries", "sector": "Cement", "market_cap": "Large", "div_yield": 0.015},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement", "market_cap": "Large", "div_yield": 0.003},
                "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile", "market_cap": "Large",
                              "div_yield": 0.005},
                "UPL": {"name": "UPL Limited", "sector": "Chemicals", "market_cap": "Large", "div_yield": 0.012},
                "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.035},
                "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large",
                             "div_yield": 0.005},
                "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large",
                            "div_yield": 0.008},
                "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.012},
                "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods", "market_cap": "Large",
                              "div_yield": 0.008},
                "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.015},
                "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas", "market_cap": "Large",
                        "div_yield": 0.042},
                "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare", "market_cap": "Large",
                               "div_yield": 0.002},
                "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services", "market_cap": "Large",
                               "div_yield": 0.008},
                "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance", "market_cap": "Large",
                             "div_yield": 0.012},
                "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance", "market_cap": "Large",
                            "div_yield": 0.008},
                "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.015},
                "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure", "market_cap": "Large",
                               "div_yield": 0.012},
                "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile", "market_cap": "Large",
                               "div_yield": 0.008},
                "ITC": {"name": "ITC Limited", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.055},

                # Additional Mid & Small Cap Stocks
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods", "market_cap": "Mid",
                             "div_yield": 0.012},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods", "market_cap": "Mid",
                           "div_yield": 0.008},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals", "market_cap": "Mid",
                               "div_yield": 0.005},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods", "market_cap": "Mid",
                           "div_yield": 0.018},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.012},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals", "market_cap": "Mid",
                          "div_yield": 0.008},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals", "market_cap": "Mid",
                           "div_yield": 0.005},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile", "market_cap": "Mid",
                               "div_yield": 0.012},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile", "market_cap": "Mid",
                             "div_yield": 0.008},
                "MRF": {"name": "MRF Limited", "sector": "Automobile", "market_cap": "Mid", "div_yield": 0.015},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.001},
                "TRENT": {"name": "Trent Limited", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.002},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles", "market_cap": "Mid", "div_yield": 0.003},
            }

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

            logger.info(f"Initialized database with {len(self.indian_stocks)} Indian stocks")

        except Exception as e:
            logger.error(f"Error initializing stock database: {str(e)}")
            # Fallback to minimal database
            self.indian_stocks = {
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large",
                             "div_yield": 0.003},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large",
                        "div_yield": 0.025},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    

    def get_indian_stock_data(self, symbol, period="5y"):
        """Fetches stock data reliably from the EODHD API."""
        try:
            if not self.eodhd_api_key:
                raise ValueError("EODHD_API_KEY not set in environment.")
            
            symbol = str(symbol).upper().replace(".NS", "").replace(".BO", "")
            
            # EODHD uses .INDX for NSE stocks
            api_symbol = f"{symbol}.INDX" 
            
            logger.info(f"Trying to fetch data for {api_symbol} from EODHD")
            
            eod_client = EODHDAPIWrapper(self.eodhd_api_key)
            
            # The API returns a list of dictionaries, which we'll convert to a DataFrame
            resp = eod_client.get_historical_data(api_symbol, 'd') # 'd' for daily
            
            if not resp:
                logger.error(f"No data returned from EODHD for {api_symbol}")
                return None, None, None
                
            data = pd.DataFrame(resp).set_index('date')
            data.index = pd.to_datetime(data.index)
            
            # Rename columns to match the format the rest of your code expects
            data = data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'adjusted_close': 'Close', 'volume': 'Volume'
            })
            
            # For position trading, we need at least a year of data
            if len(data) < 252:
                 logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
                 return None, None, None

            # Get company info as a fallback
            info = {}
            try:
                info = yf.Ticker(f"{symbol}.NS").info
            except Exception:
                info = {'shortName': self.get_stock_info_from_db(symbol).get('name', symbol)}
                                
            logger.info(f"Successfully fetched {len(data)} days of data for {api_symbol}")
            return data, info, api_symbol

        except Exception as e:
            logger.error(f"Critical error in get_indian_stock_data for {symbol}: {e}")
            return None, None, None

    def analyze_fundamental_metrics(self, symbol, info):
        """Analyze fundamental metrics crucial for position trading"""
        try:
            fundamentals = {
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'dividend_yield': info.get('dividendYield', None),
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'book_value': info.get('bookValue', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None)
            }

            # Add sector-specific metrics from our database
            stock_info = self.get_stock_info_from_db(symbol)
            fundamentals['expected_div_yield'] = stock_info.get('div_yield', 0)
            fundamentals['market_cap_category'] = stock_info.get('market_cap', 'Unknown')

            return fundamentals
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {str(e)}")
            return {}

    def calculate_fundamental_score(self, fundamentals, sector):
        """Calculate fundamental score for position trading (0-100)"""
        try:
            score = 0
            max_score = 100

            # P/E Ratio Analysis (15 points)
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio and 8 < pe_ratio < 25:
                score += 15
            elif pe_ratio and 5 < pe_ratio <= 8:
                score += 12  # Undervalued
            elif pe_ratio and 25 <= pe_ratio < 35:
                score += 8  # Slightly expensive

            # PEG Ratio Analysis (15 points)
            peg_ratio = fundamentals.get('peg_ratio')
            if peg_ratio and 0.5 < peg_ratio < 1.0:
                score += 15  # Undervalued growth
            elif peg_ratio and 1.0 <= peg_ratio < 1.5:
                score += 10  # Fair value

            # Revenue and Earnings Growth (20 points)
            revenue_growth = fundamentals.get('revenue_growth', 0)
            earnings_growth = fundamentals.get('earnings_growth', 0)

            if revenue_growth and revenue_growth > 0.20:  # 20% revenue growth
                score += 10
            elif revenue_growth and revenue_growth > 0.15:
                score += 8
            elif revenue_growth and revenue_growth > 0.10:
                score += 5

            if earnings_growth and earnings_growth > 0.25:  # 25% earnings growth
                score += 10
            elif earnings_growth and earnings_growth > 0.15:
                score += 8
            elif earnings_growth and earnings_growth > 0.10:
                score += 5

            # ROE Analysis (10 points)
            roe = fundamentals.get('roe')
            if roe and roe > 0.20:  # 20% ROE
                score += 10
            elif roe and roe > 0.15:
                score += 8
            elif roe and roe > 0.12:
                score += 5

            # Debt Analysis (10 points)
            debt_equity = fundamentals.get('debt_to_equity')
            if debt_equity is not None:
                if debt_equity < 0.3:
                    score += 10  # Very low debt
                elif debt_equity < 0.6:
                    score += 8  # Manageable debt
                elif debt_equity < 1.0:
                    score += 4  # High but acceptable

            # Profitability Margins (10 points)
            profit_margin = fundamentals.get('profit_margin')
            operating_margin = fundamentals.get('operating_margin')

            if profit_margin and profit_margin > 0.15:
                score += 5
            elif profit_margin and profit_margin > 0.10:
                score += 3

            if operating_margin and operating_margin > 0.20:
                score += 5
            elif operating_margin and operating_margin > 0.15:
                score += 3

            # Dividend Yield (10 points) - Important for position trading
            div_yield = fundamentals.get('dividend_yield') or fundamentals.get('expected_div_yield', 0)
            if div_yield and div_yield > 0.03:  # 3% dividend yield
                score += 10
            elif div_yield and div_yield > 0.015:
                score += 6
            elif div_yield and div_yield > 0.005:
                score += 3

            # Financial Health (10 points)
            current_ratio = fundamentals.get('current_ratio')
            if current_ratio and current_ratio > 1.5:
                score += 5
            elif current_ratio and current_ratio > 1.2:
                score += 3

            price_to_book = fundamentals.get('price_to_book')
            if price_to_book and price_to_book < 2.0:
                score += 5
            elif price_to_book and price_to_book < 3.0:
                score += 3

            return min(score, max_score)

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {str(e)}")
            return 0

    def analyze_long_term_trends(self, data):
        """Analyze long-term trends for position trading"""
        try:
            # Multiple timeframe moving averages for position trading
            ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean')
            ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean')
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')

            current_price = data['Close'].iloc[-1]

            # Trend strength analysis
            trend_score = 0

            # Price above all major MAs (Strong uptrend)
            if (current_price > ma_50.iloc[-1] > ma_100.iloc[-1] > ma_200.iloc[-1]):
                trend_score = 100
            elif (current_price > ma_50.iloc[-1] > ma_100.iloc[-1]):
                trend_score = 75  # Moderate uptrend
            elif (current_price > ma_100.iloc[-1]):
                trend_score = 50  # Weak uptrend
            elif (current_price < ma_50.iloc[-1] < ma_100.iloc[-1] < ma_200.iloc[-1]):
                trend_score = 0  # Strong downtrend
            else:
                trend_score = 25  # Sideways/mixed

            # Calculate trend momentum (slope of moving averages)
            ma_50_slope = (ma_50.iloc[-1] - ma_50.iloc[-20]) / ma_50.iloc[-20] if len(ma_50) > 20 else 0
            ma_200_slope = (ma_200.iloc[-1] - ma_200.iloc[-50]) / ma_200.iloc[-50] if len(ma_200) > 50 else 0

            # Long-term price momentum
            price_6m_ago = data['Close'].iloc[-126] if len(data) > 126 else data['Close'].iloc[0]
            price_1y_ago = data['Close'].iloc[-252] if len(data) > 252 else data['Close'].iloc[0]

            momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            momentum_1y = (current_price - price_1y_ago) / price_1y_ago

            return {
                'trend_score': trend_score,
                'ma_50_slope': ma_50_slope,
                'ma_200_slope': ma_200_slope,
                'above_ma_200': current_price > ma_200.iloc[-1],
                'momentum_6m': momentum_6m,
                'momentum_1y': momentum_1y,
                'ma_50': ma_50.iloc[-1],
                'ma_100': ma_100.iloc[-1],
                'ma_200': ma_200.iloc[-1]
            }

        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {str(e)}")
            return {'trend_score': 50, 'ma_50_slope': 0, 'ma_200_slope': 0, 'above_ma_200': False,
                    'momentum_6m': 0, 'momentum_1y': 0, 'ma_50': 0, 'ma_100': 0, 'ma_200': 0}

    def analyze_market_cycles(self, symbol, data):
        """Analyze market cycles and sector rotation"""
        try:
            sector = self.get_stock_info_from_db(symbol)['sector']

            # Sector strength analysis based on current market conditions
            sector_score = 0

            # Interest rate sensitive sectors
            if sector in ['Banking', 'Financial Services']:
                # Banks benefit from rising rates, hurt by falling rates
                sector_score = 65  # Neutral to positive
            elif sector in ['Real Estate', 'Infrastructure']:
                # Real estate hurt by rising rates
                sector_score = 55

            # Defensive sectors (good for position trading)
            elif sector in ['Consumer Goods', 'Pharmaceuticals', 'Healthcare']:
                sector_score = 75  # Generally stable for position trading

            # Cyclical sectors
            elif sector in ['Automobile', 'Steel', 'Cement', 'Metals']:
                sector_score = 60  # Depends on economic cycle

            # Growth sectors
            elif sector in ['Information Technology']:
                sector_score = 70  # Good for position trading

            # Utility and Power
            elif sector in ['Power', 'Utilities']:
                sector_score = 80  # Excellent for position trading (stable dividends)

            # Commodity sectors
            elif sector in ['Oil & Gas', 'Mining']:
                sector_score = 55  # Volatile but can be good long-term

            else:
                sector_score = 60  # Default for other sectors

            return {
                'sector_score': sector_score,
                'sector': sector,
                'cycle_stage': self.determine_market_cycle(data),
                'sector_preference': self.get_sector_preference(sector)
            }

        except Exception as e:
            logger.error(f"Error in market cycle analysis: {str(e)}")
            return {'sector_score': 60, 'sector': 'Unknown', 'cycle_stage': 'Unknown', 'sector_preference': 'Neutral'}

    def determine_market_cycle(self, data):
        """Determine current market cycle stage"""
        try:
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')
            current_price = data['Close'].iloc[-1]

            # Check if price has been above MA200 for extended period
            above_ma_200_days = 0
            check_period = min(120, len(data))  # Check last 120 days

            for i in range(check_period):
                if data['Close'].iloc[-(i + 1)] > ma_200.iloc[-(i + 1)]:
                    above_ma_200_days += 1

            above_ma_200_pct = above_ma_200_days / check_period

            if above_ma_200_pct > 0.75:
                return "Bull Market"
            elif above_ma_200_pct < 0.25:
                return "Bear Market"
            else:
                return "Transitional"

        except Exception as e:
            logger.error(f"Error determining market cycle: {str(e)}")
            return "Unknown"

    def get_sector_preference(self, sector):
        """Get sector preference for position trading"""
        high_preference = ['Consumer Goods', 'Information Technology', 'Healthcare',
                           'Pharmaceuticals', 'Power', 'Banking']
        medium_preference = ['Telecommunications', 'Oil & Gas', 'Chemicals', 'Cement']

        if sector in high_preference:
            return 'High'
        elif sector in medium_preference:
            return 'Medium'
        else:
            return 'Low'

    def calculate_technical_score_position(self, data):
        """Calculate technical score optimized for position trading"""
        try:
            if data is None or data.empty:
                return 0

            technical_score = 0
            current_price = data['Close'].iloc[-1]

            # RSI Analysis with longer period (30 points)
            rsi = self.calculate_rsi(data['Close'], period=30)  # Longer period for position trading
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                current_rsi = rsi.iloc[-1]
                if 40 <= current_rsi <= 60:  # Neutral zone - good for position trading
                    technical_score += 30
                elif 30 <= current_rsi < 40:  # Slight oversold
                    technical_score += 25
                elif 60 < current_rsi <= 70:  # Slight overbought
                    technical_score += 20
                elif current_rsi < 30:  # Very oversold
                    technical_score += 15

            # Moving Average Analysis (25 points)
            if len(data) >= 200:
                ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
                ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]

                if not any(pd.isna([ma_50, ma_100, ma_200])):
                    if current_price > ma_50 > ma_100 > ma_200:  # Perfect alignment
                        technical_score += 25
                    elif current_price > ma_50 > ma_100:  # Good alignment
                        technical_score += 20
                    elif current_price > ma_200:  # Above long-term trend
                        technical_score += 15
                    elif ma_50 > ma_100:  # Short term stronger than medium term
                        technical_score += 10

            # Volume Trend Analysis (15 points)
            if 'Volume' in data.columns and len(data) >= 50:
                recent_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                long_term_volume = self.safe_rolling_calculation(data['Volume'], 50, 'mean').iloc[-1]

                if not pd.isna(recent_volume) and not pd.isna(long_term_volume) and long_term_volume > 0:
                    volume_ratio = recent_volume / long_term_volume
                    if volume_ratio > 1.2:  # Increasing volume
                        technical_score += 15
                    elif volume_ratio > 1.0:
                        technical_score += 10

            # Long-term MACD (15 points)
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'], fast=26, slow=52, signal=18)
            if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                    technical_score += 15
                if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                    if histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                        technical_score += 5

            # Support/Resistance for position entries (15 points)
            support, resistance = self.calculate_support_resistance(data, window=50)  # Longer window
            if support and resistance and not any(pd.isna([support, resistance])):
                distance_to_support = (current_price - support) / support
                distance_to_resistance = (resistance - current_price) / current_price

                if 0.05 <= distance_to_support <= 0.20:  # Good entry zone above support
                    technical_score += 15
                elif distance_to_support > 0.20:  # Well above support
                    technical_score += 10
                elif distance_to_support < 0.05:  # Very close to support
                    technical_score += 8

            return min(100, max(0, technical_score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0

    def calculate_sentiment_score(self, sentiment_data):
        """Calculate sentiment score from news analysis"""
        try:
            if not sentiment_data or len(sentiment_data) < 3:
                return 50  # Neutral sentiment

            sentiments, _, confidences, _, _ = sentiment_data
            if not sentiments or not confidences:
                return 50

            sentiment_value = 0
            total_weight = 0

            for sentiment, confidence in zip(sentiments, confidences):
                weight = confidence if not pd.isna(confidence) else 0.5
                if sentiment == 'positive':
                    sentiment_value += weight
                elif sentiment == 'negative':
                    sentiment_value -= weight
                # neutral adds 0
                total_weight += weight

            if total_weight > 0:
                normalized_sentiment = sentiment_value / total_weight
                sentiment_score = 50 + (normalized_sentiment * 50)  # Scale to 0-100
            else:
                sentiment_score = 50

            return min(100, max(0, sentiment_score))

        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 50

    def generate_position_trading_plan(self, data, score, risk_metrics, fundamentals, trends):
        """
        Generate realistic position trading plan for 6-18 month holding periods.
        Position trading targets are more conservative than swing trading due to:
        - Longer holding periods allow fundamental value to materialize
        - Lower transaction costs justify patience
        - Focus on quality over quick gains
        """
        default_plan = {
            'entry_signal': 'ERROR',
            'entry_strategy': 'Analysis failed',
            'entry_timing': 'Unknown',
            'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'support': 0,
            'resistance': 0,
            'holding_period': 'Unknown',
            'trade_management_note': 'Unable to generate plan',
            'stop_distance_pct': 0,
            'upside_potential': 0
        }

        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)

            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02

            # Entry signal and strategy based on score
            if score >= 80:
                entry_signal = "STRONG BUY"
                entry_strategy = "Accumulate on dips; high conviction core holding"
                holding_period = "12-24 months"
            elif score >= 65:
                entry_signal = "BUY"
                entry_strategy = "Enter gradually over 2-4 weeks; good long-term prospects"
                holding_period = "9-18 months"
            elif score >= 40:
                entry_signal = "HOLD/WATCH"
                entry_strategy = "Wait for better entry or more confirmation"
                holding_period = "Monitor for opportunity"
            else:
                entry_signal = "AVOID"
                entry_strategy = "Fundamental or technical weaknesses detected"
                holding_period = "Not recommended"

            # Stop-Loss: Wider for position trading (7-10% typical)
            # Use 3x ATR, capped at 10% to limit downside
            stop_loss_distance = min(atr * 3.0, current_price * 0.10)
            stop_loss = max(current_price - stop_loss_distance, 0)

            # Support and Resistance (longer 100-day window for position trading)
            support, resistance = self.calculate_support_resistance(data, window=100)
            if not support or pd.isna(support):
                support = current_price * 0.90
            if not resistance or pd.isna(resistance):
                resistance = current_price * 1.15

            # Adjust stop-loss to be above major support
            if support > 0:
                stop_loss = max(stop_loss, support * 0.97)

            # REALISTIC POSITION TRADING TARGETS
            # For 6-18 month holds, expecting 15-35% total gains is reasonable
            # Using progressive risk-reward ratios: 1.5:1, 2.5:1, 3.5:1

            target_1 = current_price + (stop_loss_distance * 1.5)  # ~10-15% gain
            target_2 = current_price + (stop_loss_distance * 2.5)  # ~17-25% gain
            target_3 = current_price + (stop_loss_distance * 3.5)  # ~24-35% gain

            # Cap targets at realistic resistance levels
            # Don't set targets more than 35% above current price
            max_reasonable_target = current_price * 1.35

            target_1 = min(target_1, current_price * 1.15)  # Cap at 15%
            target_2 = min(target_2, current_price * 1.25)  # Cap at 25%
            target_3 = min(target_3, max_reasonable_target)  # Cap at 35%

            # Ensure targets are progressive and above entry
            if target_1 <= current_price:
                target_1 = current_price * 1.10
            if target_2 <= target_1:
                target_2 = current_price * 1.20
            if target_3 <= target_2:
                target_3 = current_price * 1.30

            # Entry Timing based on trend
            ma_200 = trends.get('ma_200', current_price)
            if current_price > ma_200 * 1.02:
                entry_timing = "Buy on pullbacks to support or continue accumulating"
            elif current_price > ma_200 * 0.98:
                entry_timing = "Price near long-term trend; wait for bullish confirmation"
            else:
                entry_timing = "Wait for price to reclaim 200-day moving average"

            # Calculate metrics
            stop_distance_pct = ((current_price - stop_loss) / current_price) * 100
            upside_potential = ((target_2 - current_price) / current_price) * 100
            risk_reward_ratio = upside_potential / stop_distance_pct if stop_distance_pct > 0 else 0

            # Trade management note
            trade_management_note = (
                "Position trading approach: Book 1/3 position at each target. "
                "Trail stop loss to breakeven after Target 1. "
                "Review quarterly and adjust based on fundamental changes."
            )

            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'entry_timing': entry_timing,
                'stop_loss': round(stop_loss, 2),
                'targets': {
                    'target_1': round(target_1, 2),
                    'target_2': round(target_2, 2),
                    'target_3': round(target_3, 2)
                },
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'holding_period': holding_period,
                'trade_management_note': trade_management_note,
                'stop_distance_pct': round(stop_distance_pct, 2),
                'upside_potential': round(upside_potential, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }

        except Exception as e:
            logger.error(f"Error generating position trading plan: {str(e)}")
            logger.error(traceback.format_exc())
            return default_plan
    # Helper methods for technical analysis
    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations"""
        try:
            if data is None or data.empty:
                return pd.Series(dtype=float)

            if len(data) < window:
                return pd.Series([np.nan] * len(data), index=data.index)

            if operation == 'mean':
                return data.rolling(window=window, min_periods=1).mean()
            elif operation == 'std':
                return data.rolling(window=window, min_periods=1).std()
            elif operation == 'min':
                return data.rolling(window=window, min_periods=1).min()
            elif operation == 'max':
                return data.rolling(window=window, min_periods=1).max()
            else:
                logger.error(f"Unknown rolling operation: {operation}")
                return pd.Series([np.nan] * len(data), index=data.index)

        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {str(e)}")
            return pd.Series([np.nan] * len(data), index=data.index if hasattr(data, 'index') else range(len(data)))

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            if prices is None or prices.empty:
                return pd.Series(dtype=float)

            if len(prices) < period:
                return pd.Series([50] * len(prices), index=prices.index)

            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = self.safe_rolling_calculation(gain, period, 'mean')
            avg_loss = self.safe_rolling_calculation(loss, period, 'mean')

            if avg_gain.empty or avg_loss.empty:
                return pd.Series([50] * len(prices), index=prices.index)

            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < slow:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()

            if exp1.empty or exp2.empty:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            zeros = pd.Series([0] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return zeros, zeros, zeros

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels"""
        try:
            if data is None or data.empty:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                logger.error("Missing High/Low columns for support/resistance calculation")
                return None, None

            if len(data) < window:
                return data['Low'].min(), data['High'].max()

            highs = self.safe_rolling_calculation(data['High'], window, 'max')
            lows = self.safe_rolling_calculation(data['Low'], window, 'min')

            if highs.empty or lows.empty:
                return data['Low'].min(), data['High'].max()

            # Find significant levels
            resistance_levels = []
            support_levels = []

            for i in range(window, len(data)):
                try:
                    if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                        resistance_levels.append(data['High'].iloc[i])

                    if not pd.isna(lows.iloc[i]) and data['Low'].iloc[i] == lows.iloc[i]:
                        support_levels.append(data['Low'].iloc[i])
                except Exception as e:
                    logger.warning(f"Error processing level at index {i}: {str(e)}")
                    continue

            # Get most recent levels
            if len(resistance_levels) >= 3:
                current_resistance = max(resistance_levels[-3:])
            else:
                current_resistance = data['High'].max()

            if len(support_levels) >= 3:
                current_support = min(support_levels[-3:])
            else:
                current_support = data['Low'].min()

            return current_support, current_resistance

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            try:
                return data['Low'].min(), data['High'].max()
            except:
                return None, None

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics"""
        default_metrics = {
            'volatility': 0.3,
            'var_95': -0.05,
            'max_drawdown': -0.2,
            'sharpe_ratio': 0,
            'atr': 0,
            'risk_level': 'HIGH'
        }

        try:
            if data is None or data.empty or 'Close' not in data.columns:
                logger.error("Invalid data for risk metrics calculation")
                return default_metrics

            returns = data['Close'].pct_change().dropna()

            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient returns data for risk metrics")
                return default_metrics

            # Volatility (annualized)
            try:
                volatility = returns.std() * np.sqrt(252)
                if pd.isna(volatility) or volatility < 0:
                    volatility = 0.3
            except Exception:
                volatility = 0.3

            # Value at Risk (95% confidence)
            try:
                var_95 = np.percentile(returns.dropna(), 5)
                if pd.isna(var_95):
                    var_95 = -0.05
            except Exception:
                var_95 = -0.05

            # Maximum Drawdown
            try:
                rolling_max = data['Close'].expanding().max()
                drawdown = (data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                if pd.isna(max_drawdown):
                    max_drawdown = -0.2
            except Exception:
                max_drawdown = -0.2

            # Sharpe Ratio
            try:
                risk_free_rate = 0.06
                excess_returns = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                if pd.isna(sharpe_ratio):
                    sharpe_ratio = 0
            except Exception:
                sharpe_ratio = 0

            # ATR for position sizing
            try:
                if len(data) >= 14 and all(col in data.columns for col in ['High', 'Low', 'Close']):
                    high_low = data['High'] - data['Low']
                    high_close = np.abs(data['High'] - data['Close'].shift())
                    low_close = np.abs(data['Low'] - data['Close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean().iloc[-1]
                    if pd.isna(atr):
                        atr = data['Close'].iloc[-1] * 0.02
                else:
                    atr = data['Close'].iloc[-1] * 0.02
            except Exception:
                atr = data['Close'].iloc[-1] * 0.02 if not data['Close'].empty else 0

            # Risk level determination (adjusted for position trading)
            try:
                if volatility > 0.40:
                    risk_level = 'HIGH'
                elif volatility > 0.25:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
            except Exception:
                risk_level = 'HIGH'

            return {
                'volatility': volatility,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'atr': atr,
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return default_metrics

    # News sentiment methods
    def analyze_news_sentiment(self, symbol, num_articles=20):
        """Main sentiment analysis function updated to use the API."""
        try:
            articles = self.fetch_indian_news(symbol, num_articles) or self.get_sample_news(symbol)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"

            if not articles:
                return [], [], [], "No Analysis", "No Source"

            if self.sentiment_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    return sentiments, articles, confidences, "SBERT API", news_source

            logging.warning(f"Falling back to TextBlob for news sentiment for {symbol}.")
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob Fallback", news_source
        except Exception as e:
            logging.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error"

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob"""
        sentiments = []
        confidences = []

        if not articles:
            return sentiments, confidences

        for article in articles:
            try:
                if not article or not isinstance(article, str):
                    sentiments.append('neutral')
                    confidences.append(0.3)
                    continue

                blob = TextBlob(article)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiments.append('positive')
                    confidences.append(min(abs(polarity), 0.8))
                elif polarity < -0.1:
                    sentiments.append('negative')
                    confidences.append(min(abs(polarity), 0.8))
                else:
                    sentiments.append('neutral')
                    confidences.append(0.5)
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for article: {str(e)}")
                sentiments.append('neutral')
                confidences.append(0.3)

        return sentiments, confidences

    def get_sentiment_summary(self, sentiment_scores):
        """Get summary of sentiment scores"""
        if not sentiment_scores:
            return {'positive': 0, 'negative': 0, 'neutral': 0}

        return {
            'positive': sentiment_scores.count('positive'),
            'negative': sentiment_scores.count('negative'),
            'neutral': sentiment_scores.count('neutral')
        }

    # Utility methods
    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {str(e)}")
            return ["RELIANCE", "TCS", "HDFCBANK"]

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol:
                raise ValueError("Invalid symbol format")

            return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown", "market_cap": "Unknown",
                                                        "div_yield": 0})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {"name": str(symbol), "sector": "Unknown", "market_cap": "Unknown", "div_yield": 0}

    def fetch_indian_news(self, symbol, num_articles=20):
        """Fetch news for Indian companies"""
        try:
            if not self.news_api_key:
                return None

            base_symbol = str(symbol).split('.')[0].upper()
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            url = f"https://newsapi.org/v2/everything?q={company_name}+India+stock&apiKey={self.news_api_key}&pageSize={num_articles}&language=en&sortBy=publishedAt"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', []):
                    if article.get('title'):
                        articles.append(article['title'])
                return articles if articles else None
            else:
                logger.warning(f"News API returned status code: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("News API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"News API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return None

    # Add this method to the EnhancedPositionTradingSystem class
    # In class EnhancedPositionTradingSystem:

    def create_personalized_portfolio(self, risk_appetite, time_period_months, budget):
        """Create a personalized portfolio using risk-based position sizing."""
        try:
            min_score = 65  # Set a minimum score for a trade to even be considered

            symbols = self.get_all_stock_symbols()
            stock_results = []

            print(f"\nAnalyzing {len(symbols)} stocks for your portfolio...")
            for symbol in symbols:
                result = self.analyze_position_trading_stock(symbol)
                # Only consider stocks with a BUY signal and a high enough score
                if result and result.get('position_score', 0) >= min_score and \
                        result.get('trading_plan', {}).get('entry_signal') in ['BUY', 'STRONG BUY']:
                    stock_results.append(result)

            # Sort by score to prioritize the best setups first
            sorted_stocks = sorted(stock_results, key=lambda x: x['position_score'], reverse=True)

            if not sorted_stocks:
                return {"error": "No stocks meet the minimum criteria for your risk profile."}

            # --- UPDATED: Pass the entire budget to the new risk-based calculator ---
            portfolio = self.calculate_position_sizes(sorted_stocks, budget)

            if not portfolio:
                return {"error": "Could not create a portfolio with the given risk parameters and budget."}

            summary = self.generate_portfolio_summary(portfolio, time_period_months)

            return {
                'portfolio': portfolio,
                'summary': summary,
                'risk_profile': risk_appetite,
                'time_period_months': time_period_months,
                'budget': budget
            }

        except Exception as e:
            logger.error(f"Error creating personalized portfolio: {str(e)}")
            return {"error": str(e)}

    def calculate_position_sizes(self, selected_stocks, total_capital):
        """
        --- COMPLETELY REWRITTEN ---
        Calculate position sizes based on a fixed risk percentage of total capital.
        """
        portfolio = {}

        # Get the risk per trade from your class parameters (e.g., 0.01 for 1%)
        risk_per_trade_pct = self.position_trading_params['risk_per_trade']
        capital_at_risk_per_trade = total_capital * risk_per_trade_pct

        total_allocated = 0

        for stock_data in selected_stocks:
            try:
                current_price = stock_data.get('current_price', 0)
                trading_plan = stock_data.get('trading_plan', {})
                stop_loss = trading_plan.get('stop_loss', 0)

                # Validate data for this trade
                if current_price <= 0 or stop_loss <= 0 or current_price <= stop_loss:
                    continue

                # --- Core Risk-Based Calculation ---
                risk_per_share = current_price - stop_loss
                num_shares = int(capital_at_risk_per_trade / risk_per_share)

                if num_shares == 0:
                    continue  # Cannot afford even one share with this risk model

                investment_amount = num_shares * current_price

                # Ensure we don't allocate more than the total available capital
                if total_allocated + investment_amount > total_capital:
                    continue  # Skip trade if it exceeds total budget

                total_allocated += investment_amount

                symbol = stock_data.get('symbol', 'Unknown')
                portfolio[symbol] = {
                    'company_name': stock_data.get('company_name', 'Unknown'),
                    'sector': stock_data.get('sector', 'Unknown'),
                    'score': stock_data.get('position_score', 0),
                    'num_shares': num_shares,
                    'investment_amount': investment_amount,
                    'stop_loss': stop_loss,
                    'targets': trading_plan.get('targets')
                }

            except Exception as e:
                logger.error(f"Error sizing position for {stock_data.get('symbol')}: {e}")
                continue

        return portfolio

    def generate_portfolio_summary(self, portfolio, time_period_months):
        """Generate a summary of the portfolio"""
        total_investment = sum(stock['investment_amount'] for stock in portfolio.values())
        avg_score = sum(stock['score'] for stock in portfolio.values()) / len(portfolio)

        # Sector allocation
        sector_allocation = {}
        for stock in portfolio.values():
            sector = stock['sector']
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += stock['investment_amount']

        # Convert to percentages
        for sector in sector_allocation:
            sector_allocation[sector] = sector_allocation[sector] / total_investment * 100

        # Expected return (simplified)
        expected_return = avg_score / 100 * 0.15  # Assume 15% max return for perfect score

        # Adjust for time period (longer time period generally means higher expected returns)
        time_factor = min(2.0, 1.0 + (time_period_months / 12) * 0.1)  # 10% per year additional
        expected_return *= time_factor

        return {
            'total_investment': total_investment,
            'number_of_stocks': len(portfolio),
            'average_score': avg_score,
            'sector_allocation': sector_allocation,
            'expected_return': expected_return,
            'expected_return_percentage': expected_return * 100,
            'recommended_holding_period': f"{time_period_months} months"
        }

    # *** FIX 2: Added the missing method ***
    def get_sample_mda_analysis(self, symbol):
        """Generate sample MDA analysis for demonstration or fallback"""
        try:
            # Generate a score based on a hash of the symbol for consistency
            base_score = 50 + (hash(symbol) % 25)
            tone_map = {
                (0, 45): "Pessimistic",
                (45, 55): "Neutral",
                (55, 65): "Optimistic",
                (65, 100): "Very Optimistic"
            }
            management_tone = "Neutral"
            for (lower, upper), tone in tone_map.items():
                if lower <= base_score < upper:
                    management_tone = tone
                    break

            return {
                'mda_score': base_score,
                'sentiment_distribution': {'positive': 0.4, 'negative': 0.1, 'neutral': 0.5},
                'management_tone': management_tone,
                'confidence': 0.75,
                'analysis_method': 'Sample MDA Analysis (Fallback)',
                'sample_texts_analyzed': 0,
                'text_sources': 'No real text found; using sample data.'
            }
        except Exception as e:
            logger.error(f"Error generating sample MDA analysis for {symbol}: {str(e)}")
            return {'mda_score': 50, 'management_tone': 'Neutral', 'analysis_method': 'Error'}

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration"""
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            return [
                f"{company_name} reports strong quarterly earnings growth",
                f"Analysts upgrade {company_name} with positive long-term outlook",
                f"{company_name} announces strategic expansion and investment plans",
                f"Strong fundamentals make {company_name} attractive for long-term investors",
                f"I{company_name} dividend policy supports income-focused portfolios",
                f"Management guidance remains optimistic for {company_name}",
                f"Institutional investors increase holdings in {company_name}",
                f"{company_name} well-positioned for sector growth trends",
                f"ESG initiatives strengthen {company_name} investment case",
                f"Market leadership solidifies {company_name} competitive advantage",
                f"{company_name} balance sheet strength provides stability",
                f"Innovation pipeline drives {company_name} future growth",
                f"Regulatory tailwinds benefit {company_name} business model",
                f"{company_name} demonstrates resilient performance in volatile markets",
                f"Long-term demographic trends favor {company_name} prospects"
            ]
        except Exception as e:
            logger.error(f"Error generating sample news for {symbol}: {str(e)}")
            return [f"Long-term analysis for {symbol}", f"Investment opportunity in {symbol}"]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)











