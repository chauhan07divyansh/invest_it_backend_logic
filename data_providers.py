import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import backoff
import redis
import json
import pandas as pd

# Fyers API V3
from fyers_apiv3 import fyersModel

logger = logging.getLogger(__name__)


class FyersProvider:
    """Handles all Fyers API V3 interactions for OHLCV data with retry logic"""

    def __init__(self, app_id: str, access_token: str):
        """
        Initialize Fyers provider

        Args:
            app_id: Fyers App ID
            access_token: Fyers Access Token (generated via OAuth)
        """
        self.app_id = app_id
        self.access_token = access_token

        # Initialize Fyers session
        self.fyers = fyersModel.FyersModel(
            client_id=app_id,
            token=access_token,
            log_path=os.getenv('FYERS_LOG_PATH', '')
        )

        logger.info("✅ FyersProvider initialized successfully")

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, Exception),
        max_tries=3,
        max_time=30
    )
    def get_historical_data(
            self,
            symbol: str,
            exchange: str = 'NSE',
            period: str = '6mo',
            resolution: str = 'D'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Fyers API

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            exchange: Exchange (NSE or BSE)
            period: Time period (e.g., '6mo', '1y', '5y')
            resolution: Data resolution ('D' for daily, '1' for 1min, etc.)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Convert symbol to Fyers format: NSE:SYMBOL-EQ or BSE:SYMBOL-EQ
            fyers_symbol = f"{exchange}:{symbol}-EQ"

            # Calculate date range
            to_date = datetime.now()
            period_map = {
                '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365,
                '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650
            }
            days = period_map.get(period, 180)
            from_date = to_date - timedelta(days=days)

            # Prepare data request
            data_request = {
                "symbol": fyers_symbol,
                "resolution": resolution,
                "date_format": "1",  # Unix timestamp
                "range_from": from_date.strftime('%Y-%m-%d'),
                "range_to": to_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }

            logger.info(f"Fetching Fyers data for {fyers_symbol} from {from_date.date()} to {to_date.date()}")

            # Make API call
            response = self.fyers.history(data_request)

            if response.get('s') != 'ok':
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"Fyers API error for {fyers_symbol}: {error_msg}")
                return None

            # Parse response
            candles = response.get('candles', [])
            if not candles:
                logger.warning(f"No data returned from Fyers for {fyers_symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.drop('timestamp', axis=1)
            df = df.set_index('Date')

            logger.info(f"✅ Retrieved {len(df)} rows from Fyers for {fyers_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching Fyers data for {symbol}: {e}")
            return None


class ScreenerProvider:
    """Handles web scraping of fundamental data from Screener.in with politeness"""

    def __init__(self, delay_seconds: float = 1.5):
        """
        Initialize Screener.in scraper

        Args:
            delay_seconds: Delay between requests to be polite
        """
        self.base_url = "https://www.screener.in/company"
        self.delay = delay_seconds
        self.last_request_time = 0

        # Set up session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        logger.info("✅ ScreenerProvider initialized successfully")

    def _rate_limit(self):
        """Implement polite rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException,),
        max_tries=3,
        max_time=30
    )
    def get_fundamentals(self, company_slug: str) -> Optional[Dict[str, Any]]:
        """
        Scrape fundamental data from Screener.in

        Args:
            company_slug: Company name slug (e.g., 'reliance-industries')

        Returns:
            Dictionary with fundamental metrics or None if failed
        """
        try:
            self._rate_limit()

            url = f"{self.base_url}/{company_slug}/"
            logger.info(f"Scraping Screener.in for {company_slug}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            fundamentals = {
                'pe_ratio': None,
                'market_cap': None,
                'book_value': None,
                'dividend_yield': None,
                'roe': None,
                'roce': None,
                'debt_to_equity': None,
                'eps': None,
                'sales_growth_3y': None,
                'profit_growth_3y': None,
                'stock_pe': None,
                'peg_ratio': None,
                'price_to_book': None,
                'current_ratio': None,
                'face_value': None,
                'promoter_holding': None
            }

            # Extract data from the top ratios section
            ratios = soup.find_all('li', class_='flex flex-space-between')
            for ratio in ratios:
                try:
                    name_elem = ratio.find('span', class_='name')
                    value_elem = ratio.find('span', class_='number')

                    if not name_elem or not value_elem:
                        continue

                    name = name_elem.text.strip()
                    value = value_elem.text.strip()

                    # Map Screener fields to our standardized keys
                    field_map = {
                        'Market Cap': 'market_cap',
                        'Stock P/E': 'pe_ratio',
                        'Book Value': 'book_value',
                        'Dividend Yield': 'dividend_yield',
                        'ROCE': 'roce',
                        'ROE': 'roe',
                        'Debt to Equity': 'debt_to_equity',
                        'EPS': 'eps',
                        'Face Value': 'face_value',
                        'Promoter holding': 'promoter_holding',
                        'Current Ratio': 'current_ratio',
                        'PEG Ratio': 'peg_ratio'
                    }

                    if name in field_map:
                        fundamentals[field_map[name]] = self._parse_value(value)

                except Exception as e:
                    logger.debug(f"Error parsing ratio: {e}")
                    continue

            # Extract from compounded sales/profit growth table
            growth_section = soup.find('section', id='quarters')
            if growth_section:
                try:
                    rows = growth_section.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].text.strip()
                            if '3 Years' in label:
                                value = cells[1].text.strip()
                                if 'Sales' in label:
                                    fundamentals['sales_growth_3y'] = self._parse_value(value)
                                elif 'Profit' in label:
                                    fundamentals['profit_growth_3y'] = self._parse_value(value)
                except Exception as e:
                    logger.debug(f"Error parsing growth data: {e}")

            logger.info(f"✅ Successfully scraped fundamentals for {company_slug}")
            return fundamentals

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Company not found on Screener.in: {company_slug}")
            else:
                logger.error(f"HTTP error scraping {company_slug}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping Screener.in for {company_slug}: {e}")
            return None

    def _parse_value(self, value_str: str) -> Optional[float]:
        """Parse numeric values from Screener.in strings"""
        try:
            if not value_str or value_str == '-':
                return None

            # Remove commas and handle percentage
            value_str = value_str.replace(',', '').replace('%', '').strip()

            # Handle Cr (Crores)
            if 'Cr.' in value_str or 'Cr' in value_str:
                value_str = value_str.replace('Cr.', '').replace('Cr', '').strip()
                return float(value_str) * 10000000  # Convert to absolute number

            return float(value_str)

        except (ValueError, AttributeError):
            return None


class RedisCache:
    """Handles Redis caching with TTL support"""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL (defaults to env var REDIS_URL)
        """
        redis_url = redis_url or os.getenv('REDIS_URL')

        if not redis_url:
            logger.warning("⚠️ REDIS_URL not configured. Caching disabled.")
            self.redis_client = None
            self.enabled = False
        else:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.enabled = True
                logger.info("✅ Redis cache initialized successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
                self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None

        try:
            data = self.redis_client.get(key)
            if data:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(data)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Redis GET error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value in cache with TTL"""
        if not self.enabled:
            return

        try:
            self.redis_client.setex(key, ttl_seconds, json.dumps(value))
            logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
        except Exception as e:
            logger.error(f"Redis SET error for {key}: {e}")


class StockDataProvider:
    """
    Unified facade for fetching stock data from Fyers and Screener.in
    Orchestrates caching, symbol mapping, and error handling
    """

    def __init__(
            self,
            fyers_app_id: str,
            fyers_access_token: str,
            symbol_mapper,
            redis_url: Optional[str] = None
    ):
        """
        Initialize the unified stock data provider

        Args:
            fyers_app_id: Fyers API App ID
            fyers_access_token: Fyers API Access Token
            symbol_mapper: SymbolMapper instance
            redis_url: Redis connection URL
        """
        self.fyers = FyersProvider(fyers_app_id, fyers_access_token)
        self.screener = ScreenerProvider(delay_seconds=1.5)
        self.symbol_mapper = symbol_mapper
        self.cache = RedisCache(redis_url)

        # Cache TTLs
        self.ohlcv_ttl = int(os.getenv('OHLCV_CACHE_TTL', '3600'))  # 1 hour
        self.fundamentals_ttl = int(os.getenv('FUNDAMENTALS_CACHE_TTL', '86400'))  # 24 hours

        logger.info("✅ StockDataProvider initialized successfully")

    def get_stock_data(
            self,
            symbol: str,
            fetch_ohlcv: bool = True,
            fetch_fundamentals: bool = True,
            period: str = '6mo'
    ) -> Dict[str, Any]:
        """
        Unified method to fetch all stock data

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS' or 'RELIANCE')
            fetch_ohlcv: Whether to fetch OHLCV data
            fetch_fundamentals: Whether to fetch fundamental data
            period: Time period for OHLCV data

        Returns:
            Dictionary with all requested data
        """
        # Clean symbol
        base_symbol = symbol.replace('.NS', '').replace('.BO', '').upper().strip()

        result = {
            'symbol': base_symbol,
            'company_name': None,
            'ohlcv': None,
            'fundamentals': None,
            'errors': []
        }

        try:
            # Get company name from mapper
            company_info = self.symbol_mapper.get_company_info(base_symbol)
            result['company_name'] = company_info.get('name', base_symbol)
            company_slug = company_info.get('screener_slug')

            # Fetch OHLCV data with caching
            if fetch_ohlcv:
                cache_key = f"ohlcv:{base_symbol}:{period}"
                cached_ohlcv = self.cache.get(cache_key)

                if cached_ohlcv:
                    result['ohlcv'] = cached_ohlcv
                else:
                    # Try NSE first, then BSE
                    ohlcv_df = None
                    for exchange in ['NSE', 'BSE']:
                        ohlcv_df = self.fyers.get_historical_data(
                            base_symbol, exchange, period
                        )
                        if ohlcv_df is not None and not ohlcv_df.empty:
                            result['exchange_used'] = exchange
                            break

                    if ohlcv_df is not None and not ohlcv_df.empty:
                        # Convert DataFrame to list of dicts for JSON serialization
                        ohlcv_list = []
                        for date, row in ohlcv_df.iterrows():
                            ohlcv_list.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'close': float(row['Close']),
                                'volume': int(row['Volume'])
                            })

                        result['ohlcv'] = ohlcv_list
                        self.cache.set(cache_key, ohlcv_list, self.ohlcv_ttl)
                    else:
                        result['errors'].append(f"Failed to fetch OHLCV data for {base_symbol}")

            # Fetch fundamental data with caching
            if fetch_fundamentals and company_slug:
                cache_key = f"fundamentals:{company_slug}"
                cached_fundamentals = self.cache.get(cache_key)

                if cached_fundamentals:
                    result['fundamentals'] = cached_fundamentals
                else:
                    fundamentals = self.screener.get_fundamentals(company_slug)
                    if fundamentals:
                        result['fundamentals'] = fundamentals
                        self.cache.set(cache_key, fundamentals, self.fundamentals_ttl)
                    else:
                        result['errors'].append(f"Failed to fetch fundamentals for {company_slug}")

            return result

        except Exception as e:
            logger.error(f"Error in get_stock_data for {symbol}: {e}")
            result['errors'].append(str(e))
            return result