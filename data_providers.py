import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date
import requests
from bs4 import BeautifulSoup
import backoff
import redis
import json
import pandas as pd
import traceback

# Fyers API V3
from fyers_apiv3 import fyersModel

logger = logging.getLogger(__name__)


class FyersProvider:
    """Handles all Fyers API V3 interactions for OHLCV data with retry logic and pagination"""

    def __init__(self, app_id: str, access_token: str):
        self.app_id = app_id
        self.access_token = access_token
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
            resolution: str = 'D'  # UPDATED: Added resolution parameter
    ) -> Optional[pd.DataFrame]:
        """
        Public-facing method to fetch historical OHLCV data.
        This function calculates the date range and calls the paginated fetcher.
        """
        try:
            # 1. Calculate date range
            to_date = datetime.now().date()
            period_map = {
                '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365,
                '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650
            }
            # For intraday, Fyers has limits on days, but for daily it's fine
            days = period_map.get(period, 180)
            from_date = to_date - timedelta(days=days)

            # 2. Convert symbol to Fyers format
            fyers_symbol = f"{exchange}:{symbol}-EQ"

            # 3. Call the paginated fetcher
            df = self.fetch_paginated_history(
                symbol=fyers_symbol,
                resolution=resolution,  # UPDATED: Pass resolution
                start_date=from_date,
                end_date=to_date
            )

            if df is None or df.empty:
                logger.warning(f"No data returned from Fyers for {fyers_symbol}")
                return None

            logger.info(f"✅ Retrieved {len(df)} rows from Fyers for {fyers_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {e}")
            return None

    def fetch_paginated_history(
            self,
            symbol: str,
            resolution: str,
            start_date: date,
            end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical data by paginating requests.
        Adjusts chunk size based on resolution (Fyers has limits).
        """
        all_candles_data = []
        current_start_dt = start_date

        # Fyers limits intraday requests to ~100 days, daily to much more
        days_per_chunk = 100 if resolution not in ['D', '1D'] else 365

        while current_start_dt < end_date:
            # 1. Calculate the end of this chunk
            chunk_end_dt = current_start_dt + timedelta(days=days_per_chunk)

            # 2. Cap the chunk's end date at the final end_date
            if chunk_end_dt > end_date:
                chunk_end_dt = end_date

            # 3. Format dates for the API request
            range_from = current_start_dt.strftime('%Y-%m-%d')
            range_to = chunk_end_dt.strftime('%Y-%m-%d')

            logger.info(f"Fyers Fetch: Requesting {symbol} ({resolution}) from {range_from} to {range_to}")

            # 4. Create the Fyers API payload
            payload = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",  # 1 = YYYY-MM-DD
                "range_from": range_from,
                "range_to": range_to,
                "cont_flag": "1"
            }

            try:
                response = self.fyers.history(data=payload)

                if response.get('s') == 'ok' and response.get('candles'):
                    all_candles_data.extend(response['candles'])
                elif response.get('s') == 'no_data':
                    logger.warning(f"Fyers Fetch: No data for {symbol} in range {range_from} to {range_to}.")
                else:
                    error_msg = response.get('message', 'Unknown error')
                    logger.error(f"Fyers API error for {symbol}: {error_msg}")
                    if 'Invalid' in error_msg:
                        logger.critical(f"Stopping pagination due to invalid request: {payload}")
                        break

            except Exception as e:
                logger.error(f"Fyers Fetch: Exception during API call for {symbol}: {e}")
                break  # Stop trying if a critical exception occurs

            # 5. Set the start of the *next* loop
            current_start_dt = chunk_end_dt + timedelta(days=1)

            # 6. IMPORTANT: Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)  # 500ms delay

        if not all_candles_data:
            return None

        # 7. Convert the final list of lists into a DataFrame
        try:
            df = pd.DataFrame(all_candles_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Use 'datetime' for intraday, 'date' for daily
            if resolution in ['D', '1D']:
                 df['Date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
            else:
                 df['Date'] = pd.to_datetime(df['timestamp'], unit='s')

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.drop(columns=['timestamp'])

            # Rename columns to match your StockDataProvider's expectations
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            df = df.set_index('Date')

            # Remove any duplicate dates (can happen at the seams) and sort
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()

            return df

        except Exception as e:
            logger.error(f"Fyers Fetch: Failed to create DataFrame for {symbol}. Error: {e}")
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
        Scrape fundamental data from Screener.in's CONSOLIDATED page.
        """
        try:
            self._rate_limit()

            url = f"{self.base_url}/{company_slug}/consolidated/"
            logger.info(f"Scraping Screener.in for {company_slug} at {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            fundamentals = {
                'pe_ratio': None, 'market_cap': None, 'book_value': None,
                'dividend_yield': None, 'roe': None, 'roce': None,
                'debt_to_equity': None, 'eps': None, 'sales_growth_3y': None,
                'profit_growth_3y': None, 'stock_pe': None, 'peg_ratio': None,
                'price_to_book': None, 'current_ratio': None, 'face_value': None,
                'promoter_holding': None, 'industry_pe': None  # UPDATED: Added Industry P/E
            }

            field_map = {
                'Market Cap': 'market_cap', 'Stock P/E': 'pe_ratio',
                'Book Value': 'book_value', 'Dividend Yield': 'dividend_yield',
                'ROCE': 'roce', 'ROE': 'roe', 'Debt to Equity': 'debt_to_equity',
                'EPS': 'eps', 'Face Value': 'face_value',
                'Promoter holding': 'promoter_holding', 'Current Ratio': 'current_ratio',
                'PEG Ratio': 'peg_ratio',
                'Industry P/E': 'industry_pe'  # UPDATED: Added Industry P/E
            }

            top_ratios_div = soup.find('div', id='top-ratios')

            if not top_ratios_div:
                logger.warning(f"Could not find 'div#top-ratios' on {url}. Scraping may be inaccurate.")
                ratios_list = soup.find_all('li', class_='flex flex-space-between')
            else:
                ratios_list = top_ratios_div.find_all('li', class_='flex flex-space-between')

            for ratio in ratios_list:
                try:
                    name_elem = ratio.find('span', class_='name')
                    value_elem = ratio.find('span', class_='number')

                    if not name_elem or not value_elem:
                        continue

                    name = name_elem.text.strip()
                    value = value_elem.text.strip()

                    if name in field_map:
                        fundamentals[field_map[name]] = self._parse_value(value)

                except Exception as e:
                    logger.debug(f"Error parsing ratio: {e}")
                    continue

            # --- UPDATED: Corrected section ID from 'quarters' to 'growth' ---
            growth_section = soup.find('section', id='growth')
            if growth_section:
                try:
                    # Find the table for "Compounded Sales Growth"
                    sales_header = growth_section.find('h3', string=lambda t: t and 'Compounded Sales Growth' in t)
                    if sales_header:
                        sales_table = sales_header.find_next_sibling('table')
                        for row in sales_table.find_all('tr'):
                            cells = row.find_all('td')
                            if len(cells) >= 2 and '3 Years' in cells[0].text:
                                fundamentals['sales_growth_3y'] = self._parse_value(cells[1].text)
                                break
                    
                    # Find the table for "Compounded Profit Growth"
                    profit_header = growth_section.find('h3', string=lambda t: t and 'Compounded Profit Growth' in t)
                    if profit_header:
                        profit_table = profit_header.find_next_sibling('table')
                        for row in profit_table.find_all('tr'):
                            cells = row.find_all('td')
                            if len(cells) >= 2 and '3 Years' in cells[0].text:
                                fundamentals['profit_growth_3y'] = self._parse_value(cells[1].text)
                                break
                                
                except Exception as e:
                    logger.debug(f"Error parsing growth data: {e}")

            logger.info(f"✅ Successfully scraped fundamentals for {company_slug}")
            return fundamentals

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Company not found on Screener.in (or 404 on consolidated page): {company_slug}")
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

            value_str = value_str.replace(',', '').replace('%', '').strip()

            if 'Cr.' in value_str or 'Cr' in value_str:
                value_str = value_str.replace('Cr.', '').replace('Cr', '').strip()
                return float(value_str) * 10000000

            return float(value_str)

        except (ValueError, AttributeError):
            return None


class RedisCache:
    """Handles Redis caching with TTL support"""

    def __init__(self, redis_url: Optional[str] = None):
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
        self.fyers = FyersProvider(fyers_app_id, fyers_access_token)
        self.screener = ScreenerProvider(delay_seconds=1.5)
        self.symbol_mapper = symbol_mapper
        self.cache = RedisCache(redis_url)

        self.ohlcv_ttl = int(os.getenv('OHLCV_CACHE_TTL', '3600'))  # 1 hour
        self.fundamentals_ttl = int(os.getenv('FUNDAMENTALS_CACHE_TTL', '86400'))  # 24 hours

        logger.info("✅ StockDataProvider initialized successfully")

    def get_stock_data(
            self,
            symbol: str,
            fetch_ohlcv: bool = True,
            fetch_fundamentals: bool = True,
            period: str = '6mo',
            resolution: str = 'D'  # UPDATED: Added resolution
    ) -> Dict[str, Any]:
        """
        Unified method to fetch all stock data
        """
        base_symbol = symbol.replace('.NS', '').replace('.BO', '').upper().strip()

        result = {
            'symbol': base_symbol,
            'company_name': None,
            'ohlcv': None,
            'fundamentals': None,
            'errors': []
        }

        try:
            company_info = self.symbol_mapper.get_company_info(base_symbol)
            result['company_name'] = company_info.get('name', base_symbol)
            company_slug = company_info.get('screener_slug')

            if fetch_ohlcv:
                # UPDATED: Cache key now includes resolution
                cache_key = f"ohlcv:{base_symbol}:{period}:{resolution}"
                cached_ohlcv = self.cache.get(cache_key)

                if cached_ohlcv:
                    logger.info(f"OHLCV Cache HIT for {base_symbol} ({resolution})")
                    result['ohlcv'] = cached_ohlcv
                else:
                    logger.info(f"OHLCV Cache MISS for {base_symbol} ({resolution})")
                    ohlcv_df = None
                    for exchange in ['NSE', 'BSE']:
                        ohlcv_df = self.fyers.get_historical_data(
                            base_symbol, exchange, period, resolution  # UPDATED: Pass resolution
                        )
                        if ohlcv_df is not None and not ohlcv_df.empty:
                            result['exchange_used'] = exchange
                            break

                    if ohlcv_df is not None and not ohlcv_df.empty:
                        
                        # --- UPDATED: Replaced iterrows() with vectorized conversion ---
                        logger.debug("Optimizing OHLCV DataFrame to list conversion...")
                        ohlcv_df_processed = ohlcv_df.reset_index()
                        
                        # Format date based on resolution
                        if resolution in ['D', '1D']:
                             ohlcv_df_processed['date'] = ohlcv_df_processed['Date'].dt.strftime('%Y-%m-%d')
                        else:
                             ohlcv_df_processed['date'] = ohlcv_df_processed['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                        # Ensure correct types and lowercase keys for JSON
                        ohlcv_df_processed = ohlcv_df_processed.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                        })

                        output_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                        
                        # Use list comprehension on to_dict('records') to ensure standard python types
                        ohlcv_records = ohlcv_df_processed[output_columns].to_dict('records')
                        
                        ohlcv_list = [
                            {
                                'date': r['date'],
                                'open': float(r['open']),
                                'high': float(r['high']),
                                'low': float(r['low']),
                                'close': float(r['close']),
                                'volume': int(r['volume'])
                            } for r in ohlcv_records
                        ]
                        # --- End of optimization ---

                        result['ohlcv'] = ohlcv_list
                        self.cache.set(cache_key, ohlcv_list, self.ohlcv_ttl)
                    else:
                        result['errors'].append(f"Failed to fetch OHLCV data for {base_symbol}")

            if fetch_fundamentals:
                if not company_slug:
                    logger.warning(f"No screener_slug for {base_symbol}, cannot fetch fundamentals.")
                    result['errors'].append(f"No screener_slug mapped for {base_symbol}")
                else:
                    cache_key = f"fundamentals:{company_slug}"
                    cached_fundamentals = self.cache.get(cache_key)

                    if cached_fundamentals:
                        logger.info(f"Fundamentals Cache HIT for {company_slug}")
                        result['fundamentals'] = cached_fundamentals
                    else:
                        logger.info(f"Fundamentals Cache MISS for {company_slug}")
                        fundamentals = self.screener.get_fundamentals(company_slug)

                        if fundamentals:
                            result['fundamentals'] = fundamentals
                            self.cache.set(cache_key, fundamentals, self.fundamentals_ttl)
                        else:
                            result['errors'].append(f"Failed to fetch fundamentals for {company_slug}")

            return result

        except Exception as e:
            logger.error(f"Critical error in get_stock_data for {symbol}: {e}")
            logger.error(traceback.format_exc())
            result['errors'].append(f"Internal server error: {e}")
            return result
