"""
EODHD API Wrapper for Indian Stock Data
Uses direct REST API calls instead of the problematic Python package
"""
import os
import logging
import requests
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EODHDClient:
    """
    Direct EODHD API client for fetching Indian stock data
    Documentation: https://eodhd.com/financial-apis/
    """
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize EODHD client
        
        Args:
            api_key: EODHD API key. If None, reads from EODHD_API_KEY env var
        """
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        if not self.api_key:
            raise ValueError("EODHD_API_KEY not found in environment variables")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("✅ EODHD Client initialized successfully")
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make authenticated request to EODHD API
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ EODHD API request failed: {e}")
            raise
    
    def get_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        period: str = "d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for Indian stocks
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            exchange: Exchange code ('NSE' or 'BSE')
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
            period: Data period ('d' for daily, 'w' for weekly, 'm' for monthly)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        # Format symbol for EODHD (e.g., RELIANCE.NSE)
        ticker = f"{symbol}.{exchange}"
        
        # Default to last 1 year if no dates provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        params = {
            'from': from_date,
            'to': to_date,
            'period': period
        }
        
        logger.info(f"📊 Fetching data for {ticker} from {from_date} to {to_date}")
        
        try:
            data = self._make_request(f"eod/{ticker}", params)
            
            if not data:
                logger.warning(f"⚠️ No data returned for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjusted_close': 'Adj Close'
            })
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            logger.info(f"✅ Retrieved {len(df)} data points for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_realtime_price(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """
        Get current/latest price for a stock
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code ('NSE' or 'BSE')
            
        Returns:
            Current price as float, or None if unavailable
        """
        ticker = f"{symbol}.{exchange}"
        
        try:
            data = self._make_request(f"real-time/{ticker}", {})
            
            if data and 'close' in data:
                return float(data['close'])
            
            logger.warning(f"⚠️ No realtime price for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching realtime price for {ticker}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get fundamental data for a stock
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            
        Returns:
            Dictionary with fundamental data
        """
        ticker = f"{symbol}.{exchange}"
        
        try:
            data = self._make_request(f"fundamentals/{ticker}", {})
            return data
        except Exception as e:
            logger.error(f"❌ Error fetching fundamentals for {ticker}: {e}")
            return {}
    
    def search_symbol(self, query: str, exchange: str = "NSE") -> List[Dict]:
        """
        Search for stock symbols
        
        Args:
            query: Search query
            exchange: Exchange to search in
            
        Returns:
            List of matching symbols
        """
        try:
            data = self._make_request("search", {
                'q': query,
                'exchange': exchange
            })
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error searching for '{query}': {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize client
        client = EODHDClient()
        
        # Test 1: Get historical data for Reliance
        print("\n" + "="*50)
        print("TEST 1: Historical Data for RELIANCE")
        print("="*50)
        df = client.get_historical_data("RELIANCE", "NSE", period="d")
        if not df.empty:
            print(f"\n✅ Retrieved {len(df)} days of data")
            print(df.head())
            print(df.tail())
        
        # Test 2: Get realtime price
        print("\n" + "="*50)
        print("TEST 2: Realtime Price for TCS")
        print("="*50)
        price = client.get_realtime_price("TCS", "NSE")
        if price:
            print(f"✅ Current TCS price: ₹{price}")
        
        # Test 3: Search for symbol
        print("\n" + "="*50)
        print("TEST 3: Search for 'INFY'")
        print("="*50)
        results = client.search_symbol("INFY", "NSE")
        if results:
            print(f"✅ Found {len(results)} results")
            for r in results[:3]:
                print(f"  - {r.get('Code', 'N/A')}: {r.get('Name', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
