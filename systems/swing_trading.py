import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List
import requests
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset  # <-- Add this
import traceback  # <-- Add this
from textblob import TextBlob
# from eodhd_wrapper import EODHDClient  # <-- No longer needed
# import yfinance as yf  # <-- No longer needed
import config
from hf_utils import query_hf_api

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedSwingTradingSystem:
    def __init__(self, data_provider=None):  # ← ADD THIS PARAMETER
        try:
            self.news_api_key = config.NEWS_API_KEY
            self.swing_trading_params = config.SWING_TRADING_PARAMS
            self._validate_trading_params()
            self.initialize_stock_database()

            # --- API CONFIGURATION CHECKS ---
            self.sentiment_api_url = config.HF_SENTIMENT_API_URL
            self.model_api_available = bool(self.sentiment_api_url)
            self.model_type = "SBERT API" if self.model_api_available else "TextBlob"

            # --- NEW: Store injected data provider ---
            self.data_provider = data_provider
            if data_provider:
                logger.info("✅ Data provider injected into SwingTradingSystem")
            else:
                logger.warning("⚠️ No data provider provided - data fetching will fail")

            # --- REMOVE ALL THIS EODHD CODE ---
            # try:
            #     self.eodhd_client = EODHDClient()
            #     logger.info("✅ EODHD Client initialized successfully")
            # except Exception as e:
            #     logger.warning(f"⚠️ EODHD initialization failed: {e}. Will use yfinance as fallback")
            #     self.eodhd_client = None

            logger.info("✅ EnhancedSwingTradingSystem initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing EnhancedSwingTradingSystem: {e}")
            raise

    def _validate_trading_params(self):
        """Validate trading parameters"""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier']

            for param in required_params:
                if param not in self.swing_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")

                value = self.swing_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")

            # Additional validation
            if self.swing_trading_params['min_holding_period'] >= self.swing_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")

            if self.swing_trading_params['risk_per_trade'] > 0.1:  # 10% max risk per trade
                raise ValueError("risk_per_trade cannot exceed 10%")

            logger.info("Trading parameters validated successfully")

        except Exception as e:
            logger.error(f"Error validating trading parameters: {str(e)}")
            raise

    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database (BSE + NSE) with backtest-based improvements"""
        try:
            self.indian_stocks = {
                # NIFTY 50 Stocks
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas"},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology"},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
                "LT": {"name": "Larsen & Toubro", "sector": "Construction"},
                "SBIN": {"name": "State Bank of India", "sector": "Banking"},
                "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications"},
                "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods"},
                "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile"},
                "TITAN": {"name": "Titan Company", "sector": "Consumer Goods"},
                "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals"},
                "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement"},
                "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods"},
                "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology"},
                "AXISBANK": {"name": "Axis Bank", "sector": "Banking"},
                "WIPRO": {"name": "Wipro", "sector": "Information Technology"},
                "NTPC": {"name": "NTPC", "sector": "Power"},
                "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power"},
                "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas"},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology"},
                "TATASTEEL": {"name": "Tata Steel", "sector": "Steel"},
                "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate"},
                "COALINDIA": {"name": "Coal India", "sector": "Mining"},
                "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals"},
                "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel"},
                "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile"},
                "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile"},
                "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile"},
                "GRASIM": {"name": "Grasim Industries", "sector": "Cement"},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement"},
                "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile"},
                "UPL": {"name": "UPL Limited", "sector": "Chemicals"},
                "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas"},
                "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals"},
                "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals"},
                "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals"},
                "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods"},
                "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods"},
                "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas"},
                "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare"},
                "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services"},
                "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance"},
                "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance"},
                "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking"},
                "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure"},
                "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile"},
                "ITC": {"name": "ITC Limited", "sector": "Consumer Goods"},
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods"},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods"},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
                "BAJAJHLDNG": {"name": "Bajaj Holdings", "sector": "Financial Services"},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods"},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods"},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals"},
                "CADILAHC": {"name": "Cadila Healthcare", "sector": "Pharmaceuticals"},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals"},
                "ALKEM": {"name": "Alkem Laboratories", "sector": "Pharmaceuticals"},
                "TORNTPHARM": {"name": "Torrent Pharmaceuticals", "sector": "Pharmaceuticals"},
                "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharmaceuticals"},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile"},
                "BOSCHLTD": {"name": "Bosch Limited", "sector": "Automobile"},
                "EXIDEIND": {"name": "Exide Industries", "sector": "Automobile"},
                "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile"},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile"},
                "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile"},
                "MRF": {"name": "MRF Limited", "sector": "Automobile"},
                "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile"},
                "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile"},
                "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking"},
                "BANDHANBNK": {"name": "Bandhan Bank", "sector": "Banking"},
                "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Banking"},
                "RBLBANK": {"name": "RBL Bank", "sector": "Banking"},
                "YESBANK": {"name": "Yes Bank", "sector": "Banking"},
                "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
                "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking"},
                "CANBK": {"name": "Canara Bank", "sector": "Banking"},
                "UNIONBANK": {"name": "Union Bank of India", "sector": "Banking"},
                "CHOLAFIN": {"name": "Cholamandalam Investment", "sector": "Financial Services"},
                "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services"},
                "MANAPPURAM": {"name": "Manappuram Finance", "sector": "Financial Services"},
                "MMFIN": {"name": "Mahindra & Mahindra Financial", "sector": "Financial Services"},
                "SRTRANSFIN": {"name": "Shriram Transport Finance", "sector": "Financial Services"},
                "MINDTREE": {"name": "Mindtree Limited", "sector": "Information Technology"},
                "LTTS": {"name": "L&T Technology Services", "sector": "Information Technology"},
                "PERSISTENT": {"name": "Persistent Systems", "sector": "Information Technology"},
                "CYIENT": {"name": "Cyient Limited", "sector": "Information Technology"},
                "NIITTECH": {"name": "NIIT Technologies", "sector": "Information Technology"},
                "ROLTA": {"name": "Rolta India", "sector": "Information Technology"},
                "HEXATECHNO": {"name": "Hexa Technologies", "sector": "Information Technology"},
                "COFORGE": {"name": "Coforge Limited", "sector": "Information Technology"},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
                "TRENT": {"name": "Trent Limited", "sector": "Retail"},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
                "RAYMOND": {"name": "Raymond Limited", "sector": "Textiles"},
                "VBL": {"name": "Varun Beverages", "sector": "Consumer Goods"},
                "EMAMILTD": {"name": "Emami Limited", "sector": "Consumer Goods"},
                "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Goods"},
            }

            # --- STRATEGY IMPROVEMENT BASED ON BACKTEST ---
            # The backtest for the Swing Trading strategy showed consistent losses
            # on these specific low-volatility, blue-chip stocks. They are being
            # excluded from this strategy's universe to improve performance.
            symbols_to_exclude = ['RELIANCE', 'HDFCBANK', 'TCS']

            original_count = len(self.indian_stocks)

            self.indian_stocks = {
                symbol: info
                for symbol, info in self.indian_stocks.items()
                if symbol not in symbols_to_exclude
            }

            logger.info(
                f"Excluded {len(symbols_to_exclude)} underperforming symbols based on backtest. "
                f"Universe size reduced from {original_count} to {len(self.indian_stocks)}."
            )

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

        except Exception as e:
            logger.error(f"Error initializing stock database: {str(e)}")
            # Fallback to minimal database
            self.indian_stocks = {
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis with error handling"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {str(e)}")
            return ["RELIANCE", "TCS", "HDFCBANK"]  # Fallback symbols

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database with error handling"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol:
                raise ValueError("Invalid symbol format")

            return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown"})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {"name": str(symbol), "sector": "Unknown"}

    def load_sbert_model(self, model_path):
        """Load trained SBERT sentiment model with comprehensive error handling"""
        try:
            if not SBERT_AVAILABLE:
                logger.warning("sentence-transformers not available, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not model_path:
                logger.warning("No model path provided, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not os.path.exists(model_path):
                logger.warning(f"SBERT model not found at {model_path}")
                logger.info("Using TextBlob as fallback for sentiment analysis")
                self.model_type = "TextBlob"
                return

            logger.info(f"Loading SBERT sentiment model from {model_path}...")

            # Load with timeout protection
            self.sentiment_pipeline = joblib.load(model_path)

            if not isinstance(self.sentiment_pipeline, dict):
                raise ValueError("Invalid model format - expected dictionary")

            self.vectorizer = self.sentiment_pipeline.get("vectorizer")
            self.model = self.sentiment_pipeline.get("model")
            self.label_encoder = self.sentiment_pipeline.get("label_encoder")

            if all([self.vectorizer, self.model, self.label_encoder]):
                # Validate model components
                if not hasattr(self.vectorizer, 'transform'):
                    raise ValueError("Invalid vectorizer - missing transform method")
                if not hasattr(self.model, 'predict'):
                    raise ValueError("Invalid model - missing predict method")
                if not hasattr(self.label_encoder, 'classes_'):
                    raise ValueError("Invalid label encoder - missing classes_")

                logger.info("SBERT sentiment model loaded successfully!")
                logger.info(f"Model classes: {list(self.label_encoder.classes_)}")
                self.model_loaded = True
                self.model_type = "SBERT + RandomForest"
            else:
                logger.warning("Model components incomplete, using TextBlob fallback")
                self.model_type = "TextBlob"
                self.sentiment_pipeline = None

        except Exception as e:
            logger.error(f"Error loading SBERT model: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("Using TextBlob as fallback for sentiment analysis")
            self.model_type = "TextBlob"
            self.sentiment_pipeline = None

    def get_sector_weights(self, sector):
        """Get dynamic weights based on sector for swing trading with error handling"""
        try:
            if not sector:
                logger.warning("Empty sector provided, using default weights")
                return 0.55, 0.45

            sector = str(sector).lower().strip()

            # Swing trading weights (balanced approach)
            tech_weight, sentiment_weight = 0.55, 0.45

            weights_map = {
                "technology": (0.45, 0.55),  # Tech more sentiment driven
                "information technology": (0.45, 0.55),
                "tech": (0.45, 0.55),
                "it": (0.45, 0.55),
                "financial": (0.60, 0.40),  # Finance more technical
                "financial services": (0.60, 0.40),
                "banking": (0.60, 0.40),
                "finance": (0.60, 0.40),
                "consumer staples": (0.65, 0.35),
                "staples": (0.65, 0.35),
                "consumer goods": (0.65, 0.35),
                "food & staples retailing": (0.65, 0.35),
                "energy": (0.55, 0.45),
                "oil & gas": (0.55, 0.45),
                "utilities": (0.70, 0.30),
                "electric": (0.70, 0.30),
                "power": (0.70, 0.30),
                "healthcare": (0.50, 0.50),
                "pharmaceuticals": (0.50, 0.50),
                "health care": (0.50, 0.50),
                "pharma": (0.50, 0.50),
                "consumer discretionary": (0.45, 0.55),
                "consumer cyclicals": (0.45, 0.55),
                "retail": (0.45, 0.55),
                "automobile": (0.45, 0.55),
                "auto": (0.45, 0.55),
            }

            for key, weights in weights_map.items():
                if key in sector:
                    tech_weight, sentiment_weight = weights
                    break

            # Validate weights
            if tech_weight + sentiment_weight != 1.0:
                logger.warning(f"Invalid weights for sector {sector}, using defaults")
                return 0.55, 0.45

            return tech_weight, sentiment_weight

        except Exception as e:
            logger.error(f"Error getting sector weights for {sector}: {str(e)}")
            return 0.55, 0.45  # Default weights

    def get_indian_stock_data(self, symbol, period="6mo"):

        try:
            if not self.data_provider:
                logger.error("❌ Data provider not available")
                return None, None, None

            symbol = str(symbol).upper().replace(".NS", "").replace(".BO", "")

            logger.info(f"Fetching data for {symbol} via unified data provider")

            # Fetch data using new provider
            stock_data = self.data_provider.get_stock_data(
                symbol=symbol,
                fetch_ohlcv=True,
                fetch_fundamentals=False,  # Not needed for swing trading
                period=period
            )

            # Check for errors
            if stock_data.get('errors'):
                logger.warning(f"Data fetch had errors: {stock_data['errors']}")

            # Extract OHLCV data
            ohlcv_list = stock_data.get('ohlcv')
            if not ohlcv_list:
                logger.error(f"❌ No OHLCV data returned for {symbol}")
                return None, None, None

            # Convert list of dicts back to pandas DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlcv_list)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Capitalize for consistency

            # Create info dict
            info = {
                'shortName': stock_data.get('company_name', symbol),
                'symbol': symbol
            }

            final_symbol = f"{symbol}.{stock_data.get('exchange_used', 'NSE')}"

            logger.info(f"✅ Retrieved {len(df)} days for {symbol} from new data provider")
            return df, info, final_symbol

        except Exception as e:
            logger.error(f"Critical error in get_indian_stock_data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations with error handling"""
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

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands with comprehensive error handling"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < period:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            sma = self.safe_rolling_calculation(prices, period, 'mean')
            std = self.safe_rolling_calculation(prices, period, 'std')

            if sma.empty or std.empty:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return upper_band, sma, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            nan_series = pd.Series([np.nan] * len(prices),
                                   index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return nan_series, nan_series, nan_series

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series

            if len(close) < k_period:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            lowest_low = self.safe_rolling_calculation(low, k_period, 'min')
            highest_high = self.safe_rolling_calculation(high, k_period, 'max')

            if lowest_low.empty or highest_high.empty:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            # Avoid division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)

            k_percent = 100 * ((close - lowest_low) / denominator)
            d_percent = self.safe_rolling_calculation(k_percent, d_period, 'mean')

            return k_percent, d_percent

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            nan_series = pd.Series([np.nan] * len(close),
                                   index=close.index if hasattr(close, 'index') else range(len(close)))
            return nan_series, nan_series

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels with error handling"""
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
                    # Check if current high is a local maximum
                    if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                        resistance_levels.append(data['High'].iloc[i])

                    # Check if current low is a local minimum
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

    def calculate_volume_profile(self, data, bins=20):
        """Calculate Volume Profile with error handling"""
        try:
            if data is None or data.empty or 'Volume' not in data.columns:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                return None, None

            price_range = data['High'].max() - data['Low'].min()
            if price_range <= 0:
                return None, None

            bin_size = price_range / bins
            volume_profile = {}

            for i in range(len(data)):
                try:
                    price = (data['High'].iloc[i] + data['Low'].iloc[i]) / 2
                    volume = data['Volume'].iloc[i]

                    if pd.isna(price) or pd.isna(volume) or volume <= 0:
                        continue

                    bin_level = int((price - data['Low'].min()) / bin_size)
                    bin_level = min(bin_level, bins - 1)
                    bin_level = max(bin_level, 0)

                    price_level = data['Low'].min() + (bin_level * bin_size)

                    if price_level not in volume_profile:
                        volume_profile[price_level] = 0
                    volume_profile[price_level] += volume
                except Exception as e:
                    logger.warning(f"Error processing volume at index {i}: {str(e)}")
                    continue

            if not volume_profile:
                return None, None

            # Find Point of Control (POC) - highest volume level
            poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
            poc_volume = volume_profile[poc_price]

            return volume_profile, poc_price

        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return None, None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with comprehensive error handling"""
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

            # Avoid division by zero
            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)  # Fill NaN with neutral RSI

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD with error handling"""
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

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                return pd.Series(dtype=float)

            if len(close) < period:
                return pd.Series([np.nan] * len(close), index=close.index)

            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self.safe_rolling_calculation(tr, period, 'mean')

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series([np.nan] * len(close), index=close.index if hasattr(close, 'index') else range(len(close)))

    def fetch_indian_news(self, symbol, num_articles=15):
        """Fetch news for Indian companies with error handling"""
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

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration with error handling"""
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            return [
                f"{company_name} reports strong quarterly earnings beating estimates",
                f"Analysts upgrade {company_name} target price citing strong fundamentals",
                f"{company_name} announces major expansion plans and new product launches",
                f"Regulatory approval boosts {company_name} market position",
                f"{company_name} forms strategic partnership with global leader",
                f"Market volatility creates buying opportunity in {company_name}",
                f"{company_name} invests heavily in R&D and digital transformation",
                f"Industry experts bullish on {company_name} long-term prospects",
                f"Competitive pressure intensifies for {company_name} in key markets",
                f"Strong domestic demand drives {company_name} revenue growth",
                f"{company_name} management provides optimistic guidance for next quarter",
                f"Foreign institutional investors increase stake in {company_name}",
                f"Technical breakout signals potential upside for {company_name}",
                f"{company_name} benefits from favorable government policy changes",
                f"Sector rotation favors {company_name} business model"
            ]
        except Exception as e:
            logger.error(f"Error generating sample news for {symbol}: {str(e)}")
            return [f"Market analysis for {symbol}", f"Investment opportunity in {symbol}"]

    def _analyze_sentiment_via_api(self, articles: list) -> tuple[list, list] | None:
        """Analyzes sentiment by calling the remote SBERT Hugging Face API."""
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
            logging.error(f"Could not parse SBERT API response. Error: {e}. Response: {api_results}")
            return None

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob with error handling"""
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

    def analyze_news_sentiment(self, symbol, num_articles=15):
        """Main sentiment analysis function updated to use the API."""
        try:
            articles = self.fetch_indian_news(symbol, num_articles) or self.get_sample_news(symbol)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"
            if not articles:
                return [], [], [], "No Analysis", "No Source"

            if self.model_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    return sentiments, articles, confidences, "SBERT API", news_source

            logging.warning(f"Falling back to TextBlob for news sentiment for {symbol}.")
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob Fallback", news_source
        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error"

    def calculate_swing_trading_score(self, data, sentiment_data, sector):
        """Calculate comprehensive swing trading score with error handling"""
        try:
            tech_weight, sentiment_weight = self.get_sector_weights(sector)

            # Initialize components
            technical_score = 0
            sentiment_score = 50  # Default neutral sentiment

            if data is None or data.empty:
                logger.error("No data provided for scoring")
                return 0

            # ===== TECHNICAL ANALYSIS (Enhanced for Swing Trading) =====
            try:
                current_price = data['Close'].iloc[-1]
                if pd.isna(current_price) or current_price <= 0:
                    logger.error("Invalid current price")
                    return 0
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                return 0

            # RSI Analysis (20 points)
            try:
                rsi = self.calculate_rsi(data['Close'])
                if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                    current_rsi = rsi.iloc[-1]
                    if 30 <= current_rsi <= 70:  # Good for swing trading
                        technical_score += 20
                    elif current_rsi < 30:  # Oversold - potential reversal
                        technical_score += 15
                    elif current_rsi > 70:  # Overbought - potential reversal
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")

            # Bollinger Bands Analysis (15 points)
            try:
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
                if not bb_upper.empty and not any(pd.isna([bb_upper.iloc[-1], bb_lower.iloc[-1]])):
                    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                    if 0.2 <= bb_position <= 0.8:  # Good swing trading zone
                        technical_score += 15
                    elif bb_position < 0.2:  # Near lower band - potential buy
                        technical_score += 12
                    elif bb_position > 0.8:  # Near upper band - potential sell
                        technical_score += 8
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}")

            # Stochastic Analysis (15 points)
            try:
                stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
                if not stoch_k.empty and not any(pd.isna([stoch_k.iloc[-1], stoch_d.iloc[-1]])):
                    k_val = stoch_k.iloc[-1]
                    d_val = stoch_d.iloc[-1]
                    if k_val > d_val and k_val < 80:  # Bullish crossover
                        technical_score += 15
                    elif 20 <= k_val <= 80:  # Good swing range
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {str(e)}")

            # MACD Analysis (15 points)
            try:
                macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
                if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                    if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                        technical_score += 15
                    if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                        if histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")

            # Volume Analysis (10 points)
            try:
                if 'Volume' in data.columns:
                    avg_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    if not pd.isna(avg_volume) and not pd.isna(current_volume) and avg_volume > 0:
                        if current_volume > avg_volume * 1.2:  # Above average volume
                            technical_score += 10
                        elif current_volume > avg_volume:
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating volume: {str(e)}")

            # Support/Resistance Analysis (10 points)
            try:
                support, resistance = self.calculate_support_resistance(data)
                if support and resistance and not any(pd.isna([support, resistance])):
                    distance_to_support = (current_price - support) / support
                    distance_to_resistance = (resistance - current_price) / current_price

                    if distance_to_support < 0.05:  # Near support
                        technical_score += 8
                    elif distance_to_resistance < 0.05:  # Near resistance
                        technical_score += 5
                    elif 0.05 <= distance_to_support <= 0.15:  # Good swing zone
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating support/resistance: {str(e)}")

            # Moving Average Analysis (15 points)
            try:
                if len(data) >= 50:
                    ma_20 = self.safe_rolling_calculation(data['Close'], 20, 'mean').iloc[-1]
                    ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                    if not any(pd.isna([ma_20, ma_50])):
                        if current_price > ma_20 > ma_50:  # Strong uptrend
                            technical_score += 15
                        elif current_price > ma_20:  # Above short-term MA
                            technical_score += 10
                        elif ma_20 > ma_50:  # MA alignment positive
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating moving averages: {str(e)}")

            # Normalize technical score to 0-100
            technical_score = min(100, max(0, technical_score))

            # ===== SENTIMENT ANALYSIS =====
            try:
                if sentiment_data and len(sentiment_data) >= 3:
                    sentiments, _, confidences, _, _ = sentiment_data
                    if sentiments and confidences:
                        sentiment_value = 0
                        total_weight = 0

                        for sentiment, confidence in zip(sentiments, confidences):
                            weight = confidence if not pd.isna(confidence) else 0.5
                            if sentiment == 'positive':
                                sentiment_value += weight
                            elif sentiment == 'negative':
                                sentiment_value -= weight
                            total_weight += weight

                        if total_weight > 0:
                            normalized_sentiment = sentiment_value / total_weight
                            sentiment_score = 50 + (normalized_sentiment * 50)
                        else:
                            sentiment_score = 50
                    else:
                        sentiment_score = 50
                else:
                    sentiment_score = 50
            except Exception as e:
                logger.warning(f"Error calculating sentiment score: {str(e)}")
                sentiment_score = 50

            # ===== COMBINE SCORES =====
            sentiment_score = min(100, max(0, sentiment_score))
            final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
            final_score = min(100, max(0, final_score))

            return final_score

        except Exception as e:
            logger.error(f"Error calculating swing trading score: {str(e)}")
            return 0

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics with comprehensive error handling"""
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

            # Sharpe Ratio (assuming 6% risk-free rate)
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
                atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
                current_atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else data['Close'].iloc[
                                                                                                   -1] * 0.02
            except Exception:
                current_atr = data['Close'].iloc[-1] * 0.02 if not data['Close'].empty else 0

            # Risk level determination
            try:
                if volatility > 0.4:
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
                'atr': current_atr,
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return default_metrics

    def generate_trading_plan(self, data, score, risk_metrics):
        """Generate complete trading plan with realistic targets for swing trading (1-4 weeks)."""
        default_plan = {
            'entry_signal': "HOLD/WATCH",
            'entry_strategy': "Wait for clearer signals",
            'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days",
            'trade_management_note': 'N/A'
        }

        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02

            trade_management_note = "After hitting Target 1, move Stop Loss to breakeven. Trail stops as price advances to lock in profits."

            # Entry Strategy based on score
            if score >= 75:
                entry_signal = "STRONG BUY"
            elif score >= 60:
                entry_signal = "BUY"
            elif score >= 45:
                entry_signal = "HOLD/WATCH"
            elif score >= 30:
                entry_signal = "SELL"
            else:
                entry_signal = "STRONG SELL"

            entry_strategy_map = {
                "STRONG BUY": "Enter on any pullback or at market. Strong momentum and favorable conditions.",
                "BUY": "Enter on minor dips or breakouts above resistance. Good risk-reward setup.",
                "HOLD/WATCH": "Wait for clearer signals. Current setup lacks strong conviction.",
                "SELL": "Exit long positions. Consider taking profits or avoiding new entries.",
                "STRONG SELL": "Exit all positions immediately. Unfavorable conditions detected."
            }
            entry_strategy = entry_strategy_map.get(entry_signal, "Wait for clearer signals")

            # REALISTIC PRICE TARGETS for Swing Trading
            stop_loss_distance = atr * 1.5
            stop_loss = max(current_price - stop_loss_distance, 0)

            # Simple, conservative targets based on ATR
            target_1 = current_price + (stop_loss_distance * 1.0)  # 1:1 RR
            target_2 = current_price + (stop_loss_distance * 1.5)  # 1:1.5 RR
            target_3 = current_price + (stop_loss_distance * 2.0)  # 1:2 RR

            # IMPORTANT: Ensure targets are always ABOVE current price
            if target_1 <= current_price or target_2 <= current_price or target_3 <= current_price:
                logger.warning(f"Invalid targets detected for {data.name}, recalculating...")
                # Fallback: use percentage-based targets
                target_1 = current_price * 1.03  # 3% gain
                target_2 = current_price * 1.05  # 5% gain
                target_3 = current_price * 1.08  # 8% gain

            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'stop_loss': stop_loss,
                'targets': {
                    'target_1': target_1,
                    'target_2': target_2,
                    'target_3': target_3
                },
                'support': None,  # Remove support/resistance from here - causes issues
                'resistance': None,
                'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days",
                'trade_management_note': trade_management_note
            }

        except Exception as e:
            logger.error(f"Error generating trading plan: {str(e)}")
            return default_plan

    def analyze_swing_trading_stock(self, symbol, period="6mo"):
        """Comprehensive swing trading analysis for a single stock with full error handling"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting analysis for {symbol}")

            # Get stock data
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Extract information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)

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

            # Technical indicators
            rsi = self.calculate_rsi(data['Close'])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            support, resistance = self.calculate_support_resistance(data)
            volume_profile, poc_price = self.calculate_volume_profile(data)

            # Sentiment analysis
            sentiment_results = self.analyze_news_sentiment(final_symbol)

            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(data)

            # Swing trading score
            swing_score = self.calculate_swing_trading_score(data, sentiment_results, sector)

            # Trading plan
            trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)

            # Safe value extraction
            rsi_val = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
            bb_upper_val = bb_upper.iloc[-1] if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else None
            bb_middle_val = bb_middle.iloc[-1] if not bb_middle.empty and not pd.isna(bb_middle.iloc[-1]) else None
            bb_lower_val = bb_lower.iloc[-1] if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else None
            bb_position = ((current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)) if all(
                x is not None for x in [bb_upper_val, bb_lower_val]) and bb_upper_val != bb_lower_val else None
            stoch_k_val = stoch_k.iloc[-1] if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else None
            stoch_d_val = stoch_d.iloc[-1] if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else None
            macd_line_val = macd_line.iloc[-1] if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else None
            signal_line_val = signal_line.iloc[-1] if not signal_line.empty and not pd.isna(
                signal_line.iloc[-1]) else None
            histogram_val = histogram.iloc[-1] if not histogram.empty and not pd.isna(histogram.iloc[-1]) else None

            # Compile results
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'rsi': rsi_val,
                'bollinger_bands': {
                    'upper': bb_upper_val, 'middle': bb_middle_val, 'lower': bb_lower_val, 'position': bb_position
                },
                'stochastic': {'k': stoch_k_val, 'd': stoch_d_val},
                'macd': {'line': macd_line_val, 'signal': signal_line_val, 'histogram': histogram_val},
                'support_resistance': {
                    'support': support, 'resistance': resistance,
                    'distance_to_support': ((current_price - support) / support * 100) if support else None,
                    'distance_to_resistance': (
                            (resistance - current_price) / current_price * 100) if resistance else None
                },
                'volume_profile': {
                    'poc_price': poc_price,
                    'current_vs_poc': ((current_price - poc_price) / poc_price * 100) if poc_price else None
                },
                'sentiment': {
                    'scores': sentiment_results[0], 'articles': sentiment_results[1],
                    'confidence': sentiment_results[2], 'method': sentiment_results[3],
                    'source': sentiment_results[4],
                    'sentiment_summary': {
                        'positive': sentiment_results[0].count('positive'),
                        'negative': sentiment_results[0].count('negative'),
                        'neutral': sentiment_results[0].count('neutral')
                    }
                },
                'risk_metrics': risk_metrics,
                'swing_score': swing_score,
                'trading_plan': trading_plan,
                'model_type': self.model_type,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"Successfully analyzed {symbol} with score {swing_score}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def analyze_multiple_stocks(self, symbols, period="6mo"):
        """Analyze multiple stocks with progress tracking and comprehensive error handling"""
        results = []
        total_stocks = len(symbols) if symbols else 0

        if total_stocks == 0:
            logger.error("No symbols provided for analysis")
            return results

        print(f"Analyzing {total_stocks} stocks...")
        logger.info(f"Starting analysis of {total_stocks} stocks for swing trading.")

        successful_analyses = 0
        failed_analyses = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                # Progress printout
                progress_pct = (i / total_stocks) * 100
                print(f"\rAnalyzing: [{i}/{total_stocks}] {symbol}... ({progress_pct:.0f}%)", end="")

                if not symbol or not isinstance(symbol, str):
                    logger.warning(f"Invalid symbol at position {i}: {symbol}")
                    failed_analyses += 1
                    continue

                analysis = self.analyze_swing_trading_stock(symbol.strip(), period)
                if analysis and analysis.get('swing_score', 0) > 0:
                    results.append(analysis)
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    logger.warning(f"Analysis failed or returned zero score for {symbol}")

            except KeyboardInterrupt:
                logger.info("Analysis interrupted by user")
                print(f"\nAnalysis interrupted. Processed {i - 1}/{total_stocks} stocks.")
                break
            except Exception as e:
                logger.error(f"Unexpected error analyzing {symbol}: {str(e)}")
                failed_analyses += 1
                continue

        print("\nAnalysis complete.")
        # Sort by swing trading score
        try:
            results.sort(key=lambda x: x.get('swing_score', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting results: {str(e)}")

        logger.info(f"Analysis completed: {successful_analyses} successful, {failed_analyses} failed")

        return results

    def filter_stocks_by_risk_appetite(self, results, risk_appetite):
        """Filter stocks based on user's risk appetite with error handling"""
        try:
            if not results:
                logger.warning("No results to filter")
                return []

            if not risk_appetite:
                logger.warning("No risk appetite specified, using MEDIUM")
                risk_appetite = "MEDIUM"

            risk_thresholds = {
                'LOW': 0.25,  # <=25% volatility
                'MEDIUM': 0.40,  # <=40% volatility
                'HIGH': 1.0  # <=100% volatility (all stocks)
            }

            max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)

            filtered_stocks = []
            for stock in results:
                try:
                    if not isinstance(stock, dict):
                        continue

                    risk_metrics = stock.get('risk_metrics', {})
                    trading_plan = stock.get('trading_plan', {})

                    volatility = risk_metrics.get('volatility', 1.0)  # Default high volatility
                    entry_signal = trading_plan.get('entry_signal', 'HOLD/WATCH')

                    if (volatility <= max_volatility and
                            entry_signal in ['BUY', 'STRONG BUY']):
                        filtered_stocks.append(stock)

                except Exception as e:
                    logger.warning(f"Error filtering stock {stock.get('symbol', 'Unknown')}: {str(e)}")
                    continue

            logger.info(
                f"Filtered {len(filtered_stocks)} stocks from {len(results)} based on {risk_appetite} risk appetite")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering stocks by risk appetite: {str(e)}")
            return []

    def generate_portfolio_allocation(self, results, total_capital, risk_appetite):
        """
        Generate diversified portfolio allocation with structured JSON return.
        Returns list of dicts with numeric values for API consumption.
        """
        try:
            if not results or not isinstance(results, list):
                logger.error("No suitable stocks found for portfolio creation")
                return None

            if total_capital <= 0:
                logger.error("Invalid total capital amount")
                return None

            # Diversification parameters
            max_positions = min(10, len(results))
            min_positions = min(5, len(results))

            # Risk parameters
            risk_per_trade_pct = self.swing_trading_params['risk_per_trade']
            max_position_size_pct = 0.20

            # Calculate ideal position size
            if len(results) >= min_positions:
                ideal_positions = min(max_positions, len(results))
                base_allocation_per_stock = total_capital / ideal_positions
            else:
                base_allocation_per_stock = total_capital / len(results)

            # Console output (keep for logging)
            print(f"\nPORTFOLIO ALLOCATION (Total Capital: Rs.{total_capital:,.2f})")
            print(
                f"Diversification Target: {min(ideal_positions if len(results) >= min_positions else len(results), max_positions)} positions")
            print(
                f"Risk Model: Max {risk_per_trade_pct:.0%} risk per trade, Max {max_position_size_pct:.0%} per position")
            print("=" * 110)
            print(
                f"{'Rank':<4} {'Symbol':<12} {'Company':<25} {'Score':<6} {'Price':<10} {'Stop Loss':<10} {'Allocation':<12} {'Shares':<8} {'Risk':<12}")
            print("-" * 110)

            portfolio_data = []
            total_allocated = 0
            position_count = 0
            target_positions = min(ideal_positions if len(results) >= min_positions else len(results), max_positions)

            for i, result in enumerate(results, 1):
                try:
                    if position_count >= target_positions:
                        break

                    current_price = result.get('current_price', 0)
                    trading_plan = result.get('trading_plan', {})
                    stop_loss = trading_plan.get('stop_loss', 0)
                    swing_score = result.get('swing_score', 0)

                    if current_price <= 0 or stop_loss <= 0 or current_price <= stop_loss:
                        continue

                    risk_per_share = current_price - stop_loss

                    # Position sizing calculations (your existing logic)
                    capital_at_risk = total_capital * risk_per_trade_pct
                    risk_based_shares = int(capital_at_risk / risk_per_share)

                    equal_weight_amount = base_allocation_per_stock
                    equal_weight_shares = int(equal_weight_amount / current_price)

                    score_weight = swing_score / 100.0
                    score_weighted_amount = base_allocation_per_stock * (0.7 + 0.3 * score_weight)
                    score_weighted_shares = int(score_weighted_amount / current_price)

                    max_position_amount = total_capital * max_position_size_pct
                    max_position_shares = int(max_position_amount / current_price)

                    # Choose allocation method
                    if equal_weight_shares > 0 and equal_weight_amount <= max_position_amount:
                        final_shares = min(equal_weight_shares, risk_based_shares, max_position_shares)
                    elif score_weighted_shares > 0 and score_weighted_amount <= max_position_amount:
                        final_shares = min(score_weighted_shares, risk_based_shares, max_position_shares)
                    else:
                        final_shares = min(risk_based_shares, max_position_shares)

                    if final_shares <= 0:
                        continue

                    final_amount = final_shares * current_price

                    # Check remaining capital
                    if total_allocated + final_amount > total_capital:
                        remaining_capital = total_capital - total_allocated
                        if remaining_capital >= current_price:
                            final_shares = int(remaining_capital / current_price)
                            final_amount = final_shares * current_price
                        else:
                            continue

                    actual_risk = final_shares * risk_per_share
                    allocation_pct = (final_amount / total_capital) * 100

                    total_allocated += final_amount
                    position_count += 1

                    company_name = result.get('company_name', result.get('symbol', 'Unknown'))

                    # CRITICAL: Build structured data with exact numeric types
                    portfolio_data.append({
                        'rank': i,
                        'symbol': result.get('symbol', 'Unknown'),
                        'company': company_name,
                        'score': float(swing_score),
                        'price': float(current_price),  # NUMERIC
                        'stop_loss': float(stop_loss),  # NUMERIC
                        'allocation_pct': float(allocation_pct),  # NUMERIC (8.1, not "8.1%")
                        'investment_amount': float(final_amount),
                        'number_of_shares': int(final_shares),
                        'risk': float(actual_risk),
                        'sector': result.get('sector', 'Unknown')
                    })

                    # Console output (keep for logging)
                    company_short = company_name[:23] + "..." if len(company_name) > 25 else company_name
                    print(f"{i:<4} {result.get('symbol', 'Unk'):<12} {company_short:<25} "
                          f"{swing_score:<6.0f} ₹{current_price:<9.2f} ₹{stop_loss:<9.2f} "
                          f"{allocation_pct:<11.1f}% {final_shares:<8} ₹{actual_risk:<11,.0f}")

                except Exception as e:
                    logger.error(f"Error processing stock {i}: {str(e)}")
                    continue

            if not portfolio_data:
                logger.error("Could not allocate any positions")
                return None

            # Console summary (keep for logging)
            avg_score = sum(r['score'] for r in portfolio_data) / len(portfolio_data)
            total_risk = sum(r['risk'] for r in portfolio_data)

            print(f"\nPORTFOLIO SUMMARY")
            print("-" * 50)
            print(f"Total Allocated: Rs.{total_allocated:,.2f} ({total_allocated / total_capital * 100:.1f}%)")
            print(f"Positions: {len(portfolio_data)}")
            print(f"Avg Score: {avg_score:.1f}/100")
            print(f"Total Risk: Rs.{total_risk:,.2f}")

            # Return ONLY the structured list (no text, no extra dict)
            return portfolio_data

        except Exception as e:
            logger.error(f"Error generating portfolio: {str(e)}")
            return None

    def get_single_best_recommendation(self, results):
        """Get detailed recommendation for the single best stock with enhanced formatting and error handling."""
        try:
            if not results or not isinstance(results, list):
                logger.warning("No results available for recommendation")
                return None

            best_stock = results[0]
            if not isinstance(best_stock, dict):
                logger.error("Invalid best stock data format")
                return None

            print(f"\n{Fore.YELLOW}⭐ SINGLE BEST STOCK RECOMMENDATION ⭐{Style.RESET_ALL}")
            print("=" * 70)

            # --- Safely extract all data using .get() to prevent errors ---
            company_name = best_stock.get('company_name', 'Unknown')
            symbol = best_stock.get('symbol', 'Unknown')
            sector = best_stock.get('sector', 'Unknown')
            swing_score = best_stock.get('swing_score', 0)
            current_price = best_stock.get('current_price', 0)
            price_change = best_stock.get('price_change', 0)
            price_change_pct = best_stock.get('price_change_pct', 0)
            risk_metrics = best_stock.get('risk_metrics', {})
            risk_level = risk_metrics.get('risk_level', 'Unknown')

            # --- Enhanced Display ---
            price_color = Fore.GREEN if price_change >= 0 else Fore.RED
            score_color = Fore.GREEN if swing_score >= 75 else Fore.YELLOW if swing_score >= 60 else Fore.RED

            print(f"{'Company:':<15} {company_name} ({symbol})")
            print(f"{'Sector:':<15} {sector}")
            print(
                f"{'Price:':<15} {price_color}₹{current_price:,.2f} ({price_change:+.2f}, {price_change_pct:+.2f}%){Style.RESET_ALL}")
            print(f"{'Swing Score:':<15} {score_color}{swing_score:.0f}/100{Style.RESET_ALL}")
            print(f"{'Risk Level:':<15} {risk_level}")

            # --- Trading Recommendation with more color ---
            trading_plan = best_stock.get('trading_plan', {})
            signal = trading_plan.get('entry_signal', 'N/A')
            signal_color = Fore.GREEN if "BUY" in signal else (Fore.RED if "SELL" in signal else Fore.YELLOW)

            print(f"\n{Fore.YELLOW}ACTIONABLE TRADING PLAN{Style.RESET_ALL}")
            print("-" * 30)
            print(f"{'Signal:':<15} {signal_color}{signal}{Style.RESET_ALL}")
            print(f"{'Strategy:':<15} {trading_plan.get('entry_strategy', 'N/A')}")

            targets = trading_plan.get('targets', {})
            print(f"{'Stop Loss:':<15} {Fore.RED}₹{trading_plan.get('stop_loss', 0):.2f}{Style.RESET_ALL}")
            print(f"{'Target 1:':<15} {Fore.GREEN}₹{targets.get('target_1', 0):.2f}{Style.RESET_ALL}")
            print(f"{'Target 2:':<15} {Fore.GREEN}₹{targets.get('target_2', 0):.2f}{Style.RESET_ALL}")

            # Display the trade management advice
            if trading_plan.get('trade_management_note'):
                print(f"{'Pro Tip:':<15} {Fore.CYAN}{trading_plan.get('trade_management_note')}{Style.RESET_ALL}")

            # Key technical levels
            print(f"\n{Fore.YELLOW}KEY LEVELS & DATA{Style.RESET_ALL}")
            print("-" * 30)
            print(f"{'Support:':<15} ₹{trading_plan.get('support', 0):.2f}")
            print(f"{'Resistance:':<15} ₹{trading_plan.get('resistance', 0):.2f}")

            rsi_val = best_stock.get('rsi')
            if rsi_val is not None:
                print(f"{'RSI (14-day):':<15} {rsi_val:.1f}")

            # Sentiment summary
            sentiment = best_stock.get('sentiment', {}).get('sentiment_summary', {})
            print(
                f"{'News Sentiment:':<15} Pos: {sentiment.get('positive', 0)}, Neg: {sentiment.get('negative', 0)}, Neu: {sentiment.get('neutral', 0)}")

            return best_stock

        except Exception as e:
            logger.error(f"Error getting single best recommendation: {str(e)}")
            return None

    def print_analysis_summary(self, all_results, filtered_results, risk_appetite, total_budget):
        """Print comprehensive analysis summary with error handling"""
        try:
            print(f"\nMARKET ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Total Stocks Analyzed: {len(all_results) if all_results else 0}")
            print(f"Risk Appetite: {risk_appetite}")
            print(f"Budget: Rs.{total_budget:,}")
            print(f"Suitable Stocks Found: {len(filtered_results) if filtered_results else 0}")

            if all_results and len(all_results) > 0:
                try:
                    avg_market_score = sum(r.get('swing_score', 0) for r in all_results) / len(all_results)
                    print(f"Average Market Score: {avg_market_score:.1f}/100")
                except:
                    print("Average Market Score: Unable to calculate")

                # Risk distribution
                risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'UNKNOWN': 0}
                for result in all_results:
                    try:
                        risk_level = result.get('risk_metrics', {}).get('risk_level', 'UNKNOWN')
                        if risk_level in risk_distribution:
                            risk_distribution[risk_level] += 1
                        else:
                            risk_distribution['UNKNOWN'] += 1
                    except:
                        risk_distribution['UNKNOWN'] += 1

                print(f"\nMARKET RISK DISTRIBUTION")
                print("-" * 25)
                for risk, count in risk_distribution.items():
                    if count > 0:
                        percentage = (count / len(all_results)) * 100
                        print(f"{risk} Risk: {count} stocks ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Error printing analysis summary: {str(e)}")
            print(f"Error generating analysis summary: {str(e)}")





















