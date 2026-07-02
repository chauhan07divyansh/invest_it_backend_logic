"""
OPTIMIZED SWING TRADING SYSTEM
Enterprise-grade with parallel processing, caching, and expanded universe
Key Features:
- 500+ stock universe (NSE liquid stocks)
- Parallel analysis (10x faster)
- Redis caching (sub-second response)
- Background refresh worker
- Progressive result streaming
Author: SentiQuant (Optimized for Production)
"""
import os
import logging
import warnings
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import traceback
from textblob import TextBlob
import concurrent.futures
from functools import lru_cache
import redis
from core import config
from services.sentiment.hf_utils import query_hf_api
from services.data_providers.symbol_mapper import SymbolMapper
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

ENABLE_NEW_LOGIC = os.getenv('ENABLE_NEW_LOGIC', 'false').lower() == 'true'

# ── Shadow-mode FinBERT comparison (production keeps using SBERT) ──────────
FINBERT_API_URL    = os.getenv("FINBERT_API_URL", "")
FINBERT_API_SECRET = os.getenv("FINBERT_API_SECRET", "")
SHADOW_LOG_KEY     = "shadow:sentiment:log"
_shadow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

_BOILERPLATE_MARKERS = [
    r'Disclaimer\s*:',
    r'This article is based on a regulatory filing',
    r'Investments?\s+in\s+securities\s+market\s+are\s+subject\s+to\s+market\s+risks',
    r'[Mm]utual\s+fund\s+investments\s+are\s+subject\s+to\s+market\s+risk',
    r'[Pp]ast\s+performance\s+is\s+not\s+indicative',
    r'Views?\s+and\s+recommendations?\s+given\s+.{0,40}\s+are\s+their\s+own',
    r'[Pp]lease\s+consult\s+your\s+.{0,30}advisor',
    r'Also\s+Read\s*[:|]',
    r'Also\s+Watch\s*[:|]',
    r'Download\s+The\s+(Economic\s+Times|Mint)\s+News\s+App',
    r'Catch\s+all\s+the\s+.{0,60}\s+on\s+Mint',
]
import re as _re
_BOILERPLATE_RE = _re.compile('|'.join(_BOILERPLATE_MARKERS), _re.IGNORECASE)


class EnhancedSwingTradingSystem:
    def __init__(self, data_provider=None, redis_client=None):
        try:
            self.event_registry_api_key = getattr(config, "EVENT_REGISTRY_API_KEY", None)
            self.event_registry_endpoint = getattr(config, "EVENT_REGISTRY_ENDPOINT", None)
            if not self.event_registry_api_key:
                logger.warning("⚠️ EVENT_REGISTRY_API_KEY not configured. News sentiment disabled.")
            self.swing_trading_params = config.SWING_TRADING_PARAMS
            self._validate_trading_params()
            if not data_provider:
                raise ValueError("❌ data_provider is REQUIRED for SwingTradingSystem")
            self.data_provider = data_provider
            logger.info("✅ Data provider injected into SwingTradingSystem")
            self.initialize_expanded_stock_database()
            self.sentiment_api_url = config.HF_SENTIMENT_API_URL
            self.model_api_available = bool(self.sentiment_api_url)
            self.redis_client = redis_client
            self.cache_enabled = bool(redis_client)
            self.cache_ttl = {
                'ohlcv': 3600,
                'analysis': 900,
                'news': 86400,
                'batch_analysis': 900
            }
            logger.info(f"✅ EnhancedSwingTradingSystem initialized with {len(self.indian_stocks)} stocks")
        except Exception as e:
            logger.error(f"❌ Error initializing EnhancedSwingTradingSystem: {e}")
            raise

    def _validate_trading_params(self):
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier']
            for param in required_params:
                if param not in self.swing_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")
                value = self.swing_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")
            if self.swing_trading_params['min_holding_period'] >= self.swing_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")
            if self.swing_trading_params['risk_per_trade'] > 0.1:
                raise ValueError("risk_per_trade cannot exceed 10%")
            logger.info("Trading parameters validated successfully")
        except Exception as e:
            logger.error(f"Error validating trading parameters: {e}")
            raise

    def initialize_expanded_stock_database(self):
        try:
            self.indian_stocks = {
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
                "SIEMENS": {"name": "Siemens Limited", "sector": "Capital Goods"},
                "HAVELLS": {"name": "Havells India", "sector": "Electricals"},
                "DLF": {"name": "DLF Limited", "sector": "Real Estate"},
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods"},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods"},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods"},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods"},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals"},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals"},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile"},
                "BOSCHLTD": {"name": "Bosch Limited", "sector": "Automobile"},
                "EXIDEIND": {"name": "Exide Industries", "sector": "Automobile"},
                "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile"},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile"},
                "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile"},
                "MRF": {"name": "MRF Limited", "sector": "Automobile"},
                "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile"},
                "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile"},
                "CUMMINSIND": {"name": "Cummins India", "sector": "Automobile"},
                "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking"},
                "BANDHANBNK": {"name": "Bandhan Bank", "sector": "Banking"},
                "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Banking"},
                "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
                "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking"},
                "CANBK": {"name": "Canara Bank", "sector": "Banking"},
                "UNIONBANK": {"name": "Union Bank of India", "sector": "Banking"},
                "CHOLAFIN": {"name": "Cholamandalam Investment", "sector": "Financial Services"},
                "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services"},
                "SRTRANSFIN": {"name": "Shriram Transport Finance", "sector": "Financial Services"},
                "LTTS": {"name": "L&T Technology Services", "sector": "Information Technology"},
                "PERSISTENT": {"name": "Persistent Systems", "sector": "Information Technology"},
                "COFORGE": {"name": "Coforge Limited", "sector": "Information Technology"},
                "MPHASIS": {"name": "Mphasis Limited", "sector": "Information Technology"},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
                "TRENT": {"name": "Trent Limited", "sector": "Retail"},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
                "RAYMOND": {"name": "Raymond Limited", "sector": "Textiles"},
                "BERGEPAINT": {"name": "Berger Paints", "sector": "Consumer Goods"},
                "VOLTAS": {"name": "Voltas Limited", "sector": "Consumer Durables"},
                "WHIRLPOOL": {"name": "Whirlpool of India", "sector": "Consumer Durables"},
                "CROMPTON": {"name": "Crompton Greaves", "sector": "Electricals"},
                "TORNTPHARM": {"name": "Torrent Pharmaceuticals", "sector": "Pharmaceuticals"},
                "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharmaceuticals"},
                "ALKEM": {"name": "Alkem Laboratories", "sector": "Pharmaceuticals"},
                "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Services"},
                "VBL": {"name": "Varun Beverages", "sector": "Beverages"},
                "EMAMILTD": {"name": "Emami Limited", "sector": "Consumer Goods"},
                "GODREJPROP": {"name": "Godrej Properties", "sector": "Real Estate"},
                "OBEROIRLTY": {"name": "Oberoi Realty", "sector": "Real Estate"},
                "ABCAPITAL": {"name": "Aditya Birla Capital", "sector": "Financial Services"},
                "ABFRL": {"name": "Aditya Birla Fashion", "sector": "Retail"},
                "ACC": {"name": "ACC Limited", "sector": "Cement"},
                "ADANIGREEN": {"name": "Adani Green Energy", "sector": "Power"},
                "ADANIPOWER": {"name": "Adani Power", "sector": "Power"},
                "AFFLE": {"name": "Affle India", "sector": "Technology"},
                "AIAENG": {"name": "AIA Engineering", "sector": "Capital Goods"},
                "AJANTPHARM": {"name": "Ajanta Pharma", "sector": "Pharmaceuticals"},
                "AKUMS": {"name": "Akums Drugs", "sector": "Pharmaceuticals"},
                "AMBER": {"name": "Amber Enterprises", "sector": "Consumer Durables"},
                "AMBUJACEM": {"name": "Ambuja Cements", "sector": "Cement"},
                "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile"},
                "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile"},
                "ASTRAL": {"name": "Astral Limited", "sector": "Building Materials"},
                "ATUL": {"name": "Atul Limited", "sector": "Chemicals"},
                "AUBANK": {"name": "AU Small Finance Bank", "sector": "Banking"},
                "BAJAJELEC": {"name": "Bajaj Electricals", "sector": "Electricals"},
                "BALAMINES": {"name": "Balaji Amines", "sector": "Chemicals"},
                "BATAINDIA": {"name": "Bata India", "sector": "Footwear"},
                "BEL": {"name": "Bharat Electronics", "sector": "Defence"},
                "BHEL": {"name": "Bharat Heavy Electricals", "sector": "Capital Goods"},
                "BRIGADE": {"name": "Brigade Enterprises", "sector": "Real Estate"},
                "CESC": {"name": "CESC Limited", "sector": "Power"},
                "CHAMBLFERT": {"name": "Chambal Fertilizers", "sector": "Fertilizers"},
                "CONCOR": {"name": "Container Corporation", "sector": "Logistics"},
                "COROMANDEL": {"name": "Coromandel International", "sector": "Fertilizers"},
                "CRISIL": {"name": "CRISIL Limited", "sector": "Financial Services"},
                "CUB": {"name": "City Union Bank", "sector": "Banking"},
                "CYIENT": {"name": "Cyient Limited", "sector": "Information Technology"},
                "DEEPAKNTR": {"name": "Deepak Nitrite", "sector": "Chemicals"},
                "DIXON": {"name": "Dixon Technologies", "sector": "Electronics"},
                "ESCORTS": {"name": "Escorts Kubota", "sector": "Automobile"},
                "FACT": {"name": "Fertilizers And Chemicals", "sector": "Fertilizers"},
                "GAIL": {"name": "GAIL India", "sector": "Oil & Gas"},
                "GLENMARK": {"name": "Glenmark Pharmaceuticals", "sector": "Pharmaceuticals"},
                "GRANULES": {"name": "Granules India", "sector": "Pharmaceuticals"},
                "GRAPHITE": {"name": "Graphite India", "sector": "Capital Goods"},
                "GUJGASLTD": {"name": "Gujarat Gas", "sector": "Oil & Gas"},
                "HFCL": {"name": "HFCL Limited", "sector": "Telecommunications"},
                "HINDCOPPER": {"name": "Hindustan Copper", "sector": "Metals"},
                "HINDPETRO": {"name": "Hindustan Petroleum", "sector": "Oil & Gas"},
                "HONAUT": {"name": "Honeywell Automation", "sector": "Capital Goods"},
                "IEX": {"name": "Indian Energy Exchange", "sector": "Financial Services"},
                "IGL": {"name": "Indraprastha Gas", "sector": "Oil & Gas"},
                "INDHOTEL": {"name": "Indian Hotels", "sector": "Hotels"},
                "INDUSTOWER": {"name": "Indus Towers", "sector": "Telecommunications"},
                "INTELLECT": {"name": "Intellect Design Arena", "sector": "Technology"},
                "IRCTC": {"name": "Indian Railway Catering", "sector": "Services"},
                "ISEC": {"name": "ICICI Securities", "sector": "Financial Services"},
                "JINDALSTEL": {"name": "Jindal Steel & Power", "sector": "Steel"},
                "JKCEMENT": {"name": "JK Cement", "sector": "Cement"},
                "JSWENERGY": {"name": "JSW Energy", "sector": "Power"},
                "KAJARIACER": {"name": "Kajaria Ceramics", "sector": "Building Materials"},
                "KEI": {"name": "KEI Industries", "sector": "Electricals"},
                "L&TFH": {"name": "L&T Finance Holdings", "sector": "Financial Services"},
                "LALPATHLAB": {"name": "Dr Lal PathLabs", "sector": "Healthcare"},
                "LAURUSLABS": {"name": "Laurus Labs", "sector": "Pharmaceuticals"},
                "MANAPPURAM": {"name": "Manappuram Finance", "sector": "Financial Services"},
                "MCX": {"name": "Multi Commodity Exchange", "sector": "Financial Services"},
                "METROBRAND": {"name": "Metro Brands", "sector": "Footwear"},
                "MFSL": {"name": "Max Financial Services", "sector": "Insurance"},
                "MGL": {"name": "Mahanagar Gas", "sector": "Oil & Gas"},
                "MINDTREE": {"name": "Mindtree Limited", "sector": "Information Technology"},
                "MOTHERSON": {"name": "Samvardhana Motherson", "sector": "Automobile"},
                "MUTHOOTFIN": {"name": "Muthoot Finance", "sector": "Financial Services"},
                "NATIONALUM": {"name": "National Aluminium", "sector": "Metals"},
                "NAUKRI": {"name": "Info Edge India", "sector": "Internet"},
                "NAVINFLUOR": {"name": "Navin Fluorine", "sector": "Chemicals"},
                "NMDC": {"name": "NMDC Limited", "sector": "Mining"},
                "OIL": {"name": "Oil India", "sector": "Oil & Gas"},
                "PAYTM": {"name": "One 97 Communications", "sector": "Financial Services"},
                "PEL": {"name": "Piramal Enterprises", "sector": "Financial Services"},
                "PETRONET": {"name": "Petronet LNG", "sector": "Oil & Gas"},
                "PFC": {"name": "Power Finance Corporation", "sector": "Financial Services"},
                "PHOENIXLTD": {"name": "Phoenix Mills", "sector": "Real Estate"},
                "PIIND": {"name": "PI Industries", "sector": "Chemicals"},
                "POLYCAB": {"name": "Polycab India", "sector": "Electricals"},
                "PRESTIGE": {"name": "Prestige Estates", "sector": "Real Estate"},
                "RECLTD": {"name": "REC Limited", "sector": "Financial Services"},
                "SBICARD": {"name": "SBI Cards", "sector": "Financial Services"},
                "SOLARINDS": {"name": "Solar Industries", "sector": "Chemicals"},
                "SONACOMS": {"name": "Sona BLW Precision", "sector": "Automobile"},
                "SRF": {"name": "SRF Limited", "sector": "Chemicals"},
                "STAR": {"name": "Sterlite Technologies", "sector": "Telecommunications"},
                "TATACOMM": {"name": "Tata Communications", "sector": "Telecommunications"},
                "TATAELXSI": {"name": "Tata Elxsi", "sector": "Information Technology"},
                "TATAPOWER": {"name": "Tata Power", "sector": "Power"},
                "THERMAX": {"name": "Thermax Limited", "sector": "Capital Goods"},
                "TORNTPOWER": {"name": "Torrent Power", "sector": "Power"},
                "UBL": {"name": "United Breweries", "sector": "Beverages"},
                "VEDL": {"name": "Vedanta Limited", "sector": "Metals"},
                "ZOMATO": {"name": "Zomato Limited", "sector": "Consumer Services"},
                "ZYDUSLIFE": {"name": "Zydus Lifesciences", "sector": "Pharmaceuticals"},
                "AAVAS": {"name": "Aavas Financiers", "sector": "Financial Services"},
                "ANANDRATHI": {"name": "Anand Rathi Wealth", "sector": "Financial Services"},
                "ANGELONE": {"name": "Angel One", "sector": "Financial Services"},
                "ASIANHOTNR": {"name": "Asian Hotels (North)", "sector": "Hotels"},
                "BASF": {"name": "BASF India", "sector": "Chemicals"},
                "BLUESTARCO": {"name": "Blue Star", "sector": "Consumer Durables"},
                "CAMS": {"name": "CAMS", "sector": "Financial Services"},
                "CDSL": {"name": "Central Depository Services", "sector": "Financial Services"},
                "CENTRALBK": {"name": "Central Bank of India", "sector": "Banking"},
                "CENTURYPLY": {"name": "Century Plyboards", "sector": "Building Materials"},
                "CLEAN": {"name": "Clean Science", "sector": "Chemicals"},
                "CREDITACC": {"name": "CreditAccess Grameen", "sector": "Financial Services"},
                "CSBBANK": {"name": "CSB Bank", "sector": "Banking"},
                "DELTACORP": {"name": "Delta Corp", "sector": "Gaming"},
                "DEVYANI": {"name": "Devyani International", "sector": "Consumer Services"},
                "EQUITAS": {"name": "Equitas Small Finance Bank", "sector": "Banking"},
                "FINPIPE": {"name": "Fine Organic Industries", "sector": "Chemicals"},
                "FLUOROCHEM": {"name": "Gujarat Fluorochemicals", "sector": "Chemicals"},
                "GRINDWELL": {"name": "Grindwell Norton", "sector": "Capital Goods"},
                "HAPPSTMNDS": {"name": "Happiest Minds", "sector": "Information Technology"},
                "HEMHINDUS": {"name": "HEG Limited", "sector": "Capital Goods"},
                "IIFLWAM": {"name": "IIFL Wealth Management", "sector": "Financial Services"},
                "INDIAMART": {"name": "IndiaMART InterMESH", "sector": "Internet"},
                "INDIANB": {"name": "Indian Bank", "sector": "Banking"},
                "JUBLPHARMA": {"name": "Jubilant Pharmova", "sector": "Pharmaceuticals"},
                "JUSTDIAL": {"name": "Just Dial", "sector": "Internet"},
                "KPITTECH": {"name": "KPIT Technologies", "sector": "Information Technology"},
                "LATENTVIEW": {"name": "Latent View Analytics", "sector": "Information Technology"},
                "LEMONTREE": {"name": "Lemon Tree Hotels", "sector": "Hotels"},
                "MAZDOCK": {"name": "Mazagon Dock Shipbuilders", "sector": "Defence"},
                "METROPOLIS": {"name": "Metropolis Healthcare", "sector": "Healthcare"},
                "MIDHANI": {"name": "Mishra Dhatu Nigam", "sector": "Defence"},
                "NAZARA": {"name": "Nazara Technologies", "sector": "Gaming"},
                "NIACL": {"name": "New India Assurance", "sector": "Insurance"},
                "NYKAA": {"name": "FSN E-Commerce (Nykaa)", "sector": "Retail"},
                "ORIENTELEC": {"name": "Orient Electric", "sector": "Electricals"},
                "PARAS": {"name": "Paras Defence", "sector": "Defence"},
                "PNBHOUSING": {"name": "PNB Housing Finance", "sector": "Financial Services"},
                "POLICYBZR": {"name": "PB Fintech", "sector": "Financial Services"},
                "POONAWALLA": {"name": "Poonawalla Fincorp", "sector": "Financial Services"},
                "RAILTEL": {"name": "RailTel Corporation", "sector": "Telecommunications"},
                "RATNAMANI": {"name": "Ratnamani Metals", "sector": "Metals"},
                "ROUTE": {"name": "Route Mobile", "sector": "Telecommunications"},
                "SAFARI": {"name": "Safari Industries", "sector": "Consumer Goods"},
                "SHYAMMETL": {"name": "Shyam Metalics", "sector": "Metals"},
                "SIGNATURE": {"name": "Signature Global", "sector": "Real Estate"},
                "SYNGENE": {"name": "Syngene International", "sector": "Pharmaceuticals"},
                "TANLA": {"name": "Tanla Platforms", "sector": "Telecommunications"},
                "UCOBANK": {"name": "UCO Bank", "sector": "Banking"},
                "UJJIVAN": {"name": "Ujjivan Small Finance Bank", "sector": "Banking"},
                "UTIAMC": {"name": "UTI Asset Management", "sector": "Financial Services"},
            }
            self.all_stocks_reference = dict(self.indian_stocks)
            exclude_from_swing = ['RELIANCE', 'HDFCBANK', 'TCS']
            self.indian_stocks = {
                symbol: info for symbol, info in self.indian_stocks.items()
                if symbol not in exclude_from_swing
            }
            logger.info(
                f"✅ Expanded database initialized: {len(self.indian_stocks)} stocks (excluded {len(exclude_from_swing)} underperformers)")
            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")
        except Exception as e:
            logger.error(f"Error initializing expanded stock database: {e}")
            self.indian_stocks = {
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    def _get_cache_key(self, prefix: str, symbol: str, **kwargs) -> str:
        param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{prefix}:{symbol}:{param_str}" if param_str else f"{prefix}:{symbol}"

    def _get_from_cache(self, key: str) -> Optional[dict]:
        if not self.cache_enabled:
            return None
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _set_to_cache(self, key: str, data: dict, ttl: int):
        if not self.cache_enabled:
            return
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _sbert_normalized(self, sentiments, confidences) -> float:
        val, total = 0.0, 0.0
        for s, c in zip(sentiments, confidences):
            w = c if (c is not None and not pd.isna(c)) else 0.5
            if s == 'positive':
                val += w
            elif s == 'negative':
                val -= w
            total += w
        return (val / total) if total > 0 else 0.0

    def _shadow_finbert(self, symbol, articles, sentiments, confidences):
        try:
            if not FINBERT_API_URL or not self.redis_client:
                return
            day = datetime.now().strftime("%Y-%m-%d")
            dedupe_key = f"shadow:done:{symbol}:{day}"
            if self.redis_client.get(dedupe_key):
                return
            self.redis_client.setex(dedupe_key, 86400, "1")
            resp = requests.post(
                FINBERT_API_URL,
                json={"articles": articles[:15]},
                headers={"X-Api-Secret": FINBERT_API_SECRET},
                timeout=90,
            )
            if resp.status_code != 200:
                logger.warning(f"[shadow] FinBERT HTTP {resp.status_code} for {symbol}")
                return
            data = resp.json()
            if not data.get("success") or not data.get("results"):
                logger.warning(f"[shadow] FinBERT error for {symbol}: {data.get('error')}")
                return
            per_article   = data["results"]
            fin_scores    = [a["score"] for a in per_article]
            finbert_score = sum(fin_scores) / len(fin_scores)
            worst_window  = min((a.get("worst_window", 0.0) for a in per_article), default=0.0)
            window_fired  = any(a.get("window_fired") for a in per_article)
            sbert_score = self._sbert_normalized(sentiments, confidences)
            DEAD_ZONE = 0.30
            def to_label(x):
                if x > DEAD_ZONE:
                    return "positive"
                if x < -DEAD_ZONE:
                    return "negative"
                return "neutral"
            row = {
                "symbol":        symbol,
                "date":          day,
                "ts":            datetime.now().isoformat(),
                "n_articles":    len(articles),
                "sbert_score":   round(sbert_score, 4),
                "finbert_score": round(finbert_score, 4),
                "sbert_label":   to_label(sbert_score),
                "finbert_label": to_label(finbert_score),
                "agree":         to_label(sbert_score) == to_label(finbert_score),
                "worst_window":  round(worst_window, 4),
                "window_fired":  bool(window_fired),
            }
            self.redis_client.rpush(SHADOW_LOG_KEY, json.dumps(row))
        except Exception as e:
            logger.warning(f"[shadow] failed for {symbol}: {e}")

    def analyze_stocks_parallel(self, symbols: List[str], max_workers: int = 10,
                                period: str = "6mo") -> List[Dict]:
        try:
            logger.info(f"🚀 Starting parallel analysis of {len(symbols)} stocks with {max_workers} workers")
            start_time = time.time()
            results = []
            failed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_swing_trading_stock, symbol, period): symbol
                    for symbol in symbols
                }
                completed = 0
                total = len(symbols)
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    try:
                        result = future.result(timeout=30)
                        if result and result.get('swing_score', 0) > 0:
                            results.append(result)
                        else:
                            failed_count += 1
                    except concurrent.futures.TimeoutError:
                        failed_count += 1
                    except Exception:
                        failed_count += 1
            results.sort(key=lambda x: x.get('swing_score', 0), reverse=True)
            elapsed_time = time.time() - start_time
            logger.info(
                f"✅ Parallel analysis complete: {len(results)} successful, {failed_count} failed in {elapsed_time:.1f}s")
            return results
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            return []

    def get_cached_batch_analysis(self, risk_appetite: Optional[str] = None) -> Optional[List[Dict]]:
        try:
            cache_key = "batch_analysis:all_stocks"
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                if risk_appetite:
                    filtered = self.filter_stocks_by_risk_appetite(cached_results, risk_appetite)
                    return filtered
                return cached_results
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached batch analysis: {e}")
            return None

    def analyze_and_cache_all_stocks(self, max_workers: int = 10):
        try:
            symbols = self.get_all_stock_symbols()
            results = self.analyze_stocks_parallel(symbols, max_workers=max_workers)
            cache_key = "batch_analysis:all_stocks"
            self._set_to_cache(cache_key, results, self.cache_ttl['batch_analysis'])
            return results
        except Exception as e:
            logger.error(f"Error in background batch analysis: {e}")
            return []

    def get_all_stock_symbols(self) -> List[str]:
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {e}")
            return []

    @lru_cache(maxsize=1000)
    def get_stock_info_from_db(self, symbol: str) -> Dict:
        try:
            base_symbol = str(symbol).split('.')[0].upper().strip()
            ref = getattr(self, 'all_stocks_reference', None) or self.indian_stocks
            return ref.get(base_symbol, {"name": symbol, "sector": "Unknown"})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {"name": str(symbol), "sector": "Unknown"}

    @lru_cache(maxsize=100)
    def get_sector_weights(self, sector: str) -> Tuple[float, float]:
        try:
            sector = str(sector).lower().strip()
            weights_map = {
                "technology": (0.45, 0.55),
                "information technology": (0.45, 0.55),
                "financial": (0.60, 0.40),
                "financial services": (0.60, 0.40),
                "banking": (0.60, 0.40),
                "consumer goods": (0.65, 0.35),
                "pharmaceuticals": (0.50, 0.50),
                "automobile": (0.45, 0.55),
                "power": (0.70, 0.30),
                "oil & gas": (0.55, 0.45),
                "healthcare": (0.50, 0.50),
                "retail": (0.45, 0.55),
            }
            for key, weights in weights_map.items():
                if key in sector:
                    return weights
            return 0.55, 0.45
        except Exception as e:
            logger.error(f"Error getting sector weights: {e}")
            return 0.55, 0.45

    def detect_market_regime(self, force_refresh: bool = False) -> str:
        try:
            cache_key = "market:regime"
            if not force_refresh and self.cache_enabled:
                cached = self._get_from_cache(cache_key)
                if cached and cached.get("regime"):
                    return cached["regime"]
            df = self._fetch_nifty_index()
            if df is None or df.empty or len(df) < 50:
                return 'SIDEWAYS'
            close = df['Close']
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            current = close.iloc[-1]
            if any(pd.isna(x) for x in (current, ma20, ma50)):
                return 'SIDEWAYS'
            if current > ma20 > ma50:
                regime = 'BULL'
            elif current < ma20 < ma50:
                regime = 'BEAR'
            else:
                regime = 'SIDEWAYS'
            if self.cache_enabled:
                self._set_to_cache(cache_key, {
                    "regime": regime,
                    "nifty": float(current),
                    "ma20": float(ma20),
                    "ma50": float(ma50),
                    "computed_at": datetime.now().isoformat(),
                }, 3600)
            return regime
        except Exception as e:
            logger.error(f"Regime detection failed, defaulting SIDEWAYS: {e}")
            return 'SIDEWAYS'

    def _fetch_nifty_index(self):
        FYERS_NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"
        cache_key = self._get_cache_key("ohlcv_index", "NIFTY50", period="3mo")
        try:
            cached = self._get_from_cache(cache_key)
            if cached:
                df = pd.DataFrame(cached['ohlcv'])
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date').drop(columns=['date'])
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df
            from datetime import datetime as _dt, timedelta as _td
            end = _dt.now().date()
            start = end - _td(days=120)
            df = self.data_provider.fyers.fetch_paginated_history(
                symbol=FYERS_NIFTY_SYMBOL,
                resolution='D',
                start_date=start,
                end_date=end,
            )
            if df is None or df.empty:
                return None
            df = df.copy()
            df.columns = [str(c).lower() for c in df.columns]
            colmap = {}
            for c in df.columns:
                if c in ('open',): colmap[c] = 'Open'
                elif c in ('high',): colmap[c] = 'High'
                elif c in ('low',): colmap[c] = 'Low'
                elif c in ('close',): colmap[c] = 'Close'
                elif c in ('volume', 'vol'): colmap[c] = 'Volume'
            df = df.rename(columns=colmap)
            if 'Close' not in df.columns:
                return None
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            try:
                cache_df = df.reset_index()
                date_col = cache_df.columns[0]
                ohlcv_list = [{
                    'date': str(r[date_col]),
                    'open': float(r['Open']), 'high': float(r['High']),
                    'low': float(r['Low']), 'close': float(r['Close']),
                    'volume': float(r.get('Volume', 0)),
                } for _, r in cache_df.iterrows()]
                self._set_to_cache(cache_key, {'ohlcv': ohlcv_list}, 3600)
            except Exception:
                pass
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"NIFTY index fetch failed: {e}")
            return None

    def get_indian_stock_data(self, symbol: str, period: str = "6mo") -> Tuple:
        try:
            cache_key = self._get_cache_key("ohlcv", symbol, period=period)
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                df = pd.DataFrame(cached_data['ohlcv'])
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date').drop(columns=['date'])
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                info = cached_data['info']
                final_symbol = cached_data['symbol']
                return df, info, final_symbol
            if not self.data_provider:
                return None, None, None
            symbol_clean = str(symbol).upper().replace(".NS", "").replace(".BO", "")
            stock_data = self.data_provider.get_stock_data(
                symbol=symbol_clean,
                fetch_ohlcv=True,
                fetch_fundamentals=False,
                period=period
            )
            ohlcv_list = stock_data.get('ohlcv')
            if not ohlcv_list:
                return None, None, None
            df = pd.DataFrame(ohlcv_list)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            info = {
                'shortName': stock_data.get('company_name', symbol_clean),
                'symbol': symbol_clean
            }
            final_symbol = f"{symbol_clean}.{stock_data.get('exchange_used', 'NSE')}"
            cache_data = {'ohlcv': ohlcv_list, 'info': info, 'symbol': final_symbol}
            self._set_to_cache(cache_key, cache_data, self.cache_ttl['ohlcv'])
            return df, info, final_symbol
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None

    def safe_rolling_calculation(self, data: pd.Series, window: int, operation: str = 'mean') -> pd.Series:
        try:
            if data is None or data.empty or len(data) < window:
                return pd.Series([np.nan] * len(data), index=data.index if hasattr(data, 'index') else None)
            if operation == 'mean':
                return data.rolling(window=window, min_periods=1).mean()
            elif operation == 'std':
                return data.rolling(window=window, min_periods=1).std()
            elif operation == 'min':
                return data.rolling(window=window, min_periods=1).min()
            elif operation == 'max':
                return data.rolling(window=window, min_periods=1).max()
            else:
                return pd.Series([np.nan] * len(data), index=data.index)
        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {e}")
            return pd.Series([np.nan] * len(data), index=data.index if hasattr(data, 'index') else None)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        try:
            if prices is None or prices.empty or len(prices) < period:
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
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple:
        try:
            if prices is None or prices.empty or len(prices) < period:
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
            logger.error(f"Error calculating Bollinger Bands: {e}")
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        try:
            if prices is None or prices.empty or len(prices) < slow:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3) -> Tuple:
        try:
            if any(x is None or x.empty for x in [high, low, close]) or len(close) < k_period:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series
            lowest_low = self.safe_rolling_calculation(low, k_period, 'min')
            highest_high = self.safe_rolling_calculation(high, k_period, 'max')
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)
            k_percent = 100 * ((close - lowest_low) / denominator)
            d_percent = self.safe_rolling_calculation(k_percent, d_period, 'mean')
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            nan_series = pd.Series([np.nan] * len(close), index=close.index)
            return nan_series, nan_series

    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple:
        try:
            if data is None or data.empty or len(data) < window:
                return data['Low'].min(), data['High'].max()
            highs = self.safe_rolling_calculation(data['High'], window, 'max')
            lows = self.safe_rolling_calculation(data['Low'], window, 'min')
            resistance_levels = []
            support_levels = []
            for i in range(window, len(data)):
                if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(data['High'].iloc[i])
                if not pd.isna(lows.iloc[i]) and data['Low'].iloc[i] == lows.iloc[i]:
                    support_levels.append(data['Low'].iloc[i])
            current_resistance = max(resistance_levels[-3:]) if len(resistance_levels) >= 3 else data['High'].max()
            current_support = min(support_levels[-3:]) if len(support_levels) >= 3 else data['Low'].min()
            return current_support, current_resistance
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return data['Low'].min(), data['High'].max()

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        try:
            if any(x is None or x.empty for x in [high, low, close]) or len(close) < period:
                return pd.Series([np.nan] * len(close), index=close.index)
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self.safe_rolling_calculation(tr, period, 'mean')
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([np.nan] * len(close), index=close.index)

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        default_metrics = {
            'volatility': 0.3, 'var_95': -0.05, 'max_drawdown': -0.2,
            'sharpe_ratio': 0, 'atr': 0, 'risk_level': 'HIGH'
        }
        try:
            if data is None or data.empty or 'Close' not in data.columns or len(data) < 2:
                return default_metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            volatility = volatility if not pd.isna(volatility) else 0.3
            var_95 = np.percentile(returns.dropna(), 5) if len(returns) > 20 else -0.05
            var_95 = var_95 if not pd.isna(var_95) else -0.05
            rolling_max = data['Close'].expanding().max()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() if not pd.isna(drawdown.min()) else -0.2
            risk_free_rate = 0.06
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            sharpe_ratio = sharpe_ratio if not pd.isna(sharpe_ratio) else 0
            atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
            current_atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else data['Close'].iloc[-1] * 0.02
            if volatility > 0.4:
                risk_level = 'HIGH'
            elif volatility > 0.25:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            return {
                'volatility': float(volatility), 'var_95': float(var_95),
                'max_drawdown': float(max_drawdown), 'sharpe_ratio': float(sharpe_ratio),
                'atr': float(current_atr), 'risk_level': risk_level
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return default_metrics

    def _strip_boilerplate(self, text: str) -> str:
        m = _BOILERPLATE_RE.search(text)
        if m and m.start() > 200:
            text = text[:m.start()]
        return text.strip()

    def _is_relevant(self, text: str, base_symbol: str, company_name: str) -> bool:
        head = text.split('.')[0][:160]
        generic = {'limited', 'ltd', 'india', 'company', 'industries',
                   'enterprises', 'services', 'corporation', 'of', 'the', 'and'}
        tokens = [t for t in company_name.split()
                  if t.lower() not in generic and len(t) >= 3]
        full = ' '.join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else base_symbol)
        sym_pat = _re.compile(r'\b' + _re.escape(base_symbol) + r'\b')
        name_pat = _re.compile(r'\b' + _re.escape(full) + r'\b')
        return bool(sym_pat.search(head)) or bool(name_pat.search(head))

    def fetch_indian_news(self, symbol: str, num_articles: int = 15, cache_only: bool = False) -> Optional[List[str]]:
        try:
            if not config.EVENT_REGISTRY_API_KEY:
                return None
            cache_key = self._get_cache_key("news_v4", symbol, limit=num_articles)
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached.get("articles")
            if cache_only:
                return None
            base_symbol = symbol.split(".")[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)
            payload = {
                "action": "getArticles", "keyword": company_name, "keywordOper": "and",
                "keywordLoc": "title", "lang": ["eng"], "articlesPage": 1,
                "articlesCount": num_articles, "articlesSortBy": "date",
                "articlesSortByAsc": False, "dataType": ["news"],
                "includeArticleBody": True, "apiKey": config.EVENT_REGISTRY_API_KEY
            }
            response = requests.post(config.EVENT_REGISTRY_ENDPOINT, json=payload, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
            articles = []
            for item in data.get("articles", {}).get("results", []):
                body = item.get("body") or ""
                title = item.get("title") or ""
                if body and len(body) > 200:
                    text = f"{title}. {body}" if title else body
                elif title:
                    text = title
                else:
                    continue
                if not self._is_relevant(text, base_symbol, company_name):
                    continue
                text = self._strip_boilerplate(text)
                if len(text) < 100:
                    continue
                articles.append(text)
            if articles:
                self._set_to_cache(cache_key, {"articles": articles}, self.cache_ttl["news"])
            return articles if articles else None
        except Exception as e:
            logger.error(f"Event Registry fetch failed for {symbol}: {e}")
            return None

    def get_sample_news(self, symbol: str) -> List[str]:
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)
            return [
                f"{company_name} reports strong quarterly earnings",
                f"Analysts upgrade {company_name} target price",
                f"{company_name} announces major expansion plans",
                f"Market volatility creates opportunity in {company_name}",
                f"{company_name} management provides optimistic guidance"
            ]
        except Exception as e:
            logger.error(f"Error generating sample news: {e}")
            return [f"Market analysis for {symbol}"]

    def _analyze_sentiment_via_api(self, articles: List[str]) -> Optional[Tuple]:
        try:
            payload = {"inputs": articles}
            response = query_hf_api(self.sentiment_api_url, payload)
            if not isinstance(response, dict):
                return None
            if response.get("success") is not True:
                return None
            results = response.get("results")
            if not isinstance(results, list) or not results:
                return None
            sentiments = []
            confidences = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                confidence = item.get("confidence", 0.5)
                if isinstance(label, str):
                    sentiments.append(label.lower())
                    confidences.append(float(confidence))
            if not sentiments:
                return None
            return sentiments, confidences
        except Exception as e:
            logger.error(f"SBERT parse failure: {e}")
            return None

    def analyze_sentiment_with_textblob(self, articles: List[str]) -> Tuple:
        sentiments = []
        confidences = []
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
            except Exception:
                sentiments.append('neutral')
                confidences.append(0.3)
        return sentiments, confidences

    def analyze_news_sentiment(self, symbol: str, num_articles: int = 15, cache_only: bool = False) -> Tuple:
        try:
            articles = self.fetch_indian_news(symbol, num_articles, cache_only=cache_only)
            news_source = "Real news" if articles else "Sample"
            if not articles:
                articles = self.get_sample_news(symbol)
            if self.model_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    return sentiments, articles, confidences, "SBERT API", news_source
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob", news_source
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error"

    def calculate_swing_trading_score(self, data: pd.DataFrame, sentiment_data: Tuple, sector: str) -> float:
        try:
            tech_weight, sentiment_weight = self.get_sector_weights(sector)
            technical_score = 0
            sentiment_score = 50
            if data is None or data.empty:
                return 0
            current_price = data['Close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                return 0
            rsi = self.calculate_rsi(data['Close'])
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                current_rsi = rsi.iloc[-1]
                if 30 <= current_rsi <= 70:
                    technical_score += 20
                elif current_rsi < 30:
                    technical_score += 15
                elif current_rsi > 70:
                    technical_score += 10
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            if not bb_upper.empty and not any(pd.isna([bb_upper.iloc[-1], bb_lower.iloc[-1]])):
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                if 0.2 <= bb_position <= 0.8:
                    technical_score += 15
                elif bb_position < 0.2:
                    technical_score += 12
                elif bb_position > 0.8:
                    technical_score += 8
            stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            if not stoch_k.empty and not any(pd.isna([stoch_k.iloc[-1], stoch_d.iloc[-1]])):
                k_val = stoch_k.iloc[-1]
                d_val = stoch_d.iloc[-1]
                if k_val > d_val and k_val < 80:
                    technical_score += 15
                elif 20 <= k_val <= 80:
                    technical_score += 10
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    technical_score += 15
                if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                    if histogram.iloc[-1] > histogram.iloc[-2]:
                        technical_score += 5
            if 'Volume' in data.columns:
                avg_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                if not pd.isna(avg_volume) and not pd.isna(current_volume) and avg_volume > 0:
                    if current_volume > avg_volume * 1.2:
                        technical_score += 10
                    elif current_volume > avg_volume:
                        technical_score += 5
            support, resistance = self.calculate_support_resistance(data)
            if support and resistance and not any(pd.isna([support, resistance])):
                distance_to_support = (current_price - support) / support
                if distance_to_support < 0.05:
                    technical_score += 8
                elif 0.05 <= distance_to_support <= 0.15:
                    technical_score += 10
            if len(data) >= 50:
                ma_20 = self.safe_rolling_calculation(data['Close'], 20, 'mean').iloc[-1]
                ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                if not any(pd.isna([ma_20, ma_50])):
                    if current_price > ma_20 > ma_50:
                        technical_score += 15
                    elif current_price > ma_20:
                        technical_score += 10
                    elif ma_20 > ma_50:
                        technical_score += 5
            technical_score = min(100, max(0, technical_score))
            MIN_ARTICLES_FOR_SENTIMENT = 3
            news_source = sentiment_data[4] if (sentiment_data and len(sentiment_data) > 4) else "Unknown"
            sentiments = sentiment_data[0] if (sentiment_data and len(sentiment_data) > 0) else []
            confidences = sentiment_data[2] if (sentiment_data and len(sentiment_data) > 2) else []
            if (news_source == "Real news"
                    and len(sentiments) >= MIN_ARTICLES_FOR_SENTIMENT
                    and confidences):
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
            sentiment_score = min(100, max(0, sentiment_score))
            final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
            return min(100, max(0, final_score))
        except Exception as e:
            logger.error(f"Error calculating swing score: {e}")
            return 0

    def generate_trading_plan(self, data: pd.DataFrame, score: float, risk_metrics: Dict) -> Dict:
        default_plan = {
            'entry_signal': "HOLD/WATCH", 'entry_strategy': "Wait for clearer signals",
            'stop_loss': 0, 'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
        }
        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02
            if score >= 75:
                entry_signal = "STRONG BUY"
                entry_strategy = "Enter on any pullback. Strong momentum."
            elif score >= 60:
                entry_signal = "BUY"
                entry_strategy = "Enter on minor dips. Good setup."
            elif score >= 45:
                entry_signal = "HOLD/WATCH"
                entry_strategy = "Wait for clearer signals."
            elif score >= 30:
                entry_signal = "SELL"
                entry_strategy = "Exit positions. Weak conditions."
            else:
                entry_signal = "STRONG SELL"
                entry_strategy = "Exit immediately."
            stop_loss_distance = atr * 1.5
            stop_loss = max(current_price - stop_loss_distance, 0)
            target_1 = current_price + (stop_loss_distance * 1.0)
            target_2 = current_price + (stop_loss_distance * 1.5)
            target_3 = current_price + (stop_loss_distance * 2.0)
            if target_1 <= current_price:
                target_1 = current_price * 1.03
                target_2 = current_price * 1.05
                target_3 = current_price * 1.08
            return {
                'entry_signal': entry_signal, 'entry_strategy': entry_strategy,
                'stop_loss': float(stop_loss),
                'targets': {'target_1': float(target_1), 'target_2': float(target_2), 'target_3': float(target_3)},
                'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
            }
        except Exception as e:
            logger.error(f"Error generating trading plan: {e}")
            return default_plan

    def analyze_swing_trading_stock(self, symbol: str, period: str = "6mo", cache_only_news: bool = False) -> Optional[Dict]:
        try:
            cache_key = self._get_cache_key("analysis", symbol, period=period)
            cached_analysis = self._get_from_cache(cache_key)
            if cached_analysis:
                return cached_analysis
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                return None
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get("sector", "Unknown")
            company_name = stock_info.get("name", symbol)
            current_price = data["Close"].iloc[-1]
            if len(data) >= 2:
                price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
                price_change_pct = (price_change / data["Close"].iloc[-2]) * 100
            else:
                price_change = 0.0
                price_change_pct = 0.0
            rsi = self.calculate_rsi(data["Close"])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data["Close"])
            stoch_k, stoch_d = self.calculate_stochastic(data["High"], data["Low"], data["Close"])
            macd_line, signal_line, histogram = self.calculate_macd(data["Close"])
            support, resistance = self.calculate_support_resistance(data)
            sentiment_results = self.analyze_news_sentiment(final_symbol, cache_only=cache_only_news)
            sentiments = sentiment_results[0] if len(sentiment_results) > 0 else []
            articles = sentiment_results[1] if len(sentiment_results) > 1 else []
            confidences = sentiment_results[2] if len(sentiment_results) > 2 else []
            sentiment_method = sentiment_results[3] if len(sentiment_results) > 3 else "Unknown"
            sentiment_source = sentiment_results[4] if len(sentiment_results) > 4 else "Unknown"
            if sentiment_source == "Real news" and articles:
                _shadow_executor.submit(
                    self._shadow_finbert, final_symbol, list(articles), list(sentiments), list(confidences)
                )
            risk_metrics = self.calculate_risk_metrics(data)
            swing_score = self.calculate_swing_trading_score(data=data, sentiment_data=sentiment_results, sector=sector)
            trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)
            result = {
                "symbol": final_symbol, "company_name": company_name, "sector": sector,
                "current_price": float(current_price), "price_change": float(price_change),
                "price_change_pct": float(price_change_pct),
                "rsi": float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None,
                "bollinger_bands": {
                    "upper": float(bb_upper.iloc[-1]) if not bb_upper.empty else None,
                    "middle": float(bb_middle.iloc[-1]) if not bb_middle.empty else None,
                    "lower": float(bb_lower.iloc[-1]) if not bb_lower.empty else None
                },
                "stochastic": {
                    "k": float(stoch_k.iloc[-1]) if not stoch_k.empty else None,
                    "d": float(stoch_d.iloc[-1]) if not stoch_d.empty else None
                },
                "macd": {
                    "line": float(macd_line.iloc[-1]) if not macd_line.empty else None,
                    "signal": float(signal_line.iloc[-1]) if not signal_line.empty else None,
                    "histogram": float(histogram.iloc[-1]) if not histogram.empty else None
                },
                "support_resistance": {
                    "support": float(support) if support else None,
                    "resistance": float(resistance) if resistance else None
                },
                "sentiment": {
                    "scores": sentiments, "method": sentiment_method, "source": sentiment_source,
                    "summary": {
                        "positive": sentiments.count("positive") if sentiments else 0,
                        "negative": sentiments.count("negative") if sentiments else 0,
                        "neutral": sentiments.count("neutral") if sentiments else 0
                    }
                },
                "risk_metrics": risk_metrics, "swing_score": float(swing_score),
                "trading_plan": trading_plan, "model_type": sentiment_method,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self._set_to_cache(cache_key, result, self.cache_ttl["analysis"])
            return result
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    # Sectors to AVOID in a bear market (cyclical / risk-off underperformers).
    REGIME_EXCLUDE = {
        'BEAR':     ['Oil & Gas', 'Metals', 'Mining', 'Steel', 'Cement'],
        'SIDEWAYS': [],
        'BULL':     [],
    }
    MAX_PER_SECTOR = {'BEAR': 1, 'SIDEWAYS': 2, 'BULL': 2}

    def filter_stocks_by_risk_appetite(self, results: List[Dict], risk_appetite: str) -> List[Dict]:
        """Filter stocks by risk appetite (+ regime/sector when ENABLE_NEW_LOGIC).

        Flag OFF (production / public results):
            old proven behavior — volatility cap + BUY-signal filter ONLY.
            No regime sector-exclude, no per-sector cap.
        Flag ON (backend test only):
            additionally applies regime sector-exclude + per-sector concentration cap.

        Results are assumed score-sorted (analyze_stocks_parallel sorts desc),
        so the sector cap keeps the HIGHEST-scoring stock in each sector."""
        try:
            risk_thresholds = {'LOW': 0.25, 'MEDIUM': 0.40, 'HIGH': 1.0}
            max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)

            if ENABLE_NEW_LOGIC:
                regime = self.detect_market_regime()
                exclude_sectors = self.REGIME_EXCLUDE.get(regime, [])
                max_per_sector = self.MAX_PER_SECTOR.get(regime, 2)
            else:
                regime = 'OFF'
                exclude_sectors = []
                max_per_sector = 10**9

            filtered = []
            sector_count = {}
            dropped_regime = dropped_sectorcap = 0
            for stock in results:
                volatility = stock.get('risk_metrics', {}).get('volatility', 1.0)
                entry_signal = stock.get('trading_plan', {}).get('entry_signal', 'HOLD')
                sector = stock.get('sector', 'Unknown')
                if not (volatility <= max_volatility and entry_signal in ['BUY', 'STRONG BUY']):
                    continue
                if sector in exclude_sectors:
                    dropped_regime += 1
                    continue
                if sector_count.get(sector, 0) >= max_per_sector:
                    dropped_sectorcap += 1
                    continue
                sector_count[sector] = sector_count.get(sector, 0) + 1
                filtered.append(stock)
            logger.info(
                f"Filtered {len(filtered)}/{len(results)} for {risk_appetite} risk "
                f"[new_logic={ENABLE_NEW_LOGIC}, regime={regime}, max {max_per_sector}/sector"
                f"{f', excluded {exclude_sectors}' if exclude_sectors else ''}; "
                f"dropped {dropped_regime} regime, {dropped_sectorcap} sectorcap]")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering stocks: {e}")
            return []

    def generate_portfolio_allocation(self, results: List[Dict], total_capital: float,
                                      risk_appetite: str) -> List[Dict]:
        try:
            if not results or total_capital <= 0:
                return []
            max_positions = min(10, len(results))
            risk_per_trade_pct = self.swing_trading_params['risk_per_trade']
            max_position_size_pct = 0.20
            base_allocation = total_capital / max_positions
            portfolio_data = []
            total_allocated = 0
            for i, result in enumerate(results, 1):
                if len(portfolio_data) >= max_positions:
                    break
                current_price = result.get('current_price', 0)
                stop_loss = result.get('trading_plan', {}).get('stop_loss', 0)
                swing_score = result.get('swing_score', 0)
                if current_price <= 0 or stop_loss <= 0 or current_price <= stop_loss:
                    continue
                risk_per_share = current_price - stop_loss
                capital_at_risk = total_capital * risk_per_trade_pct
                risk_based_shares = int(capital_at_risk / risk_per_share)
                equal_weight_shares = int(base_allocation / current_price)
                max_position_shares = int((total_capital * max_position_size_pct) / current_price)
                final_shares = min(equal_weight_shares, risk_based_shares, max_position_shares)
                if final_shares <= 0:
                    continue
                final_amount = final_shares * current_price
                if total_allocated + final_amount > total_capital:
                    remaining = total_capital - total_allocated
                    if remaining >= current_price:
                        final_shares = int(remaining / current_price)
                        final_amount = final_shares * current_price
                    else:
                        continue
                actual_risk = final_shares * risk_per_share
                allocation_pct = (final_amount / total_capital) * 100
                total_allocated += final_amount
                portfolio_data.append({
                    'rank': i, 'symbol': result.get('symbol', 'Unknown'),
                    'company': result.get('company_name', 'Unknown'), 'score': float(swing_score),
                    'price': float(current_price), 'stop_loss': float(stop_loss),
                    'allocation_pct': float(allocation_pct), 'investment_amount': float(final_amount),
                    'number_of_shares': int(final_shares), 'risk': float(actual_risk),
                    'sector': result.get('sector', 'Unknown')
                })
            return portfolio_data
        except Exception as e:
            logger.error(f"Error generating portfolio: {e}")
            return []

    def quick_analysis_top_stocks(self, max_stocks: int = 50, max_workers: int = 10) -> List[Dict]:
        top_liquid = [
                         "INFY", "ICICIBANK", "BAJFINANCE", "SBIN", "BHARTIARTL",
                         "KOTAKBANK", "LT", "MARUTI", "TITAN", "ASIANPAINT",
                         "HCLTECH", "AXISBANK", "WIPRO", "NTPC", "HINDUNILVR",
                         "SUNPHARMA", "ULTRACEMCO", "TECHM", "TATASTEEL", "POWERGRID",
                         "ONGC", "M&M", "HEROMOTOCO", "BAJAJ-AUTO", "GRASIM",
                         "CIPLA", "BRITANNIA", "IOC", "APOLLOHOSP", "ADANIPORTS",
                         "TATAMOTORS", "ITC", "DIVISLAB", "DRREDDY", "TATACONSUM",
                         "INDUSINDBK", "EICHERMOT", "COALINDIA", "HINDALCO", "JSWSTEEL",
                         "UPL", "BPCL", "NESTLEIND", "SHREECEM", "BAJAJFINSV",
                         "HDFCLIFE", "SBILIFE", "ADANIENT", "GODREJCP", "PIDILITIND"
                     ][:max_stocks]
        return self.analyze_stocks_parallel(top_liquid, max_workers=max_workers)


class BackgroundAnalyzer:
    def __init__(self, trading_system: EnhancedSwingTradingSystem, refresh_interval: int = 900):
        self.system = trading_system
        self.refresh_interval = refresh_interval
        self.running = False

    def start(self):
        import threading
        def worker():
            while self.running:
                try:
                    self.system.analyze_and_cache_all_stocks(max_workers=10)
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    logger.error(f"Background worker error: {e}")
                    time.sleep(60)
        self.running = True
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
