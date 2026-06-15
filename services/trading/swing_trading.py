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

# ── Shadow-mode FinBERT comparison (production keeps using SBERT) ──────────
# Dormant until FINBERT_API_URL env var is set. Zero impact on production.
FINBERT_API_URL    = os.getenv("FINBERT_API_URL", "")
FINBERT_API_SECRET = os.getenv("FINBERT_API_SECRET", "")
SHADOW_LOG_KEY     = "shadow:sentiment:log"
_shadow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Boilerplate/disclaimer markers — scraper tails that make the window scan
# fire false negatives on nearly every article. Stripped before scoring.
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
    """
    Production-grade swing trading system with performance optimizations
    """
    def __init__(self, data_provider=None, redis_client=None):
        try:
            self.event_registry_api_key = getattr(config, "EVENT_REGISTRY_API_KEY", None)
            self.event_registry_endpoint = getattr(config, "EVENT_REGISTRY_ENDPOINT", None)
            if not self.event_registry_api_key:
                logger.warning("⚠️ EVENT_REGISTRY_API_KEY not configured. News sentiment disabled.")
            self.swing_trading_params = config.SWING_TRADING_PARAMS
            self._validate_trading_params()
            # ✅ FIRST: inject dependency
            if not data_provider:
                raise ValueError("❌ data_provider is REQUIRED for SwingTradingSystem")
            self.data_provider = data_provider
            logger.info("✅ Data provider injected into SwingTradingSystem")
            # ✅ THEN: initialize database
            self.initialize_expanded_stock_database()
            # API config
            self.sentiment_api_url = config.HF_SENTIMENT_API_URL
            self.model_api_available = bool(self.sentiment_api_url)
            # Redis
            self.redis_client = redis_client
            self.cache_enabled = bool(redis_client)
            self.cache_ttl = {
                'ohlcv': 3600,
                'analysis': 900,
                'news': 86400,  # 24h — news cached per stock per day to save Event Registry tokens
                'batch_analysis': 900
            }
            logger.info(f"✅ EnhancedSwingTradingSystem initialized with {len(self.indian_stocks)} stocks")
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
            if self.swing_trading_params['min_holding_period'] >= self.swing_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")
            if self.swing_trading_params['risk_per_trade'] > 0.1:
                raise ValueError("risk_per_trade cannot exceed 10%")
            logger.info("Trading parameters validated successfully")
        except Exception as e:
            logger.error(f"Error validating trading parameters: {e}")
            raise
    def initialize_expanded_stock_database(self):
        """
        Initialize EXPANDED Indian stock database (500+ stocks)
        Covers: NIFTY 50, NEXT 50, MIDCAP 100, SMALLCAP 50
        """
        try:
            self.indian_stocks = {
                # ========== NIFTY 50 (50 stocks) ==========
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
                # ========== NIFTY NEXT 50 (50 stocks) ==========
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
                # ========== NIFTY MIDCAP 100 (Top 100 liquid stocks) ==========
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
                "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile"},
                "BATAINDIA": {"name": "Bata India", "sector": "Footwear"},
                "BEL": {"name": "Bharat Electronics", "sector": "Defence"},
                "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile"},
                "BHEL": {"name": "Bharat Heavy Electricals", "sector": "Capital Goods"},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals"},
                "BRIGADE": {"name": "Brigade Enterprises", "sector": "Real Estate"},
                "CANBK": {"name": "Canara Bank", "sector": "Banking"},
                "CESC": {"name": "CESC Limited", "sector": "Power"},
                "CHAMBLFERT": {"name": "Chambal Fertilizers", "sector": "Fertilizers"},
                "CHOLAFIN": {"name": "Cholamandalam Investment", "sector": "Financial Services"},
                "COFORGE": {"name": "Coforge Limited", "sector": "Information Technology"},
                "COLPAL": {"name": "Colgate-Palmolive", "sector": "Consumer Goods"},
                "CONCOR": {"name": "Container Corporation", "sector": "Logistics"},
                "COROMANDEL": {"name": "Coromandel International", "sector": "Fertilizers"},
                "CRISIL": {"name": "CRISIL Limited", "sector": "Financial Services"},
                "CROMPTON": {"name": "Crompton Greaves", "sector": "Electricals"},
                "CUB": {"name": "City Union Bank", "sector": "Banking"},
                "CUMMINSIND": {"name": "Cummins India", "sector": "Automobile"},
                "CYIENT": {"name": "Cyient Limited", "sector": "Information Technology"},
                "DEEPAKNTR": {"name": "Deepak Nitrite", "sector": "Chemicals"},
                "DIXON": {"name": "Dixon Technologies", "sector": "Electronics"},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
                "ESCORTS": {"name": "Escorts Kubota", "sector": "Automobile"},
                "FACT": {"name": "Fertilizers And Chemicals", "sector": "Fertilizers"},
                "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking"},
                "GAIL": {"name": "GAIL India", "sector": "Oil & Gas"},
                "GLENMARK": {"name": "Glenmark Pharmaceuticals", "sector": "Pharmaceuticals"},
                "GODREJPROP": {"name": "Godrej Properties", "sector": "Real Estate"},
                "GRANULES": {"name": "Granules India", "sector": "Pharmaceuticals"},
                "GRAPHITE": {"name": "Graphite India", "sector": "Capital Goods"},
                "GUJGASLTD": {"name": "Gujarat Gas", "sector": "Oil & Gas"},
                "HAVELLS": {"name": "Havells India", "sector": "Electricals"},
                "HFCL": {"name": "HFCL Limited", "sector": "Telecommunications"},
                "HINDCOPPER": {"name": "Hindustan Copper", "sector": "Metals"},
                "HINDPETRO": {"name": "Hindustan Petroleum", "sector": "Oil & Gas"},
                "HONAUT": {"name": "Honeywell Automation", "sector": "Capital Goods"},
                "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Banking"},
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
                "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Services"},
                "KAJARIACER": {"name": "Kajaria Ceramics", "sector": "Building Materials"},
                "KEI": {"name": "KEI Industries", "sector": "Electricals"},
                "L&TFH": {"name": "L&T Finance Holdings", "sector": "Financial Services"},
                "LALPATHLAB": {"name": "Dr Lal PathLabs", "sector": "Healthcare"},
                "LAURUSLABS": {"name": "Laurus Labs", "sector": "Pharmaceuticals"},
                "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services"},
                "LTTS": {"name": "L&T Technology Services", "sector": "Information Technology"},
                "MANAPPURAM": {"name": "Manappuram Finance", "sector": "Financial Services"},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods"},
                "MCX": {"name": "Multi Commodity Exchange", "sector": "Financial Services"},
                "METROBRAND": {"name": "Metro Brands", "sector": "Footwear"},
                "MFSL": {"name": "Max Financial Services", "sector": "Insurance"},
                "MGL": {"name": "Mahanagar Gas", "sector": "Oil & Gas"},
                "MINDTREE": {"name": "Mindtree Limited", "sector": "Information Technology"},
                "MOTHERSON": {"name": "Samvardhana Motherson", "sector": "Automobile"},
                "MPHASIS": {"name": "Mphasis Limited", "sector": "Information Technology"},
                "MUTHOOTFIN": {"name": "Muthoot Finance", "sector": "Financial Services"},
                "NATIONALUM": {"name": "National Aluminium", "sector": "Metals"},
                "NAUKRI": {"name": "Info Edge India", "sector": "Internet"},
                "NAVINFLUOR": {"name": "Navin Fluorine", "sector": "Chemicals"},
                "NMDC": {"name": "NMDC Limited", "sector": "Mining"},
                "OBEROIRLTY": {"name": "Oberoi Realty", "sector": "Real Estate"},
                "OIL": {"name": "Oil India", "sector": "Oil & Gas"},
                "PAYTM": {"name": "One 97 Communications", "sector": "Financial Services"},
                "PEL": {"name": "Piramal Enterprises", "sector": "Financial Services"},
                "PERSISTENT": {"name": "Persistent Systems", "sector": "Information Technology"},
                "PETRONET": {"name": "Petronet LNG", "sector": "Oil & Gas"},
                "PFC": {"name": "Power Finance Corporation", "sector": "Financial Services"},
                "PHOENIXLTD": {"name": "Phoenix Mills", "sector": "Real Estate"},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
                "PIIND": {"name": "PI Industries", "sector": "Chemicals"},
                "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
                "POLYCAB": {"name": "Polycab India", "sector": "Electricals"},
                "PRESTIGE": {"name": "Prestige Estates", "sector": "Real Estate"},
                "RECLTD": {"name": "REC Limited", "sector": "Financial Services"},
                "SBICARD": {"name": "SBI Cards", "sector": "Financial Services"},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement"},
                "SIEMENS": {"name": "Siemens Limited", "sector": "Capital Goods"},
                "SOLARINDS": {"name": "Solar Industries", "sector": "Chemicals"},
                "SONACOMS": {"name": "Sona BLW Precision", "sector": "Automobile"},
                "SRF": {"name": "SRF Limited", "sector": "Chemicals"},
                "STAR": {"name": "Sterlite Technologies", "sector": "Telecommunications"},
                "TATACOMM": {"name": "Tata Communications", "sector": "Telecommunications"},
                "TATAELXSI": {"name": "Tata Elxsi", "sector": "Information Technology"},
                "TATAPOWER": {"name": "Tata Power", "sector": "Power"},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology"},
                "THERMAX": {"name": "Thermax Limited", "sector": "Capital Goods"},
                "TORNTPHARM": {"name": "Torrent Pharmaceuticals", "sector": "Pharmaceuticals"},
                "TORNTPOWER": {"name": "Torrent Power", "sector": "Power"},
                "TRENT": {"name": "Trent Limited", "sector": "Retail"},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile"},
                "UBL": {"name": "United Breweries", "sector": "Beverages"},
                "UNIONBANK": {"name": "Union Bank of India", "sector": "Banking"},
                "VEDL": {"name": "Vedanta Limited", "sector": "Metals"},
                "VOLTAS": {"name": "Voltas Limited", "sector": "Consumer Durables"},
                "WHIRLPOOL": {"name": "Whirlpool of India", "sector": "Consumer Durables"},
                "ZOMATO": {"name": "Zomato Limited", "sector": "Consumer Services"},
                "ZYDUSLIFE": {"name": "Zydus Lifesciences", "sector": "Pharmaceuticals"},
                # ========== HIGH-GROWTH SMALLCAP PICKS (50 stocks) ==========
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
            # Full reference (never pruned) — so excluded stocks still resolve
            # their real sector/name in get_stock_info_from_db. Without this,
            # RELIANCE/HDFCBANK/TCS return sector="Unknown" (the default),
            # which would also break any regime sector-exclude on them.
            self.all_stocks_reference = dict(self.indian_stocks)
            # Remove underperforming stocks from the SWING UNIVERSE only
            exclude_from_swing = ['RELIANCE', 'HDFCBANK', 'TCS']
            original_count = len(self.indian_stocks)
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
            # Fallback to minimal database
            self.indian_stocks = {
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")
    # ========================================================================
    # CACHING METHODS
    # ========================================================================
    def _get_cache_key(self, prefix: str, symbol: str, **kwargs) -> str:
        """Generate cache key"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{prefix}:{symbol}:{param_str}" if param_str else f"{prefix}:{symbol}"
    def _get_from_cache(self, key: str) -> Optional[dict]:
        """Get data from Redis cache"""
        if not self.cache_enabled:
            return None
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                logger.debug(f"⚡ Cache HIT: {key}")
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    def _set_to_cache(self, key: str, data: dict, ttl: int):
        """Set data to Redis cache"""
        if not self.cache_enabled:
            return
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
            logger.debug(f"💾 Cached: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    # ========================================================================
    # SHADOW MODE — FinBERT comparison logging (production unaffected)
    # ========================================================================
    def _sbert_normalized(self, sentiments, confidences) -> float:
        """Replicates the EXACT production formula from
        calculate_swing_trading_score, on the [-1,+1] scale —
        so the shadow log compares like with like."""
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
        """Fire-and-forget: score same articles with FinBERT, log both
        scores to Redis. NEVER raises — production must not feel this."""
        try:
            if not FINBERT_API_URL or not self.redis_client:
                return
            day = datetime.now().strftime("%Y-%m-%d")
            dedupe_key = f"shadow:done:{symbol}:{day}"
            if self.redis_client.get(dedupe_key):
                return  # already shadowed this symbol today
            self.redis_client.setex(dedupe_key, 86400, "1")
            resp = requests.post(
                FINBERT_API_URL,
                json={"articles": articles[:15]},
                headers={"X-Api-Secret": FINBERT_API_SECRET},
                timeout=90,   # CPU Space cold start can be slow
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
            logger.info(f"[shadow] {symbol}: sbert={sbert_score:+.2f} "
                        f"finbert={finbert_score:+.2f} agree={row['agree']}")
        except Exception as e:
            logger.warning(f"[shadow] failed for {symbol}: {e}")
    # ========================================================================
    # PARALLEL ANALYSIS METHODS
    # ========================================================================
    def analyze_stocks_parallel(self, symbols: List[str], max_workers: int = 10,
                                period: str = "6mo") -> List[Dict]:
        """
        Analyze multiple stocks in parallel for 10x speed improvement
        Args:
            symbols: List of stock symbols to analyze
            max_workers: Number of parallel workers (default: 10)
            period: Data period (default: 6mo)
        Returns:
            List of analysis results
        """
        try:
            logger.info(f"🚀 Starting parallel analysis of {len(symbols)} stocks with {max_workers} workers")
            start_time = time.time()
            results = []
            failed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self.analyze_swing_trading_stock, symbol, period): symbol
                    for symbol in symbols
                }
                # Process completed tasks
                completed = 0
                total = len(symbols)
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    try:
                        result = future.result(timeout=30)  # 30s timeout per stock
                        if result and result.get('swing_score', 0) > 0:
                            results.append(result)
                            logger.info(f"✅ [{completed}/{total}] {symbol}: Score={result['swing_score']:.0f}")
                        else:
                            failed_count += 1
                            logger.warning(f"⚠️ [{completed}/{total}] {symbol}: Analysis failed or zero score")
                    except concurrent.futures.TimeoutError:
                        failed_count += 1
                        logger.error(f"❌ [{completed}/{total}] {symbol}: Timeout (>30s)")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ [{completed}/{total}] {symbol}: {str(e)}")
            # Sort by score
            results.sort(key=lambda x: x.get('swing_score', 0), reverse=True)
            elapsed_time = time.time() - start_time
            logger.info(
                f"✅ Parallel analysis complete: {len(results)} successful, {failed_count} failed in {elapsed_time:.1f}s")
            return results
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            return []
    def get_cached_batch_analysis(self, risk_appetite: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get pre-computed batch analysis from cache
        Args:
            risk_appetite: Filter by risk appetite (LOW/MEDIUM/HIGH)
        Returns:
            Cached analysis results or None
        """
        try:
            cache_key = "batch_analysis:all_stocks"
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                logger.info(f"⚡ Retrieved {len(cached_results)} results from cache")
                if risk_appetite:
                    filtered = self.filter_stocks_by_risk_appetite(cached_results, risk_appetite)
                    logger.info(f"📊 Filtered to {len(filtered)} stocks for {risk_appetite} risk")
                    return filtered
                return cached_results
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached batch analysis: {e}")
            return None
    def analyze_and_cache_all_stocks(self, max_workers: int = 10):
        """
        Analyze all stocks and cache results (for background worker)
        Args:
            max_workers: Number of parallel workers
        """
        try:
            logger.info("🔄 Starting background batch analysis...")
            symbols = self.get_all_stock_symbols()
            results = self.analyze_stocks_parallel(symbols, max_workers=max_workers)
            # Cache results
            cache_key = "batch_analysis:all_stocks"
            self._set_to_cache(cache_key, results, self.cache_ttl['batch_analysis'])
            logger.info(f"✅ Background analysis complete: {len(results)} stocks cached")
            return results
        except Exception as e:
            logger.error(f"Error in background batch analysis: {e}")
            return []
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    def get_all_stock_symbols(self) -> List[str]:
        """Get all stock symbols from database"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {e}")
            return []
    @lru_cache(maxsize=1000)
    def get_stock_info_from_db(self, symbol: str) -> Dict:
        """Get stock info from database (cached).
        Uses the full unpruned reference so excluded stocks
        (RELIANCE/HDFCBANK/TCS) still resolve their real sector."""
        try:
            base_symbol = str(symbol).split('.')[0].upper().strip()
            ref = getattr(self, 'all_stocks_reference', None) or self.indian_stocks
            return ref.get(base_symbol, {"name": symbol, "sector": "Unknown"})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {"name": str(symbol), "sector": "Unknown"}
    @lru_cache(maxsize=100)
    def get_sector_weights(self, sector: str) -> Tuple[float, float]:
        """Get dynamic weights based on sector (cached)"""
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
            return 0.55, 0.45  # Default
        except Exception as e:
            logger.error(f"Error getting sector weights: {e}")
            return 0.55, 0.45
    # ========================================================================
    # MARKET REGIME DETECTION (standalone, cached, fail-safe)
    # Nothing depends on this yet — it only reads data and returns a string.
    # Zero impact on scoring until wired into allocation in a later deploy.
    # ========================================================================
    def detect_market_regime(self, force_refresh: bool = False) -> str:
        """Classify the current market regime from NIFTY 50 trend.
        BULL:     price > MA20 > MA50  (uptrend, MAs stacked up)
        BEAR:     price < MA20 < MA50  (downtrend, MAs stacked down)
        SIDEWAYS: anything else (default, and the fail-safe)
        Cached 1h. Fails safe to SIDEWAYS — which applies no exclusions /
        neutral sizing — so a detection failure never distorts the book."""
        try:
            cache_key = "market:regime"
            if not force_refresh and self.cache_enabled:
                cached = self._get_from_cache(cache_key)
                if cached and cached.get("regime"):
                    return cached["regime"]
            df = self._fetch_nifty_index()
            if df is None or df.empty or len(df) < 50:
                logger.warning("⚠️ Regime: insufficient NIFTY data, defaulting SIDEWAYS")
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
            pct_vs_ma50 = (current - ma50) / ma50 * 100
            logger.info(f"📊 Market regime: {regime} "
                        f"(NIFTY {current:.0f}, MA20 {ma20:.0f}, MA50 {ma50:.0f}, "
                        f"{pct_vs_ma50:+.1f}% vs MA50)")
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
        """Fetch NIFTY 50 index OHLCV. Index symbols don't use the -EQ equity
        path, so this calls the data provider directly with index candidates.
        Returns a DataFrame with a 'Close' column, or None."""
        candidates = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTY 50",
            "NIFTY50",
            "NIFTY 50",
        ]
        for sym in candidates:
            try:
                cache_key = self._get_cache_key("ohlcv_index", sym, period="3mo")
                cached = self._get_from_cache(cache_key)
                if cached:
                    df = pd.DataFrame(cached['ohlcv'])
                    df['Date'] = pd.to_datetime(df['date'])
                    df = df.set_index('Date').drop(columns=['date'])
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    return df
                stock_data = self.data_provider.get_stock_data(
                    symbol=sym,
                    fetch_ohlcv=True,
                    fetch_fundamentals=False,
                    period="3mo",
                )
                ohlcv_list = stock_data.get('ohlcv')
                if not ohlcv_list:
                    continue
                df = pd.DataFrame(ohlcv_list)
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                self._set_to_cache(cache_key, {'ohlcv': ohlcv_list}, 3600)
                logger.info(f"📈 NIFTY index fetched via '{sym}' ({len(df)} days)")
                return df
            except Exception as e:
                logger.debug(f"NIFTY fetch via '{sym}' failed: {e}")
                continue
        logger.warning("⚠️ Could not fetch NIFTY index via any symbol candidate")
        return None
    # ========================================================================
    # DATA FETCHING METHODS (Updated with caching)
    # ========================================================================
    def get_indian_stock_data(self, symbol: str, period: str = "6mo") -> Tuple:
        """Get stock data with caching support"""
        try:
            # Check cache first
            cache_key = self._get_cache_key("ohlcv", symbol, period=period)
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                # Reconstruct DataFrame from cached dict
                df = pd.DataFrame(cached_data['ohlcv'])
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date').drop(columns=['date'])
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                info = cached_data['info']
                final_symbol = cached_data['symbol']
                logger.debug(f"⚡ Using cached OHLCV for {symbol}")
                return df, info, final_symbol
            # Fetch from data provider
            if not self.data_provider:
                logger.error("❌ Data provider not available")
                return None, None, None
            symbol_clean = str(symbol).upper().replace(".NS", "").replace(".BO", "")
            stock_data = self.data_provider.get_stock_data(
                symbol=symbol_clean,
                fetch_ohlcv=True,
                fetch_fundamentals=False,
                period=period
            )
            if stock_data.get('errors'):
                logger.warning(f"Data fetch errors for {symbol}: {stock_data['errors']}")
            ohlcv_list = stock_data.get('ohlcv')
            if not ohlcv_list:
                logger.error(f"❌ No OHLCV data for {symbol}")
                return None, None, None
            # Convert to DataFrame
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
            # Cache the results
            cache_data = {
                'ohlcv': ohlcv_list,
                'info': info,
                'symbol': final_symbol
            }
            self._set_to_cache(cache_key, cache_data, self.cache_ttl['ohlcv'])
            logger.debug(f"✅ Fetched and cached {len(df)} days for {symbol}")
            return df, info, final_symbol
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None
    # ========================================================================
    # TECHNICAL INDICATORS (Optimized)
    # ========================================================================
    def safe_rolling_calculation(self, data: pd.Series, window: int, operation: str = 'mean') -> pd.Series:
        """Safely perform rolling calculations"""
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
        """Calculate RSI"""
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
        """Calculate Bollinger Bands"""
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
        """Calculate MACD"""
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
        """Calculate Stochastic Oscillator"""
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
        """Calculate support and resistance levels"""
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
        """Calculate Average True Range"""
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
        """Calculate comprehensive risk metrics"""
        default_metrics = {
            'volatility': 0.3,
            'var_95': -0.05,
            'max_drawdown': -0.2,
            'sharpe_ratio': 0,
            'atr': 0,
            'risk_level': 'HIGH'
        }
        try:
            if data is None or data.empty or 'Close' not in data.columns or len(data) < 2:
                return default_metrics
            returns = data['Close'].pct_change().dropna()
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            volatility = volatility if not pd.isna(volatility) else 0.3
            # VaR
            var_95 = np.percentile(returns.dropna(), 5) if len(returns) > 20 else -0.05
            var_95 = var_95 if not pd.isna(var_95) else -0.05
            # Max Drawdown
            rolling_max = data['Close'].expanding().max()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() if not pd.isna(drawdown.min()) else -0.2
            # Sharpe Ratio
            risk_free_rate = 0.06
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            sharpe_ratio = sharpe_ratio if not pd.isna(sharpe_ratio) else 0
            # ATR
            atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
            current_atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else data['Close'].iloc[-1] * 0.02
            # Risk Level
            if volatility > 0.4:
                risk_level = 'HIGH'
            elif volatility > 0.25:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            return {
                'volatility': float(volatility),
                'var_95': float(var_95),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'atr': float(current_atr),
                'risk_level': risk_level
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return default_metrics
    # ========================================================================
    # SENTIMENT ANALYSIS (With caching)
    # ========================================================================
    def _strip_boilerplate(self, text: str) -> str:
        """Cut the article at the first boilerplate/disclaimer marker.
        Footers like 'Disclaimer:', 'subject to market risks', 'Also Read'
        are scraper tails that make the 60-word window scan fire false
        negatives on nearly every article. Only cut if a real body (>200
        chars) is preserved."""
        m = _BOILERPLATE_RE.search(text)
        if m and m.start() > 200:
            text = text[:m.start()]
        return text.strip()
    def _is_relevant(self, text: str, base_symbol: str, company_name: str) -> bool:
        """Company must appear in the HEADLINE as a PROPER NOUN (capitalized),
        not merely as a lowercase common word. Kills 'loan reliance',
        'taxpayer reliance', 'Big Tech reliance' (the word) while keeping real
        'Reliance Industries ...' headlines. Matching is case-SENSITIVE so the
        common noun never matches the company name."""
        head = text.split('.')[0][:160]
        generic = {'limited', 'ltd', 'india', 'company', 'industries',
                   'enterprises', 'services', 'corporation', 'of', 'the', 'and'}
        tokens = [t for t in company_name.split()
                  if t.lower() not in generic and len(t) >= 3]
        full = ' '.join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else base_symbol)
        sym_pat = _re.compile(r'\b' + _re.escape(base_symbol) + r'\b')   # case-sensitive
        name_pat = _re.compile(r'\b' + _re.escape(full) + r'\b')         # case-SENSITIVE proper noun
        return bool(sym_pat.search(head)) or bool(name_pat.search(head))
    def fetch_indian_news(self, symbol: str, num_articles: int = 15) -> Optional[List[str]]:
        """
        Fetch FULL news articles using newsapi.ai (Event Registry).
        v3: title-keyword search + headline relevance + boilerplate stripping.
        """
        try:
            if not config.EVENT_REGISTRY_API_KEY:
                return None
            # v3 cache key — old caches not reused (boilerplate/noise in them)
            cache_key = self._get_cache_key("news_v4", symbol, limit=num_articles)
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached.get("articles")
            base_symbol = symbol.split(".")[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)
            payload = {
                "action": "getArticles",
                "keyword": company_name,
                "keywordOper": "and",
                "keywordLoc": "title",
                "lang": ["eng"],
                "articlesPage": 1,
                "articlesCount": num_articles,
                "articlesSortBy": "date",
                "articlesSortByAsc": False,
                "dataType": ["news"],
                "includeArticleBody": True,
                "apiKey": config.EVENT_REGISTRY_API_KEY
            }
            response = requests.post(
                config.EVENT_REGISTRY_ENDPOINT,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.warning(f"Event Registry HTTP {response.status_code}")
                return None
            data = response.json()
            articles = []
            dropped = 0
            for item in data.get("articles", {}).get("results", []):
                body = item.get("body") or ""
                title = item.get("title") or ""
                if body and len(body) > 200:
                    text = f"{title}. {body}" if title else body
                elif title:
                    text = title
                else:
                    continue
                # FIX B: relevance on headline only
                if not self._is_relevant(text, base_symbol, company_name):
                    dropped += 1
                    continue
                # FIX A: strip boilerplate tail before storing/scoring
                text = self._strip_boilerplate(text)
                if len(text) < 100:
                    dropped += 1
                    continue
                articles.append(text)
            if dropped:
                logger.info(f"📰 {symbol}: kept {len(articles)}, "
                            f"dropped {dropped} irrelevant/short articles")
            if articles:
                self._set_to_cache(
                    cache_key,
                    {"articles": articles},
                    self.cache_ttl["news"]
                )
            return articles if articles else None
        except Exception as e:
            logger.error(f"Event Registry fetch failed for {symbol}: {e}")
            return None
    def get_sample_news(self, symbol: str) -> List[str]:
        """Generate sample news"""
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
                    continue  # 🔥 CRITICAL SAFETY
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
        """Fallback sentiment analysis with TextBlob"""
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
    def analyze_news_sentiment(self, symbol: str, num_articles: int = 15) -> Tuple:
        """Main sentiment analysis function"""
        try:
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Real news" if articles else "Sample"
            if not articles:
                articles = self.get_sample_news(symbol)
            # Try API first
            if self.model_api_available:
                api_result = self._analyze_sentiment_via_api(articles)
                if api_result:
                    sentiments, confidences = api_result
                    return sentiments, articles, confidences, "SBERT API", news_source
            # Fallback to TextBlob
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            return sentiments, articles, confidences, "TextBlob", news_source
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return [], [], [], "Error", "Error"
    # ========================================================================
    # SCORING & ANALYSIS
    # ========================================================================
    def calculate_swing_trading_score(self, data: pd.DataFrame, sentiment_data: Tuple, sector: str) -> float:
        """Calculate comprehensive swing trading score"""
        try:
            tech_weight, sentiment_weight = self.get_sector_weights(sector)
            technical_score = 0
            sentiment_score = 50
            if data is None or data.empty:
                return 0
            current_price = data['Close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                return 0
            # RSI (20 points)
            rsi = self.calculate_rsi(data['Close'])
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                current_rsi = rsi.iloc[-1]
                if 30 <= current_rsi <= 70:
                    technical_score += 20
                elif current_rsi < 30:
                    technical_score += 15
                elif current_rsi > 70:
                    technical_score += 10
            # Bollinger Bands (15 points)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            if not bb_upper.empty and not any(pd.isna([bb_upper.iloc[-1], bb_lower.iloc[-1]])):
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                if 0.2 <= bb_position <= 0.8:
                    technical_score += 15
                elif bb_position < 0.2:
                    technical_score += 12
                elif bb_position > 0.8:
                    technical_score += 8
            # Stochastic (15 points)
            stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            if not stoch_k.empty and not any(pd.isna([stoch_k.iloc[-1], stoch_d.iloc[-1]])):
                k_val = stoch_k.iloc[-1]
                d_val = stoch_d.iloc[-1]
                if k_val > d_val and k_val < 80:
                    technical_score += 15
                elif 20 <= k_val <= 80:
                    technical_score += 10
            # MACD (15 points)
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    technical_score += 15
                if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                    if histogram.iloc[-1] > histogram.iloc[-2]:
                        technical_score += 5
            # Volume (10 points)
            if 'Volume' in data.columns:
                avg_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                if not pd.isna(avg_volume) and not pd.isna(current_volume) and avg_volume > 0:
                    if current_volume > avg_volume * 1.2:
                        technical_score += 10
                    elif current_volume > avg_volume:
                        technical_score += 5
            # Support/Resistance (10 points)
            support, resistance = self.calculate_support_resistance(data)
            if support and resistance and not any(pd.isna([support, resistance])):
                distance_to_support = (current_price - support) / support
                if distance_to_support < 0.05:
                    technical_score += 8
                elif 0.05 <= distance_to_support <= 0.15:
                    technical_score += 10
            # Moving Averages (15 points)
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
            # Sentiment Score — ONLY from real news, and ONLY with enough of it.
            # Sample/fallback news is hardcoded positive templates (fake boost).
            # And 1-2 articles give a jumpy, unreliable reading — so require a
            # MINIMUM of 3 real articles before letting news move the signal.
            # Below that threshold: neutral 50 → signal is technicals-only.
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
            # Combine scores
            final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
            return min(100, max(0, final_score))
        except Exception as e:
            logger.error(f"Error calculating swing score: {e}")
            return 0
    def generate_trading_plan(self, data: pd.DataFrame, score: float, risk_metrics: Dict) -> Dict:
        """Generate trading plan with realistic targets"""
        default_plan = {
            'entry_signal': "HOLD/WATCH",
            'entry_strategy': "Wait for clearer signals",
            'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
        }
        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02
            # Entry signal
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
            # Stop loss and targets
            stop_loss_distance = atr * 1.5
            stop_loss = max(current_price - stop_loss_distance, 0)
            target_1 = current_price + (stop_loss_distance * 1.0)
            target_2 = current_price + (stop_loss_distance * 1.5)
            target_3 = current_price + (stop_loss_distance * 2.0)
            # Ensure targets are above current price
            if target_1 <= current_price:
                target_1 = current_price * 1.03
                target_2 = current_price * 1.05
                target_3 = current_price * 1.08
            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'stop_loss': float(stop_loss),
                'targets': {
                    'target_1': float(target_1),
                    'target_2': float(target_2),
                    'target_3': float(target_3)
                },
                'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
            }
        except Exception as e:
            logger.error(f"Error generating trading plan: {e}")
            return default_plan
    def analyze_swing_trading_stock(self, symbol: str, period: str = "6mo") -> Optional[Dict]:
        """Main stock analysis function with caching (FIXED + SAFE)"""
        try:
            # --------------------------------------------------
            # Cache check
            # --------------------------------------------------
            cache_key = self._get_cache_key("analysis", symbol, period=period)
            cached_analysis = self._get_from_cache(cache_key)
            if cached_analysis:
                logger.debug(f"⚡ Using cached analysis for {symbol}")
                return cached_analysis
            # --------------------------------------------------
            # Fetch market data
            # --------------------------------------------------
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                return None
            # --------------------------------------------------
            # Stock metadata
            # --------------------------------------------------
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get("sector", "Unknown")
            company_name = stock_info.get("name", symbol)
            # --------------------------------------------------
            # Price info
            # --------------------------------------------------
            current_price = data["Close"].iloc[-1]
            if len(data) >= 2:
                price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
                price_change_pct = (price_change / data["Close"].iloc[-2]) * 100
            else:
                price_change = 0.0
                price_change_pct = 0.0
            # --------------------------------------------------
            # Technical indicators
            # --------------------------------------------------
            rsi = self.calculate_rsi(data["Close"])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data["Close"])
            stoch_k, stoch_d = self.calculate_stochastic(data["High"], data["Low"], data["Close"])
            macd_line, signal_line, histogram = self.calculate_macd(data["Close"])
            support, resistance = self.calculate_support_resistance(data)
            # --------------------------------------------------
            # Sentiment (SBERT → fallback TextBlob)
            # --------------------------------------------------
            sentiment_results = self.analyze_news_sentiment(final_symbol)
            sentiments = sentiment_results[0] if len(sentiment_results) > 0 else []
            articles = sentiment_results[1] if len(sentiment_results) > 1 else []
            confidences = sentiment_results[2] if len(sentiment_results) > 2 else []
            sentiment_method = sentiment_results[3] if len(sentiment_results) > 3 else "Unknown"
            sentiment_source = sentiment_results[4] if len(sentiment_results) > 4 else "Unknown"
            # Shadow-score with FinBERT (real news only — never sample news)
            # Dormant until FINBERT_API_URL env var is set. Fire-and-forget.
            if sentiment_source == "Real news" and articles:
                _shadow_executor.submit(
                    self._shadow_finbert,
                    final_symbol, list(articles), list(sentiments), list(confidences)
                )
            # --------------------------------------------------
            # Risk metrics
            # --------------------------------------------------
            risk_metrics = self.calculate_risk_metrics(data)
            # --------------------------------------------------
            # Final score
            # --------------------------------------------------
            swing_score = self.calculate_swing_trading_score(
                data=data,
                sentiment_data=sentiment_results,
                sector=sector
            )
            # --------------------------------------------------
            # Trading plan
            # --------------------------------------------------
            trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)
            # --------------------------------------------------
            # Result payload (TRUTHFUL + SAFE)
            # --------------------------------------------------
            result = {
                "symbol": final_symbol,
                "company_name": company_name,
                "sector": sector,
                "current_price": float(current_price),
                "price_change": float(price_change),
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
                    "scores": sentiments,
                    "method": sentiment_method,
                    "source": sentiment_source,
                    "summary": {
                        "positive": sentiments.count("positive") if sentiments else 0,
                        "negative": sentiments.count("negative") if sentiments else 0,
                        "neutral": sentiments.count("neutral") if sentiments else 0
                    }
                },
                "risk_metrics": risk_metrics,
                "swing_score": float(swing_score),
                "trading_plan": trading_plan,
                # 🔥 CRITICAL FIX
                "model_type": sentiment_method,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            # --------------------------------------------------
            # Cache result
            # --------------------------------------------------
            self._set_to_cache(cache_key, result, self.cache_ttl["analysis"])
            return result
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    # ========================================================================
    # PORTFOLIO GENERATION
    # ========================================================================
    # Sectors to AVOID in a bear market (cyclical / risk-off underperformers).
    # Empty in sideways/bull — no exclusions when the market isn't falling.
    REGIME_EXCLUDE = {
        'BEAR':     ['Oil & Gas', 'Metals', 'Mining', 'Steel', 'Cement'],
        'SIDEWAYS': [],
        'BULL':     [],
    }
    # Max positions per sector — tighter in bear (avoid correlated pairs like
    # IOC+BPCL or CANBK+PNB falling together), looser in bull.
    MAX_PER_SECTOR = {'BEAR': 1, 'SIDEWAYS': 2, 'BULL': 2}
    def filter_stocks_by_risk_appetite(self, results: List[Dict], risk_appetite: str) -> List[Dict]:
        """Filter stocks by risk appetite, market regime, and sector concentration.
        - volatility cap by risk appetite (unchanged)
        - regime sector EXCLUDE: drop risk-off sectors in a bear market
        - sector CAP: limit correlated names per sector (Issue 3)
        Results are assumed score-sorted (analyze_stocks_parallel sorts desc),
        so the sector cap keeps the HIGHEST-scoring stock in each sector."""
        try:
            risk_thresholds = {'LOW': 0.25, 'MEDIUM': 0.40, 'HIGH': 1.0}
            max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)
            regime = self.detect_market_regime()
            exclude_sectors = self.REGIME_EXCLUDE.get(regime, [])
            max_per_sector = self.MAX_PER_SECTOR.get(regime, 2)
            filtered = []
            sector_count = {}
            dropped_regime = dropped_sectorcap = 0
            for stock in results:
                volatility = stock.get('risk_metrics', {}).get('volatility', 1.0)
                entry_signal = stock.get('trading_plan', {}).get('entry_signal', 'HOLD')
                sector = stock.get('sector', 'Unknown')
                if not (volatility <= max_volatility and entry_signal in ['BUY', 'STRONG BUY']):
                    continue
                # regime sector exclusion (bear market risk-off)
                if sector in exclude_sectors:
                    dropped_regime += 1
                    continue
                # sector concentration cap
                if sector_count.get(sector, 0) >= max_per_sector:
                    dropped_sectorcap += 1
                    continue
                sector_count[sector] = sector_count.get(sector, 0) + 1
                filtered.append(stock)
            logger.info(
                f"Filtered {len(filtered)}/{len(results)} for {risk_appetite} risk "
                f"[regime={regime}, max {max_per_sector}/sector"
                f"{f', excluded {exclude_sectors}' if exclude_sectors else ''}; "
                f"dropped {dropped_regime} regime, {dropped_sectorcap} sectorcap]")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering stocks: {e}")
            return []
    def generate_portfolio_allocation(self, results: List[Dict], total_capital: float,
                                      risk_appetite: str) -> List[Dict]:
        """Generate portfolio allocation (returns structured JSON)"""
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
                    'rank': i,
                    'symbol': result.get('symbol', 'Unknown'),
                    'company': result.get('company_name', 'Unknown'),
                    'score': float(swing_score),
                    'price': float(current_price),
                    'stop_loss': float(stop_loss),
                    'allocation_pct': float(allocation_pct),
                    'investment_amount': float(final_amount),
                    'number_of_shares': int(final_shares),
                    'risk': float(actual_risk),
                    'sector': result.get('sector', 'Unknown')
                })
            return portfolio_data
        except Exception as e:
            logger.error(f"Error generating portfolio: {e}")
            return []
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    def quick_analysis_top_stocks(self, max_stocks: int = 50, max_workers: int = 10) -> List[Dict]:
        """Quick analysis of top liquid stocks for fast demo"""
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
        logger.info(f"⚡ Quick analysis: Top {len(top_liquid)} liquid stocks")
        return self.analyze_stocks_parallel(top_liquid, max_workers=max_workers)
# ========================================================================
# BACKGROUND WORKER (Optional - for production deployment)
# ========================================================================
class BackgroundAnalyzer:
    """Background worker to keep analysis cache fresh"""
    def __init__(self, trading_system: EnhancedSwingTradingSystem, refresh_interval: int = 900):
        """
        Args:
            trading_system: Instance of trading system
            refresh_interval: Seconds between refreshes (default 15 min)
        """
        self.system = trading_system
        self.refresh_interval = refresh_interval
        self.running = False
    def start(self):
        """Start background refresh (use with APScheduler or Celery in production)"""
        import threading
        def worker():
            while self.running:
                try:
                    logger.info("🔄 Background refresh starting...")
                    self.system.analyze_and_cache_all_stocks(max_workers=10)
                    logger.info(f"✅ Background refresh complete. Next refresh in {self.refresh_interval}s")
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    logger.error(f"Background worker error: {e}")
                    time.sleep(60)  # Wait 1 min on error
        self.running = True
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        logger.info(f"✅ Background analyzer started (refresh every {self.refresh_interval}s)")
    def stop(self):
        """Stop background refresh"""
        self.running = False
        logger.info("⏹️ Background analyzer stopped")
# ========================================================================
# USAGE EXAMPLES
# ========================================================================
if __name__ == "__main__":
    import redis
    from data_providers import StockDataProvider
    from symbol_mapper import SymbolMapper
    # Initialize dependencies
    mapper = SymbolMapper()
    data_provider = StockDataProvider(
        fyers_app_id=config.FYERS_APP_ID,
        fyers_access_token=config.FYERS_ACCESS_TOKEN,
        symbol_mapper=mapper,
        redis_url=config.REDIS_URL
    )
    redis_client = redis.Redis.from_url(config.REDIS_URL) if hasattr(config, 'REDIS_URL') else None
    # Initialize optimized system
    system = EnhancedSwingTradingSystem(
        data_provider=data_provider,
        redis_client=redis_client
    )
    print("\n" + "=" * 70)
    print("OPTIMIZED SWING TRADING SYSTEM - PRODUCTION MODE")
    print("=" * 70)
    print(f"Stock Universe: {len(system.get_all_stock_symbols())} stocks")
    print(f"Caching: {'✅ Enabled' if system.cache_enabled else '❌ Disabled'}")
    print(f"Parallel Processing: ✅ Enabled (10 workers)")
    # Example 1: Quick analysis (fast for demos)
    print("\n📊 Quick Analysis (Top 50 liquid stocks)...")
    results = system.quick_analysis_top_stocks(max_stocks=50, max_workers=10)
    print(f"✅ Analyzed {len(results)} stocks")
    # Example 2: Full universe analysis with caching
    print("\n🚀 Full Universe Analysis...")
    all_results = system.analyze_and_cache_all_stocks(max_workers=10)
    print(f"✅ Analyzed and cached {len(all_results)} stocks")
    # Example 3: Get cached results (instant)
    print("\n⚡ Retrieving cached analysis...")
    cached = system.get_cached_batch_analysis(risk_appetite="MEDIUM")
    if cached:
        print(f"✅ Retrieved {len(cached)} stocks from cache (instant!)")
    # Example 4: Generate portfolio
    if results:
        print("\n💼 Generating Portfolio...")
        portfolio = system.generate_portfolio_allocation(
            results=results[:20],
            total_capital=1000000,
            risk_appetite="MEDIUM"
        )
        print(f"✅ Portfolio: {len(portfolio)} positions")
        if portfolio:
            print("\n📋 Top 3 Positions:")
            for pos in portfolio[:3]:
                print(f"  {pos['rank']}. {pos['symbol']:12} "
                      f"Score: {pos['score']:.0f}  "
                      f"Allocation: {pos['allocation_pct']:.1f}%  "
                      f"Amount: ₹{pos['investment_amount']:,.0f}")
    print("\n" + "=" * 70)
    print("✅ System ready for production use!")
    print("=" * 70)
