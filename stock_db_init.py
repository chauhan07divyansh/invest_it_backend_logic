"""
stock_db_init.py
================
One-time setup: creates the SQLite database and seeds all stocks.
Run this once before starting the application.

    python stock_db_init.py

Tables created:
  - sectors          : sector master with dividend yield defaults
  - stocks           : all stock metadata (symbol, name, sector, market_cap, index_membership)
  - stock_aliases    : alternate ticker spellings (e.g. BAJAJ-AUTO → BAJAJ_AUTO)
"""

import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("stocks.db")

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sectors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    default_div_yield REAL  NOT NULL DEFAULT 0.010,
    sentiment_bias  REAL    NOT NULL DEFAULT 1.00,
    sector_score    INTEGER NOT NULL DEFAULT 60,
    preference      TEXT    NOT NULL DEFAULT 'Low'   -- 'High' | 'Medium' | 'Low'
);

CREATE TABLE IF NOT EXISTS stocks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL UNIQUE,          -- NSE ticker, e.g. RELIANCE
    name            TEXT    NOT NULL,
    sector_id       INTEGER NOT NULL REFERENCES sectors(id),
    market_cap      TEXT    NOT NULL DEFAULT 'Large', -- 'Large' | 'Mid' | 'Small'
    index_name      TEXT    NOT NULL DEFAULT 'Other', -- 'Nifty50' | 'NiftyNext50' | 'NiftyMidcap100' | 'Smallcap'
    is_active       INTEGER NOT NULL DEFAULT 1,       -- 0 = delisted / disabled
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stock_aliases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alias       TEXT    NOT NULL UNIQUE,
    symbol      TEXT    NOT NULL REFERENCES stocks(symbol)
);

-- Fast lookup indexes
CREATE INDEX IF NOT EXISTS idx_stocks_sector   ON stocks(sector_id);
CREATE INDEX IF NOT EXISTS idx_stocks_cap      ON stocks(market_cap);
CREATE INDEX IF NOT EXISTS idx_stocks_index    ON stocks(index_name);
CREATE INDEX IF NOT EXISTS idx_stocks_active   ON stocks(is_active);
"""

# ─────────────────────────────────────────────────────────────────────────────
# SEED DATA
# ─────────────────────────────────────────────────────────────────────────────

SECTORS = [
    # (name, default_div_yield, sentiment_bias, sector_score, preference)
    ("Oil & Gas",               0.045, 0.95, 55, "Medium"),
    ("Power",                   0.040, 0.90, 80, "High"),
    ("Utilities",               0.040, 0.90, 80, "High"),
    ("Mining",                  0.050, 1.00, 55, "Medium"),
    ("Banking",                 0.010, 1.10, 65, "High"),
    ("Financial Services",      0.008, 1.10, 65, "High"),
    ("Insurance",               0.010, 1.00, 60, "Low"),
    ("Information Technology",  0.020, 1.15, 70, "High"),
    ("Pharmaceuticals",         0.008, 1.15, 75, "High"),
    ("Healthcare",              0.006, 1.10, 75, "High"),
    ("Consumer Goods",          0.012, 1.05, 75, "High"),
    ("FMCG",                    0.012, 1.05, 75, "High"),
    ("Automobile",              0.015, 1.00, 60, "Low"),
    ("Steel",                   0.020, 1.00, 60, "Low"),
    ("Metals",                  0.015, 0.90, 60, "Low"),
    ("Cement",                  0.008, 1.00, 60, "Low"),
    ("Construction",            0.015, 1.00, 60, "Low"),
    ("Infrastructure",          0.012, 1.00, 55, "Low"),
    ("Real Estate",             0.010, 1.00, 55, "Low"),
    ("Telecommunications",      0.008, 1.00, 60, "Medium"),
    ("Media",                   0.005, 1.00, 60, "Low"),
    ("Textiles",                0.008, 1.00, 60, "Low"),
    ("Chemicals",               0.010, 1.00, 60, "Medium"),
    ("Fertilizers",             0.015, 1.00, 60, "Low"),
    ("Retail",                  0.003, 1.00, 60, "Low"),
    ("Conglomerate",            0.005, 1.00, 60, "Low"),
    ("Capital Goods",           0.012, 1.00, 60, "Low"),
    ("Electricals",             0.010, 1.00, 60, "Low"),
    ("Defence",                 0.008, 1.00, 65, "Low"),
    ("Hotels",                  0.005, 1.00, 60, "Low"),
    ("Consumer Durables",       0.008, 1.00, 60, "Low"),
    ("Footwear",                0.010, 1.00, 60, "Low"),
    ("Logistics",               0.008, 1.00, 60, "Low"),
    ("Technology",              0.020, 1.15, 70, "High"),
    ("Internet",                0.002, 1.00, 60, "Low"),
    ("Electronics",             0.005, 1.00, 60, "Low"),
    ("Building Materials",      0.010, 1.00, 60, "Low"),
    ("Beverages",               0.008, 1.00, 60, "Low"),
    ("Consumer Services",       0.003, 1.00, 60, "Low"),
    ("Gaming",                  0.001, 1.00, 60, "Low"),
    ("Default",                 0.010, 1.00, 60, "Low"),
]

# (symbol, name, sector_name, market_cap, index_name)
STOCKS = [
    # ── NIFTY 50 ─────────────────────────────────────────────────────────────
    ("RELIANCE",    "Reliance Industries",              "Oil & Gas",               "Large", "Nifty50"),
    ("TCS",         "Tata Consultancy Services",        "Information Technology",  "Large", "Nifty50"),
    ("HDFCBANK",    "HDFC Bank",                        "Banking",                 "Large", "Nifty50"),
    ("INFY",        "Infosys",                          "Information Technology",  "Large", "Nifty50"),
    ("HINDUNILVR",  "Hindustan Unilever",               "Consumer Goods",          "Large", "Nifty50"),
    ("ICICIBANK",   "ICICI Bank",                       "Banking",                 "Large", "Nifty50"),
    ("KOTAKBANK",   "Kotak Mahindra Bank",              "Banking",                 "Large", "Nifty50"),
    ("BAJFINANCE",  "Bajaj Finance",                    "Financial Services",      "Large", "Nifty50"),
    ("LT",          "Larsen & Toubro",                  "Construction",            "Large", "Nifty50"),
    ("SBIN",        "State Bank of India",              "Banking",                 "Large", "Nifty50"),
    ("BHARTIARTL",  "Bharti Airtel",                    "Telecommunications",      "Large", "Nifty50"),
    ("ASIANPAINT",  "Asian Paints",                     "Consumer Goods",          "Large", "Nifty50"),
    ("MARUTI",      "Maruti Suzuki",                    "Automobile",              "Large", "Nifty50"),
    ("TITAN",       "Titan Company",                    "Consumer Goods",          "Large", "Nifty50"),
    ("SUNPHARMA",   "Sun Pharmaceutical",               "Pharmaceuticals",         "Large", "Nifty50"),
    ("ULTRACEMCO",  "UltraTech Cement",                 "Cement",                  "Large", "Nifty50"),
    ("NESTLEIND",   "Nestle India",                     "Consumer Goods",          "Large", "Nifty50"),
    ("HCLTECH",     "HCL Technologies",                 "Information Technology",  "Large", "Nifty50"),
    ("AXISBANK",    "Axis Bank",                        "Banking",                 "Large", "Nifty50"),
    ("WIPRO",       "Wipro",                            "Information Technology",  "Large", "Nifty50"),
    ("NTPC",        "NTPC",                             "Power",                   "Large", "Nifty50"),
    ("POWERGRID",   "Power Grid Corporation",           "Power",                   "Large", "Nifty50"),
    ("ONGC",        "Oil & Natural Gas Corporation",    "Oil & Gas",               "Large", "Nifty50"),
    ("TECHM",       "Tech Mahindra",                    "Information Technology",  "Large", "Nifty50"),
    ("TATASTEEL",   "Tata Steel",                       "Steel",                   "Large", "Nifty50"),
    ("ADANIENT",    "Adani Enterprises",                "Conglomerate",            "Large", "Nifty50"),
    ("COALINDIA",   "Coal India",                       "Mining",                  "Large", "Nifty50"),
    ("HINDALCO",    "Hindalco Industries",              "Metals",                  "Large", "Nifty50"),
    ("JSWSTEEL",    "JSW Steel",                        "Steel",                   "Large", "Nifty50"),
    ("BAJAJAUTO",   "Bajaj Auto",                       "Automobile",              "Large", "Nifty50"),
    ("MM",          "Mahindra & Mahindra",              "Automobile",              "Large", "Nifty50"),
    ("HEROMOTOCO",  "Hero MotoCorp",                    "Automobile",              "Large", "Nifty50"),
    ("GRASIM",      "Grasim Industries",                "Cement",                  "Large", "Nifty50"),
    ("SHREECEM",    "Shree Cement",                     "Cement",                  "Large", "Nifty50"),
    ("EICHERMOT",   "Eicher Motors",                    "Automobile",              "Large", "Nifty50"),
    ("UPL",         "UPL Limited",                      "Chemicals",               "Large", "Nifty50"),
    ("BPCL",        "Bharat Petroleum",                 "Oil & Gas",               "Large", "Nifty50"),
    ("DIVISLAB",    "Divi's Laboratories",              "Pharmaceuticals",         "Large", "Nifty50"),
    ("DRREDDY",     "Dr. Reddy's Laboratories",         "Pharmaceuticals",         "Large", "Nifty50"),
    ("CIPLA",       "Cipla",                            "Pharmaceuticals",         "Large", "Nifty50"),
    ("BRITANNIA",   "Britannia Industries",             "Consumer Goods",          "Large", "Nifty50"),
    ("TATACONSUM",  "Tata Consumer Products",           "Consumer Goods",          "Large", "Nifty50"),
    ("IOC",         "Indian Oil Corporation",           "Oil & Gas",               "Large", "Nifty50"),
    ("APOLLOHOSP",  "Apollo Hospitals",                 "Healthcare",              "Large", "Nifty50"),
    ("BAJAJFINSV",  "Bajaj Finserv",                    "Financial Services",      "Large", "Nifty50"),
    ("HDFCLIFE",    "HDFC Life Insurance",              "Insurance",               "Large", "Nifty50"),
    ("SBILIFE",     "SBI Life Insurance",               "Insurance",               "Large", "Nifty50"),
    ("INDUSINDBK",  "IndusInd Bank",                    "Banking",                 "Large", "Nifty50"),
    ("ADANIPORTS",  "Adani Ports",                      "Infrastructure",          "Large", "Nifty50"),
    ("TATAMOTORS",  "Tata Motors",                      "Automobile",              "Large", "Nifty50"),
    ("ITC",         "ITC Limited",                      "Consumer Goods",          "Large", "Nifty50"),

    # ── NIFTY NEXT 50 ────────────────────────────────────────────────────────
    ("SIEMENS",     "Siemens Limited",                  "Capital Goods",           "Large", "NiftyNext50"),
    ("HAVELLS",     "Havells India",                    "Electricals",             "Large", "NiftyNext50"),
    ("DLF",         "DLF Limited",                      "Real Estate",             "Large", "NiftyNext50"),
    ("GODREJCP",    "Godrej Consumer Products",         "Consumer Goods",          "Large", "NiftyNext50"),
    ("COLPAL",      "Colgate-Palmolive India",          "Consumer Goods",          "Large", "NiftyNext50"),
    ("PIDILITIND",  "Pidilite Industries",              "Chemicals",               "Large", "NiftyNext50"),
    ("MARICO",      "Marico Limited",                   "Consumer Goods",          "Large", "NiftyNext50"),
    ("DABUR",       "Dabur India",                      "Consumer Goods",          "Large", "NiftyNext50"),
    ("LUPIN",       "Lupin Limited",                    "Pharmaceuticals",         "Large", "NiftyNext50"),
    ("BIOCON",      "Biocon Limited",                   "Pharmaceuticals",         "Large", "NiftyNext50"),
    ("MOTHERSUMI",  "Motherson Sumi Systems",           "Automobile",              "Large", "NiftyNext50"),
    ("BOSCHLTD",    "Bosch Limited",                    "Automobile",              "Large", "NiftyNext50"),
    ("EXIDEIND",    "Exide Industries",                 "Automobile",              "Large", "NiftyNext50"),
    ("ASHOKLEY",    "Ashok Leyland",                    "Automobile",              "Large", "NiftyNext50"),
    ("TVSMOTOR",    "TVS Motor Company",                "Automobile",              "Large", "NiftyNext50"),
    ("BALKRISIND",  "Balkrishna Industries",            "Automobile",              "Large", "NiftyNext50"),
    ("MRF",         "MRF Limited",                      "Automobile",              "Large", "NiftyNext50"),
    ("APOLLOTYRE",  "Apollo Tyres",                     "Automobile",              "Large", "NiftyNext50"),
    ("BHARATFORG",  "Bharat Forge",                     "Automobile",              "Large", "NiftyNext50"),
    ("CUMMINSIND",  "Cummins India",                    "Automobile",              "Large", "NiftyNext50"),
    ("FEDERALBNK",  "Federal Bank",                     "Banking",                 "Large", "NiftyNext50"),
    ("BANDHANBNK",  "Bandhan Bank",                     "Banking",                 "Large", "NiftyNext50"),
    ("IDFCFIRSTB",  "IDFC First Bank",                  "Banking",                 "Large", "NiftyNext50"),
    ("PNB",         "Punjab National Bank",             "Banking",                 "Large", "NiftyNext50"),
    ("BANKBARODA",  "Bank of Baroda",                   "Banking",                 "Large", "NiftyNext50"),
    ("CANBK",       "Canara Bank",                      "Banking",                 "Large", "NiftyNext50"),
    ("UNIONBANK",   "Union Bank of India",              "Banking",                 "Large", "NiftyNext50"),
    ("CHOLAFIN",    "Cholamandalam Investment",         "Financial Services",      "Large", "NiftyNext50"),
    ("LICHSGFIN",   "LIC Housing Finance",              "Financial Services",      "Large", "NiftyNext50"),
    ("SRTRANSFIN",  "Shriram Transport Finance",        "Financial Services",      "Large", "NiftyNext50"),
    ("LTTS",        "L&T Technology Services",          "Information Technology",  "Large", "NiftyNext50"),
    ("PERSISTENT",  "Persistent Systems",               "Information Technology",  "Large", "NiftyNext50"),
    ("COFORGE",     "Coforge Limited",                  "Information Technology",  "Large", "NiftyNext50"),
    ("MPHASIS",     "Mphasis Limited",                  "Information Technology",  "Large", "NiftyNext50"),
    ("DMART",       "Avenue Supermarts",                "Retail",                  "Large", "NiftyNext50"),
    ("TRENT",       "Trent Limited",                    "Retail",                  "Large", "NiftyNext50"),
    ("PAGEIND",     "Page Industries",                  "Textiles",                "Large", "NiftyNext50"),
    ("RAYMOND",     "Raymond Limited",                  "Textiles",                "Large", "NiftyNext50"),
    ("BERGEPAINT",  "Berger Paints",                    "Consumer Goods",          "Large", "NiftyNext50"),
    ("VOLTAS",      "Voltas Limited",                   "Consumer Durables",       "Large", "NiftyNext50"),
    ("WHIRLPOOL",   "Whirlpool of India",               "Consumer Durables",       "Large", "NiftyNext50"),
    ("CROMPTON",    "Crompton Greaves",                 "Electricals",             "Large", "NiftyNext50"),
    ("TORNTPHARM",  "Torrent Pharmaceuticals",          "Pharmaceuticals",         "Large", "NiftyNext50"),
    ("AUROPHARMA",  "Aurobindo Pharma",                 "Pharmaceuticals",         "Large", "NiftyNext50"),
    ("ALKEM",       "Alkem Laboratories",               "Pharmaceuticals",         "Large", "NiftyNext50"),
    ("JUBLFOOD",    "Jubilant FoodWorks",               "Consumer Services",       "Large", "NiftyNext50"),
    ("VBL",         "Varun Beverages",                  "Beverages",               "Large", "NiftyNext50"),
    ("EMAMILTD",    "Emami Limited",                    "Consumer Goods",          "Large", "NiftyNext50"),
    ("GODREJPROP",  "Godrej Properties",                "Real Estate",             "Large", "NiftyNext50"),
    ("OBEROIRLTY",  "Oberoi Realty",                    "Real Estate",             "Large", "NiftyNext50"),

    # ── NIFTY MIDCAP 100 ─────────────────────────────────────────────────────
    ("ABCAPITAL",   "Aditya Birla Capital",             "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("ABFRL",       "Aditya Birla Fashion",             "Retail",                  "Mid",   "NiftyMidcap100"),
    ("ACC",         "ACC Limited",                      "Cement",                  "Mid",   "NiftyMidcap100"),
    ("ADANIGREEN",  "Adani Green Energy",               "Power",                   "Mid",   "NiftyMidcap100"),
    ("ADANIPOWER",  "Adani Power",                      "Power",                   "Mid",   "NiftyMidcap100"),
    ("AFFLE",       "Affle India",                      "Technology",              "Mid",   "NiftyMidcap100"),
    ("AIAENG",      "AIA Engineering",                  "Capital Goods",           "Mid",   "NiftyMidcap100"),
    ("AJANTPHARM",  "Ajanta Pharma",                    "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),
    ("AKUMS",       "Akums Drugs",                      "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),
    ("AMBER",       "Amber Enterprises",                "Consumer Durables",       "Mid",   "NiftyMidcap100"),
    ("AMBUJACEM",   "Ambuja Cements",                   "Cement",                  "Mid",   "NiftyMidcap100"),
    ("ASTRAL",      "Astral Limited",                   "Building Materials",      "Mid",   "NiftyMidcap100"),
    ("ATUL",        "Atul Limited",                     "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("AUBANK",      "AU Small Finance Bank",            "Banking",                 "Mid",   "NiftyMidcap100"),
    ("BAJAJELEC",   "Bajaj Electricals",                "Electricals",             "Mid",   "NiftyMidcap100"),
    ("BALAMINES",   "Balaji Amines",                    "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("BATAINDIA",   "Bata India",                       "Footwear",                "Mid",   "NiftyMidcap100"),
    ("BEL",         "Bharat Electronics",               "Defence",                 "Mid",   "NiftyMidcap100"),
    ("BHEL",        "Bharat Heavy Electricals",         "Capital Goods",           "Mid",   "NiftyMidcap100"),
    ("BRIGADE",     "Brigade Enterprises",              "Real Estate",             "Mid",   "NiftyMidcap100"),
    ("CESC",        "CESC Limited",                     "Power",                   "Mid",   "NiftyMidcap100"),
    ("CHAMBLFERT",  "Chambal Fertilizers",              "Fertilizers",             "Mid",   "NiftyMidcap100"),
    ("CONCOR",      "Container Corporation",            "Logistics",               "Mid",   "NiftyMidcap100"),
    ("COROMANDEL",  "Coromandel International",         "Fertilizers",             "Mid",   "NiftyMidcap100"),
    ("CRISIL",      "CRISIL Limited",                   "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("CUB",         "City Union Bank",                  "Banking",                 "Mid",   "NiftyMidcap100"),
    ("CYIENT",      "Cyient Limited",                   "Information Technology",  "Mid",   "NiftyMidcap100"),
    ("DEEPAKNTR",   "Deepak Nitrite",                   "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("DIXON",       "Dixon Technologies",               "Electronics",             "Mid",   "NiftyMidcap100"),
    ("ESCORTS",     "Escorts Kubota",                   "Automobile",              "Mid",   "NiftyMidcap100"),
    ("FACT",        "Fertilizers And Chemicals",        "Fertilizers",             "Mid",   "NiftyMidcap100"),
    ("GAIL",        "GAIL India",                       "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("GLENMARK",    "Glenmark Pharmaceuticals",         "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),
    ("GRANULES",    "Granules India",                   "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),
    ("GRAPHITE",    "Graphite India",                   "Capital Goods",           "Mid",   "NiftyMidcap100"),
    ("GUJGASLTD",   "Gujarat Gas",                      "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("HFCL",        "HFCL Limited",                     "Telecommunications",      "Mid",   "NiftyMidcap100"),
    ("HINDCOPPER",  "Hindustan Copper",                 "Metals",                  "Mid",   "NiftyMidcap100"),
    ("HINDPETRO",   "Hindustan Petroleum",              "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("HONAUT",      "Honeywell Automation",             "Capital Goods",           "Mid",   "NiftyMidcap100"),
    ("IEX",         "Indian Energy Exchange",           "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("IGL",         "Indraprastha Gas",                 "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("INDHOTEL",    "Indian Hotels",                    "Hotels",                  "Mid",   "NiftyMidcap100"),
    ("INDUSTOWER",  "Indus Towers",                     "Telecommunications",      "Mid",   "NiftyMidcap100"),
    ("INTELLECT",   "Intellect Design Arena",           "Technology",              "Mid",   "NiftyMidcap100"),
    ("IRCTC",       "Indian Railway Catering",          "Consumer Services",       "Mid",   "NiftyMidcap100"),
    ("ISEC",        "ICICI Securities",                 "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("JINDALSTEL",  "Jindal Steel & Power",             "Steel",                   "Mid",   "NiftyMidcap100"),
    ("JKCEMENT",    "JK Cement",                        "Cement",                  "Mid",   "NiftyMidcap100"),
    ("JSWENERGY",   "JSW Energy",                       "Power",                   "Mid",   "NiftyMidcap100"),
    ("KAJARIACER",  "Kajaria Ceramics",                 "Building Materials",      "Mid",   "NiftyMidcap100"),
    ("KEI",         "KEI Industries",                   "Electricals",             "Mid",   "NiftyMidcap100"),
    ("LTFH",        "L&T Finance Holdings",             "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("LALPATHLAB",  "Dr Lal PathLabs",                  "Healthcare",              "Mid",   "NiftyMidcap100"),
    ("LAURUSLABS",  "Laurus Labs",                      "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),
    ("MANAPPURAM",  "Manappuram Finance",               "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("MCX",         "Multi Commodity Exchange",         "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("METROBRAND",  "Metro Brands",                     "Footwear",                "Mid",   "NiftyMidcap100"),
    ("MFSL",        "Max Financial Services",           "Insurance",               "Mid",   "NiftyMidcap100"),
    ("MGL",         "Mahanagar Gas",                    "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("MINDTREE",    "Mindtree Limited",                 "Information Technology",  "Mid",   "NiftyMidcap100"),
    ("MOTHERSON",   "Samvardhana Motherson",            "Automobile",              "Mid",   "NiftyMidcap100"),
    ("MUTHOOTFIN",  "Muthoot Finance",                  "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("NATIONALUM",  "National Aluminium",               "Metals",                  "Mid",   "NiftyMidcap100"),
    ("NAUKRI",      "Info Edge India",                  "Internet",                "Mid",   "NiftyMidcap100"),
    ("NAVINFLUOR",  "Navin Fluorine",                   "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("NMDC",        "NMDC Limited",                     "Mining",                  "Mid",   "NiftyMidcap100"),
    ("OIL",         "Oil India",                        "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("PAYTM",       "One 97 Communications",            "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("PEL",         "Piramal Enterprises",              "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("PETRONET",    "Petronet LNG",                     "Oil & Gas",               "Mid",   "NiftyMidcap100"),
    ("PFC",         "Power Finance Corporation",        "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("PHOENIXLTD",  "Phoenix Mills",                    "Real Estate",             "Mid",   "NiftyMidcap100"),
    ("PIIND",       "PI Industries",                    "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("POLYCAB",     "Polycab India",                    "Electricals",             "Mid",   "NiftyMidcap100"),
    ("PRESTIGE",    "Prestige Estates",                 "Real Estate",             "Mid",   "NiftyMidcap100"),
    ("RECLTD",      "REC Limited",                      "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("SBICARD",     "SBI Cards",                        "Financial Services",      "Mid",   "NiftyMidcap100"),
    ("SOLARINDS",   "Solar Industries",                 "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("SONACOMS",    "Sona BLW Precision",               "Automobile",              "Mid",   "NiftyMidcap100"),
    ("SRF",         "SRF Limited",                      "Chemicals",               "Mid",   "NiftyMidcap100"),
    ("STAR",        "Sterlite Technologies",            "Telecommunications",      "Mid",   "NiftyMidcap100"),
    ("TATACOMM",    "Tata Communications",              "Telecommunications",      "Mid",   "NiftyMidcap100"),
    ("TATAELXSI",   "Tata Elxsi",                       "Information Technology",  "Mid",   "NiftyMidcap100"),
    ("TATAPOWER",   "Tata Power",                       "Power",                   "Mid",   "NiftyMidcap100"),
    ("THERMAX",     "Thermax Limited",                  "Capital Goods",           "Mid",   "NiftyMidcap100"),
    ("TORNTPOWER",  "Torrent Power",                    "Power",                   "Mid",   "NiftyMidcap100"),
    ("UBL",         "United Breweries",                 "Beverages",               "Mid",   "NiftyMidcap100"),
    ("VEDL",        "Vedanta Limited",                  "Metals",                  "Mid",   "NiftyMidcap100"),
    ("ZOMATO",      "Zomato Limited",                   "Consumer Services",       "Mid",   "NiftyMidcap100"),
    ("ZYDUSLIFE",   "Zydus Lifesciences",               "Pharmaceuticals",         "Mid",   "NiftyMidcap100"),

    # ── SMALLCAP ─────────────────────────────────────────────────────────────
    ("AAVAS",       "Aavas Financiers",                 "Financial Services",      "Small", "Smallcap"),
    ("ANANDRATHI",  "Anand Rathi Wealth",               "Financial Services",      "Small", "Smallcap"),
    ("ANGELONE",    "Angel One",                        "Financial Services",      "Small", "Smallcap"),
    ("ASIANHOTNR",  "Asian Hotels (North)",             "Hotels",                  "Small", "Smallcap"),
    ("BASF",        "BASF India",                       "Chemicals",               "Small", "Smallcap"),
    ("BLUESTARCO",  "Blue Star",                        "Consumer Durables",       "Small", "Smallcap"),
    ("CAMS",        "CAMS",                             "Financial Services",      "Small", "Smallcap"),
    ("CDSL",        "Central Depository Services",      "Financial Services",      "Small", "Smallcap"),
    ("CENTRALBK",   "Central Bank of India",            "Banking",                 "Small", "Smallcap"),
    ("CENTURYPLY",  "Century Plyboards",                "Building Materials",      "Small", "Smallcap"),
    ("CLEAN",       "Clean Science",                    "Chemicals",               "Small", "Smallcap"),
    ("CREDITACC",   "CreditAccess Grameen",             "Financial Services",      "Small", "Smallcap"),
    ("CSBBANK",     "CSB Bank",                         "Banking",                 "Small", "Smallcap"),
    ("DELTACORP",   "Delta Corp",                       "Gaming",                  "Small", "Smallcap"),
    ("DEVYANI",     "Devyani International",            "Consumer Services",       "Small", "Smallcap"),
    ("EQUITAS",     "Equitas Small Finance Bank",       "Banking",                 "Small", "Smallcap"),
    ("FINPIPE",     "Fine Organic Industries",          "Chemicals",               "Small", "Smallcap"),
    ("FLUOROCHEM",  "Gujarat Fluorochemicals",          "Chemicals",               "Small", "Smallcap"),
    ("GRINDWELL",   "Grindwell Norton",                 "Capital Goods",           "Small", "Smallcap"),
    ("HAPPSTMNDS",  "Happiest Minds",                   "Information Technology",  "Small", "Smallcap"),
    ("HEMHINDUS",   "HEG Limited",                      "Capital Goods",           "Small", "Smallcap"),
    ("IIFLWAM",     "IIFL Wealth Management",           "Financial Services",      "Small", "Smallcap"),
    ("INDIAMART",   "IndiaMART InterMESH",              "Internet",                "Small", "Smallcap"),
    ("INDIANB",     "Indian Bank",                      "Banking",                 "Small", "Smallcap"),
    ("JUBLPHARMA",  "Jubilant Pharmova",                "Pharmaceuticals",         "Small", "Smallcap"),
    ("JUSTDIAL",    "Just Dial",                        "Internet",                "Small", "Smallcap"),
    ("KPITTECH",    "KPIT Technologies",                "Information Technology",  "Small", "Smallcap"),
    ("LATENTVIEW",  "Latent View Analytics",            "Information Technology",  "Small", "Smallcap"),
    ("LEMONTREE",   "Lemon Tree Hotels",                "Hotels",                  "Small", "Smallcap"),
    ("MAZDOCK",     "Mazagon Dock Shipbuilders",        "Defence",                 "Small", "Smallcap"),
    ("METROPOLIS",  "Metropolis Healthcare",            "Healthcare",              "Small", "Smallcap"),
    ("MIDHANI",     "Mishra Dhatu Nigam",               "Defence",                 "Small", "Smallcap"),
    ("NAZARA",      "Nazara Technologies",              "Gaming",                  "Small", "Smallcap"),
    ("NIACL",       "New India Assurance",              "Insurance",               "Small", "Smallcap"),
    ("NYKAA",       "FSN E-Commerce (Nykaa)",           "Retail",                  "Small", "Smallcap"),
    ("ORIENTELEC",  "Orient Electric",                  "Electricals",             "Small", "Smallcap"),
    ("PARAS",       "Paras Defence",                    "Defence",                 "Small", "Smallcap"),
    ("PNBHOUSING",  "PNB Housing Finance",              "Financial Services",      "Small", "Smallcap"),
    ("POLICYBZR",   "PB Fintech",                       "Financial Services",      "Small", "Smallcap"),
    ("POONAWALLA",  "Poonawalla Fincorp",               "Financial Services",      "Small", "Smallcap"),
    ("RAILTEL",     "RailTel Corporation",              "Telecommunications",      "Small", "Smallcap"),
    ("RATNAMANI",   "Ratnamani Metals",                 "Metals",                  "Small", "Smallcap"),
    ("ROUTE",       "Route Mobile",                     "Telecommunications",      "Small", "Smallcap"),
    ("SAFARI",      "Safari Industries",                "Consumer Goods",          "Small", "Smallcap"),
    ("SHYAMMETL",   "Shyam Metalics",                   "Metals",                  "Small", "Smallcap"),
    ("SIGNATURE",   "Signature Global",                 "Real Estate",             "Small", "Smallcap"),
    ("SYNGENE",     "Syngene International",            "Pharmaceuticals",         "Small", "Smallcap"),
    ("TANLA",       "Tanla Platforms",                  "Telecommunications",      "Small", "Smallcap"),
    ("UCOBANK",     "UCO Bank",                         "Banking",                 "Small", "Smallcap"),
    ("UJJIVAN",     "Ujjivan Small Finance Bank",       "Banking",                 "Small", "Smallcap"),
    ("UTIAMC",      "UTI Asset Management",             "Financial Services",      "Small", "Smallcap"),
]

# Aliases for tickers with special characters (original → clean DB symbol)
ALIASES = [
    ("BAJAJ-AUTO",  "BAJAJAUTO"),
    ("M&M",         "MM"),
    ("L&TFH",       "LTFH"),
    ("BAJAJ_AUTO",  "BAJAJAUTO"),
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH, force_reset: bool = False):
    if force_reset and db_path.exists():
        db_path.unlink()
        logger.info("Existing DB deleted (force_reset=True)")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create schema
    cur.executescript(SCHEMA)

    # Seed sectors
    cur.executemany(
        """INSERT OR IGNORE INTO sectors
           (name, default_div_yield, sentiment_bias, sector_score, preference)
           VALUES (?, ?, ?, ?, ?)""",
        SECTORS
    )

    # Build sector name → id map
    cur.execute("SELECT id, name FROM sectors")
    sector_map = {row["name"]: row["id"] for row in cur.fetchall()}

    # Seed stocks
    stock_rows = []
    for symbol, name, sector_name, market_cap, index_name in STOCKS:
        sector_id = sector_map.get(sector_name, sector_map["Default"])
        stock_rows.append((symbol, name, sector_id, market_cap, index_name))

    cur.executemany(
        """INSERT OR IGNORE INTO stocks
           (symbol, name, sector_id, market_cap, index_name)
           VALUES (?, ?, ?, ?, ?)""",
        stock_rows
    )

    # Seed aliases
    cur.executemany(
        "INSERT OR IGNORE INTO stock_aliases (alias, symbol) VALUES (?, ?)",
        ALIASES
    )

    conn.commit()

    # Stats
    cur.execute("SELECT COUNT(*) FROM stocks")
    total = cur.fetchone()[0]
    cur.execute("SELECT market_cap, COUNT(*) FROM stocks GROUP BY market_cap")
    caps = {row[0]: row[1] for row in cur.fetchall()}

    conn.close()
    logger.info("✅ DB initialised: %s", db_path.resolve())
    logger.info("   Total stocks : %d", total)
    logger.info("   Large cap    : %d", caps.get("Large", 0))
    logger.info("   Mid cap      : %d", caps.get("Mid", 0))
    logger.info("   Small cap    : %d", caps.get("Small", 0))
    return db_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the DB")
    parser.add_argument("--db",    default=str(DB_PATH),  help="Path to SQLite file")
    args = parser.parse_args()
    init_db(Path(args.db), force_reset=args.reset)
