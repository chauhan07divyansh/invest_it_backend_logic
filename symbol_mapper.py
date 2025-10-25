"""
Symbol Mapping Module - EXPANDED VERSION
Maps NSE/BSE symbols to Screener.in company slugs + BSE codes + IR URLs
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# COMPREHENSIVE Symbol Mapping
# Format: 'SYMBOL': {
#     'name': 'Company Name',
#     'screener_slug': 'company-slug-on-screener-in',
#     'sector': 'Sector Name',
#     'bse_code': 'BSE Script Code',
#     'ir_url': 'Investor Relations URL'
# }

SYMBOL_MAP = {
    # NIFTY 50 - Large Cap
    'RELIANCE': {
        'name': 'Reliance Industries',
        'screener_slug': 'reliance-industries',
        'sector': 'Oil & Gas',
        'bse_code': '500325',
        'ir_url': 'https://www.ril.com/InvestorRelations.aspx'
    },
    'TCS': {
        'name': 'Tata Consultancy Services',
        'screener_slug': 'tcs',
        'sector': 'Information Technology',
        'bse_code': '532540',
        'ir_url': 'https://www.tcs.com/investor-relations'
    },
    'HDFCBANK': {
        'name': 'HDFC Bank',
        'screener_slug': 'hdfc-bank',
        'sector': 'Banking',
        'bse_code': '500180',
        'ir_url': 'https://www.hdfcbank.com/personal/about-us/investor-relations'
    },
    'INFY': {
        'name': 'Infosys',
        'screener_slug': 'infosys',
        'sector': 'Information Technology',
        'bse_code': '500209',
        'ir_url': 'https://www.infosys.com/investors/reports-filings.html'
    },
    'HINDUNILVR': {
        'name': 'Hindustan Unilever',
        'screener_slug': 'hindunilvr',
        'sector': 'Consumer Goods',
        'bse_code': '500696',
        'ir_url': 'https://www.hul.co.in/investor-relations/'
    },
    'ICICIBANK': {
        'name': 'ICICI Bank',
        'screener_slug': 'icici-bank',
        'sector': 'Banking',
        'bse_code': '532174',
        'ir_url': 'https://www.icicibank.com/aboutus/annual_reports.page'
    },
    'KOTAKBANK': {
        'name': 'Kotak Mahindra Bank',
        'screener_slug': 'kotak-mahindra-bank',
        'sector': 'Banking',
        'bse_code': '500247',
        'ir_url': 'https://www.kotak.com/en/investor-relations.html'
    },
    'BAJFINANCE': {
        'name': 'Bajaj Finance',
        'screener_slug': 'bajaj-finance',
        'sector': 'Financial Services',
        'bse_code': '500034',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations'
    },
    'LT': {
        'name': 'Larsen & Toubro',
        'screener_slug': 'lt',
        'sector': 'Construction',
        'bse_code': '500510',
        'ir_url': 'https://www.larsentoubro.com/corporate/investors/'
    },
    'SBIN': {
        'name': 'State Bank of India',
        'screener_slug': 'sbi',
        'sector': 'Banking',
        'bse_code': '500112',
        'ir_url': 'https://bank.sbi/web/investor-relations'
    },
    'BHARTIARTL': {
        'name': 'Bharti Airtel',
        'screener_slug': 'bharti-airtel',
        'sector': 'Telecommunications',
        'bse_code': '532454',
        'ir_url': 'https://www.airtel.in/about-bharti/equity/investor-relations'
    },
    'ASIANPAINT': {
        'name': 'Asian Paints',
        'screener_slug': 'asian-paints',
        'sector': 'Consumer Goods',
        'bse_code': '500820',
        'ir_url': 'https://www.asianpaints.com/investors.html'
    },
    'MARUTI': {
        'name': 'Maruti Suzuki',
        'screener_slug': 'maruti-suzuki-india',
        'sector': 'Automobile',
        'bse_code': '532500',
        'ir_url': 'https://www.marutisuzuki.com/corporate/investor'
    },
    'TITAN': {
        'name': 'Titan Company',
        'screener_slug': 'titan-company',
        'sector': 'Consumer Goods',
        'bse_code': '500114',
        'ir_url': 'https://www.titancompany.in/investor-information'
    },
    'SUNPHARMA': {
        'name': 'Sun Pharmaceutical',
        'screener_slug': 'sun-pharma',
        'sector': 'Pharmaceuticals',
        'bse_code': '524715',
        'ir_url': 'https://www.sunpharma.com/investors'
    },
    'ULTRACEMCO': {
        'name': 'UltraTech Cement',
        'screener_slug': 'ultratech-cement',
        'sector': 'Cement',
        'bse_code': '532538',
        'ir_url': 'https://www.ultratechcement.com/investors'
    },
    'NESTLEIND': {
        'name': 'Nestle India',
        'screener_slug': 'nestle-india',
        'sector': 'Consumer Goods',
        'bse_code': '500790',
        'ir_url': 'https://www.nestle.in/investors'
    },
    'HCLTECH': {
        'name': 'HCL Technologies',
        'screener_slug': 'hcl-technologies',
        'sector': 'Information Technology',
        'bse_code': '532281',
        'ir_url': 'https://www.hcltech.com/investors'
    },
    'AXISBANK': {
        'name': 'Axis Bank',
        'screener_slug': 'axis-bank',
        'sector': 'Banking',
        'bse_code': '532215',
        'ir_url': 'https://www.axisbank.com/shareholders-corner'
    },
    'WIPRO': {
        'name': 'Wipro',
        'screener_slug': 'wipro',
        'sector': 'Information Technology',
        'bse_code': '507685',
        'ir_url': 'https://www.wipro.com/investors/'
    },
    'NTPC': {
        'name': 'NTPC',
        'screener_slug': 'ntpc',
        'sector': 'Power',
        'bse_code': '532555',
        'ir_url': 'https://www.ntpc.co.in/investors'
    },
    'POWERGRID': {
        'name': 'Power Grid Corporation',
        'screener_slug': 'power-grid-corporation',
        'sector': 'Power',
        'bse_code': '532898',
        'ir_url': 'https://www.powergrid.in/investor-relations'
    },
    'ONGC': {
        'name': 'Oil & Natural Gas Corporation',
        'screener_slug': 'ongc',
        'sector': 'Oil & Gas',
        'bse_code': '500312',
        'ir_url': 'https://www.ongcindia.com/wps/wcm/connect/en/investor/'
    },
    'TECHM': {
        'name': 'Tech Mahindra',
        'screener_slug': 'tech-mahindra',
        'sector': 'Information Technology',
        'bse_code': '532755',
        'ir_url': 'https://www.techmahindra.com/investors.html'
    },
    'TATASTEEL': {
        'name': 'Tata Steel',
        'screener_slug': 'tata-steel',
        'sector': 'Steel',
        'bse_code': '500470',
        'ir_url': 'https://www.tatasteel.com/investors/'
    },
    'ITC': {
        'name': 'ITC Limited',
        'screener_slug': 'itc',
        'sector': 'Consumer Goods',
        'bse_code': '500875',
        'ir_url': 'https://www.itcportal.com/about-itc/shareholder-value.aspx'
    },
    'TATAMOTORS': {
        'name': 'Tata Motors',
        'screener_slug': 'tata-motors',
        'sector': 'Automobile',
        'bse_code': '500570',
        'ir_url': 'https://www.tatamotors.com/investors/'
    },
    'ADANIPORTS': {
        'name': 'Adani Ports',
        'screener_slug': 'adani-ports-sez',
        'sector': 'Infrastructure',
        'bse_code': '532921',
        'ir_url': 'https://www.adaniports.com/Investors'
    },
    'INDUSINDBK': {
        'name': 'IndusInd Bank',
        'screener_slug': 'indusind-bank',
        'sector': 'Banking',
        'bse_code': '532187',
        'ir_url': 'https://www.indusind.com/in/en/personal/useful-links/investor-relations.html'
    },
    'CIPLA': {
        'name': 'Cipla',
        'screener_slug': 'cipla',
        'sector': 'Pharmaceuticals',
        'bse_code': '500087',
        'ir_url': 'https://www.cipla.com/investor-information'
    },
    'DRREDDY': {
        'name': "Dr. Reddy's Laboratories",
        'screener_slug': 'dr-reddys-laboratories',
        'sector': 'Pharmaceuticals',
        'bse_code': '500124',
        'ir_url': 'https://www.drreddys.com/investors/'
    },
    'DIVISLAB': {
        'name': "Divi's Laboratories",
        'screener_slug': 'divis-laboratories',
        'sector': 'Pharmaceuticals',
        'bse_code': '532488',
        'ir_url': 'https://www.divislabs.com/investors.html'
    },
    'BRITANNIA': {
        'name': 'Britannia Industries',
        'screener_slug': 'britannia-industries',
        'sector': 'Consumer Goods',
        'bse_code': '500825',
        'ir_url': 'https://www.britannia.co.in/investors'
    },
    'BPCL': {
        'name': 'Bharat Petroleum',
        'screener_slug': 'bpcl',
        'sector': 'Oil & Gas',
        'bse_code': '500547',
        'ir_url': 'https://www.bharatpetroleum.in/Investor.aspx'
    },
    'APOLLOHOSP': {
        'name': 'Apollo Hospitals',
        'screener_slug': 'apollo-hospitals-enterprise',
        'sector': 'Healthcare',
        'bse_code': '508869',
        'ir_url': 'https://www.apollohospitals.com/investor_relations/'
    },
    'BAJAJFINSV': {
        'name': 'Bajaj Finserv',
        'screener_slug': 'bajaj-finserv',
        'sector': 'Financial Services',
        'bse_code': '532978',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations'
    },
    'HDFCLIFE': {
        'name': 'HDFC Life Insurance',
        'screener_slug': 'hdfc-life-insurance',
        'sector': 'Insurance',
        'bse_code': '540777',
        'ir_url': 'https://www.hdfclife.com/investor-relations'
    },
    'SBILIFE': {
        'name': 'SBI Life Insurance',
        'screener_slug': 'sbi-life-insurance',
        'sector': 'Insurance',
        'bse_code': '540719',
        'ir_url': 'https://www.sbilife.co.in/en/investor-information'
    },

    # Mid & Small Cap Additions
    'GODREJCP': {
        'name': 'Godrej Consumer Products',
        'screener_slug': 'godrej-consumer-products',
        'sector': 'Consumer Goods',
        'bse_code': '532424',
        'ir_url': None
    },
    'COLPAL': {
        'name': 'Colgate-Palmolive India',
        'screener_slug': 'colgate-palmolive-india',
        'sector': 'Consumer Goods',
        'bse_code': '500830',
        'ir_url': None
    },
    'PIDILITIND': {
        'name': 'Pidilite Industries',
        'screener_slug': 'pidilite-industries',
        'sector': 'Chemicals',
        'bse_code': '500331',
        'ir_url': None
    },
    'MARICO': {
        'name': 'Marico Limited',
        'screener_slug': 'marico',
        'sector': 'Consumer Goods',
        'bse_code': '531642',
        'ir_url': None
    },
    'DABUR': {
        'name': 'Dabur India',
        'screener_slug': 'dabur-india',
        'sector': 'Consumer Goods',
        'bse_code': '500096',
        'ir_url': None
    },
    'LUPIN': {
        'name': 'Lupin Limited',
        'screener_slug': 'lupin',
        'sector': 'Pharmaceuticals',
        'bse_code': '500257',
        'ir_url': None
    },
    'BIOCON': {
        'name': 'Biocon Limited',
        'screener_slug': 'biocon',
        'sector': 'Pharmaceuticals',
        'bse_code': '532523',
        'ir_url': None
    },
    'MOTHERSUMI': {
        'name': 'Motherson Sumi Systems',
        'screener_slug': 'samvardhana-motherson',
        'sector': 'Automobile',
        'bse_code': '517334',
        'ir_url': None
    },
    'TVSMOTOR': {
        'name': 'TVS Motor Company',
        'screener_slug': 'tvs-motor',
        'sector': 'Automobile',
        'bse_code': '532343',
        'ir_url': None
    },
    'MRF': {
        'name': 'MRF Limited',
        'screener_slug': 'mrf',
        'sector': 'Automobile',
        'bse_code': '500290',
        'ir_url': None
    },
    'DMART': {
        'name': 'Avenue Supermarts',
        'screener_slug': 'dmart',
        'sector': 'Retail',
        'bse_code': '540376',
        'ir_url': None
    },
    'TRENT': {
        'name': 'Trent Limited',
        'screener_slug': 'trent',
        'sector': 'Retail',
        'bse_code': '500251',
        'ir_url': None
    },
    'PAGEIND': {
        'name': 'Page Industries',
        'screener_slug': 'page-industries',
        'sector': 'Textiles',
        'bse_code': '532827',
        'ir_url': None
    },
    'HEROMOTOCO': {
        'name': 'Hero MotoCorp',
        'screener_slug': 'hero-motocorp',
        'sector': 'Automobile',
        'bse_code': '500182',
        'ir_url': None
    },
    'BAJAJ-AUTO': {
        'name': 'Bajaj Auto',
        'screener_slug': 'bajaj-auto',
        'sector': 'Automobile',
        'bse_code': '532977',
        'ir_url': None
    },
    'M&M': {
        'name': 'Mahindra & Mahindra',
        'screener_slug': 'mahindra-mahindra',
        'sector': 'Automobile',
        'bse_code': '500520',
        'ir_url': None
    },
    'EICHERMOT': {
        'name': 'Eicher Motors',
        'screener_slug': 'eicher-motors',
        'sector': 'Automobile',
        'bse_code': '505200',
        'ir_url': None
    },
    'GRASIM': {
        'name': 'Grasim Industries',
        'screener_slug': 'grasim-industries',
        'sector': 'Cement',
        'bse_code': '500300',
        'ir_url': None
    },
    'SHREECEM': {
        'name': 'Shree Cement',
        'screener_slug': 'shree-cement',
        'sector': 'Cement',
        'bse_code': '500387',
        'ir_url': None
    },
    'UPL': {
        'name': 'UPL Limited',
        'screener_slug': 'upl',
        'sector': 'Chemicals',
        'bse_code': '512070',
        'ir_url': None
    },
    'JSWSTEEL': {
        'name': 'JSW Steel',
        'screener_slug': 'jsw-steel',
        'sector': 'Steel',
        'bse_code': '500228',
        'ir_url': None
    },
    'HINDALCO': {
        'name': 'Hindalco Industries',
        'screener_slug': 'hindalco-industries',
        'sector': 'Metals',
        'bse_code': '500440',
        'ir_url': None
    },
    'COALINDIA': {
        'name': 'Coal India',
        'screener_slug': 'coal-india',
        'sector': 'Mining',
        'bse_code': '533278',
        'ir_url': None
    },
    'IOC': {
        'name': 'Indian Oil Corporation',
        'screener_slug': 'indian-oil-corporation',
        'sector': 'Oil & Gas',
        'bse_code': '530965',
        'ir_url': None
    },
    'TATACONSUM': {
        'name': 'Tata Consumer Products',
        'screener_slug': 'tata-consumer-products',
        'sector': 'Consumer Goods',
        'bse_code': '500800',
        'ir_url': None
    },
    'ADANIENT': {
        'name': 'Adani Enterprises',
        'screener_slug': 'adani-enterprises',
        'sector': 'Conglomerate',
        'bse_code': '512599',
        'ir_url': None
    },
}


class SymbolMapper:
    """Handles symbol to company information mapping"""

    def __init__(self, custom_map: Optional[Dict] = None):
        """
        Initialize symbol mapper

        Args:
            custom_map: Optional custom mapping dictionary to extend defaults
        """
        self.symbol_map = SYMBOL_MAP.copy()

        if custom_map:
            self.symbol_map.update(custom_map)

        logger.info(f"✅ SymbolMapper initialized with {len(self.symbol_map)} symbols")

    def get_company_info(self, symbol: str) -> Dict[str, str]:
        """
        Get complete company information for a symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')

        Returns:
            Dictionary with name, screener_slug, sector, bse_code, ir_url
        """
        # Clean symbol
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper().strip()

        # Return mapped info or defaults
        if clean_symbol in self.symbol_map:
            return self.symbol_map[clean_symbol]
        else:
            logger.warning(f"Symbol {clean_symbol} not found in mapping")
            return {
                'name': clean_symbol,
                'screener_slug': clean_symbol.lower(),
                'sector': 'Unknown',
                'bse_code': None,
                'ir_url': None
            }

    def get_screener_slug(self, symbol: str) -> Optional[str]:
        """
        Get Screener.in slug for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Screener.in company slug or None
        """
        info = self.get_company_info(symbol)
        return info.get('screener_slug')

    def get_bse_code(self, symbol: str) -> Optional[str]:
        """
        Get BSE script code for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            BSE script code or None
        """
        info = self.get_company_info(symbol)
        return info.get('bse_code')

    def get_ir_url(self, symbol: str) -> Optional[str]:
        """
        Get Investor Relations URL for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            IR URL or None
        """
        info = self.get_company_info(symbol)
        return info.get('ir_url')

    def add_symbol(self, symbol: str, name: str, screener_slug: str,
                   sector: str = 'Unknown', bse_code: str = None, ir_url: str = None):
        """
        Add a new symbol to the mapping

        Args:
            symbol: Stock symbol
            name: Company name
            screener_slug: Screener.in company slug
            sector: Company sector
            bse_code: BSE script code
            ir_url: Investor Relations URL
        """
        self.symbol_map[symbol.upper()] = {
            'name': name,
            'screener_slug': screener_slug,
            'sector': sector,
            'bse_code': bse_code,
            'ir_url': ir_url
        }
        logger.info(f"Added symbol {symbol} to mapper")

    def get_all_symbols(self) -> list:
        """Get list of all mapped symbols"""
        return list(self.symbol_map.keys())

    def get_symbols_by_sector(self, sector: str) -> list:
        """
        Get all symbols in a specific sector

        Args:
            sector: Sector name

        Returns:
            List of symbols in that sector
        """
        return [
            symbol for symbol, info in self.symbol_map.items()
            if info.get('sector', '').lower() == sector.lower()
        ]

    def has_ir_url(self, symbol: str) -> bool:
        """Check if symbol has investor relations URL configured"""
        info = self.get_company_info(symbol)
        return bool(info.get('ir_url'))

    def has_bse_code(self, symbol: str) -> bool:
        """Check if symbol has BSE code configured"""
        info = self.get_company_info(symbol)
        return bool(info.get('bse_code'))

    def export_to_dict(self) -> Dict:
        """Export entire mapping as dictionary"""
        return self.symbol_map.copy()

    def import_from_dict(self, mapping: Dict):
        """Import mapping from dictionary"""
        self.symbol_map.update(mapping)
        logger.info(f"Imported {len(mapping)} symbols")

    def get_statistics(self) -> Dict:
        """Get statistics about the symbol mapping"""
        total = len(self.symbol_map)
        with_ir = sum(1 for info in self.symbol_map.values() if info.get('ir_url'))
        with_bse = sum(1 for info in self.symbol_map.values() if info.get('bse_code'))

        sectors = {}
        for info in self.symbol_map.values():
            sector = info.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1

        return {
            'total_symbols': total,
            'with_ir_url': with_ir,
            'with_bse_code': with_bse,
            'sectors': sectors,
            'coverage': {
                'ir_url_percent': (with_ir / total * 100) if total > 0 else 0,
                'bse_code_percent': (with_bse / total * 100) if total > 0 else 0
            }
        }


# Helper function for quick access
def get_mapper() -> SymbolMapper:
    """Get a pre-initialized SymbolMapper instance"""
    return SymbolMapper()

STOCKS = {
    'SUNPHARMA': {
        'name': 'Sun Pharmaceutical',
        'screener_slug': 'sun-pharma',
        'sector': 'Pharmaceuticals'
    },
    'ULTRACEMCO': {
        'name': 'UltraTech Cement',
        'screener_slug': 'ultratech-cement',
        'sector': 'Cement'
    },
    'NESTLEIND': {
        'name': 'Nestle India',
        'screener_slug': 'nestle-india',
        'sector': 'Consumer Goods'
    },
    'HCLTECH': {
        'name': 'HCL Technologies',
        'screener_slug': 'hcl-technologies',
        'sector': 'Information Technology'
    },
    'AXISBANK': {
        'name': 'Axis Bank',
        'screener_slug': 'axis-bank',
        'sector': 'Banking'
    },
    'WIPRO': {
        'name': 'Wipro',
        'screener_slug': 'wipro',
        'sector': 'Information Technology'
    },
    'NTPC': {
        'name': 'NTPC',
        'screener_slug': 'ntpc',
        'sector': 'Power'
    },
    'POWERGRID': {
        'name': 'Power Grid Corporation',
        'screener_slug': 'power-grid-corporation',
        'sector': 'Power'
    },
    'ONGC': {
        'name': 'Oil & Natural Gas Corporation',
        'screener_slug': 'ongc',
        'sector': 'Oil & Gas'
    },
    'TECHM': {
        'name': 'Tech Mahindra',
        'screener_slug': 'tech-mahindra',
        'sector': 'Information Technology'
    },
    'TATASTEEL': {
        'name': 'Tata Steel',
        'screener_slug': 'tata-steel',
        'sector': 'Steel'
    },
    'ITC': {
        'name': 'ITC Limited',
        'screener_slug': 'itc',
        'sector': 'Consumer Goods'
    },
    'TATAMOTORS': {
        'name': 'Tata Motors',
        'screener_slug': 'tata-motors',
        'sector': 'Automobile'
    },
    'ADANIPORTS': {
        'name': 'Adani Ports',
        'screener_slug': 'adani-ports-sez',
        'sector': 'Infrastructure'
    },
    'INDUSINDBK': {
        'name': 'IndusInd Bank',
        'screener_slug': 'indusind-bank',
        'sector': 'Banking'
    },
    'CIPLA': {
        'name': 'Cipla',
        'screener_slug': 'cipla',
        'sector': 'Pharmaceuticals'
    },
    'DRREDDY': {
        'name': "Dr. Reddy's Laboratories",
        'screener_slug': 'dr-reddys-laboratories',
        'sector': 'Pharmaceuticals'
    },
    'DIVISLAB': {
        'name': "Divi's Laboratories",
        'screener_slug': 'divis-laboratories',
        'sector': 'Pharmaceuticals'
    },
    'BRITANNIA': {
        'name': 'Britannia Industries',
        'screener_slug': 'britannia-industries',
        'sector': 'Consumer Goods'
    },
}
class SymbolMapper:
    """Handles symbol to company name/slug mapping"""

    def __init__(self, custom_map: Optional[Dict] = None):
        """
        Initialize symbol mapper

        Args:
            custom_map: Optional custom mapping dictionary to extend defaults
        """
        self.symbol_map = SYMBOL_MAP.copy()

        if custom_map:
            self.symbol_map.update(custom_map)

        logger.info(f"✅ SymbolMapper initialized with {len(self.symbol_map)} symbols")

    def get_company_info(self, symbol: str) -> Dict[str, str]:
        """
        Get company information for a symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')

        Returns:
            Dictionary with name, screener_slug, and sector
        """
        # Clean symbol
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper().strip()

        # Return mapped info or defaults
        if clean_symbol in self.symbol_map:
            return self.symbol_map[clean_symbol]
        else:
            logger.warning(f"Symbol {clean_symbol} not found in mapping")
            return {
                'name': clean_symbol,
                'screener_slug': clean_symbol.lower(),
                'sector': 'Unknown'
            }

    def get_screener_slug(self, symbol: str) -> Optional[str]:
        """
        Get Screener.in slug for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Screener.in company slug or None
        """
        info = self.get_company_info(symbol)
        return info.get('screener_slug')

    def add_symbol(self, symbol: str, name: str, screener_slug: str, sector: str = 'Unknown'):
        """
        Add a new symbol to the mapping

        Args:
            symbol: Stock symbol
            name: Company name
            screener_slug: Screener.in company slug
            sector: Company sector
        """
        self.symbol_map[symbol.upper()] = {
            'name': name,
            'screener_slug': screener_slug,
            'sector': sector
        }
        logger.info(f"Added symbol {symbol} to mapper")

    def get_all_symbols(self) -> list:
        """Get list of all mapped symbols"""
        return list(self.symbol_map.keys())