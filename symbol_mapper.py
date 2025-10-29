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
#     'screener_slug': 'company-slug-on-screener-in', # CORRECTED SLUGS
#     'sector': 'Sector Name',
#     'bse_code': 'BSE Script Code',
#     'ir_url': 'Investor Relations URL'
# }

SYMBOL_MAP = {
    # NIFTY 50 - Large Cap (Slugs Updated based on common Screener pattern)
    'RELIANCE': {
        'name': 'Reliance Industries',
        'screener_slug': 'RELIANCE', # Corrected
        'sector': 'Oil & Gas',
        'bse_code': '500325',
        'ir_url': 'https://www.ril.com/InvestorRelations.aspx'
    },
    'TCS': {
        'name': 'Tata Consultancy Services',
        'screener_slug': 'TCS', # Corrected
        'sector': 'Information Technology',
        'bse_code': '532540',
        'ir_url': 'https://www.tcs.com/investor-relations'
    },
    'HDFCBANK': {
        'name': 'HDFC Bank',
        'screener_slug': 'HDFCBANK', # Corrected
        'sector': 'Banking',
        'bse_code': '500180',
        'ir_url': 'https://www.hdfcbank.com/personal/about-us/investor-relations'
    },
    'INFY': {
        'name': 'Infosys',
        'screener_slug': 'INFY', # Corrected
        'sector': 'Information Technology',
        'bse_code': '500209',
        'ir_url': 'https://www.infosys.com/investors/reports-filings.html'
    },
    'HINDUNILVR': {
        'name': 'Hindustan Unilever',
        'screener_slug': 'HINDUNILVR', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500696',
        'ir_url': 'https://www.hul.co.in/investor-relations/'
    },
    'ICICIBANK': {
        'name': 'ICICI Bank',
        'screener_slug': 'ICICIBANK', # Corrected
        'sector': 'Banking',
        'bse_code': '532174',
        'ir_url': 'https://www.icicibank.com/aboutus/annual_reports.page'
    },
    'KOTAKBANK': {
        'name': 'Kotak Mahindra Bank',
        'screener_slug': 'KOTAKBANK', # Corrected
        'sector': 'Banking',
        'bse_code': '500247',
        'ir_url': 'https://www.kotak.com/en/investor-relations.html'
    },
    'BAJFINANCE': {
        'name': 'Bajaj Finance',
        'screener_slug': 'BAJFINANCE', # Corrected
        'sector': 'Financial Services',
        'bse_code': '500034',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations' # Note: Points to Finserv, might be specific Finance page
    },
    'LT': {
        'name': 'Larsen & Toubro',
        'screener_slug': 'LT', # Corrected (Was already correct)
        'sector': 'Construction',
        'bse_code': '500510',
        'ir_url': 'https://www.larsentoubro.com/corporate/investors/'
    },
    'SBIN': {
        'name': 'State Bank of India',
        'screener_slug': 'SBIN', # Corrected
        'sector': 'Banking',
        'bse_code': '500112',
        'ir_url': 'https://bank.sbi/web/investor-relations'
    },
    'BHARTIARTL': {
        'name': 'Bharti Airtel',
        'screener_slug': 'BHARTIARTL', # Corrected
        'sector': 'Telecommunications',
        'bse_code': '532454',
        'ir_url': 'https://www.airtel.in/about-bharti/equity/investor-relations'
    },
    'ASIANPAINT': {
        'name': 'Asian Paints',
        'screener_slug': 'ASIANPAINT', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500820',
        'ir_url': 'https://www.asianpaints.com/investors.html'
    },
    'MARUTI': {
        'name': 'Maruti Suzuki India', # Slightly more specific name
        'screener_slug': 'MARUTI', # Corrected
        'sector': 'Automobile',
        'bse_code': '532500',
        'ir_url': 'https://www.marutisuzuki.com/corporate/investor'
    },
    'TITAN': {
        'name': 'Titan Company',
        'screener_slug': 'TITAN', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500114',
        'ir_url': 'https://www.titancompany.in/investor-information'
    },
    'SUNPHARMA': {
        'name': 'Sun Pharmaceutical Industries', # More specific name
        'screener_slug': 'SUNPHARMA', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '524715',
        'ir_url': 'https://www.sunpharma.com/investors'
    },
    'ULTRACEMCO': {
        'name': 'UltraTech Cement',
        'screener_slug': 'ULTRACEMCO', # Corrected
        'sector': 'Cement',
        'bse_code': '532538',
        'ir_url': 'https://www.ultratechcement.com/investors'
    },
    'NESTLEIND': {
        'name': 'Nestle India',
        'screener_slug': 'NESTLEIND', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500790',
        'ir_url': 'https://www.nestle.in/investors'
    },
    'HCLTECH': {
        'name': 'HCL Technologies',
        'screener_slug': 'HCLTECH', # Corrected
        'sector': 'Information Technology',
        'bse_code': '532281',
        'ir_url': 'https://www.hcltech.com/investors'
    },
    'AXISBANK': {
        'name': 'Axis Bank',
        'screener_slug': 'AXISBANK', # Corrected
        'sector': 'Banking',
        'bse_code': '532215',
        'ir_url': 'https://www.axisbank.com/shareholders-corner'
    },
    'WIPRO': {
        'name': 'Wipro',
        'screener_slug': 'WIPRO', # Corrected
        'sector': 'Information Technology',
        'bse_code': '507685',
        'ir_url': 'https://www.wipro.com/investors/'
    },
    'NTPC': {
        'name': 'NTPC',
        'screener_slug': 'NTPC', # Corrected
        'sector': 'Power',
        'bse_code': '532555',
        'ir_url': 'https://www.ntpc.co.in/investors'
    },
    'POWERGRID': {
        'name': 'Power Grid Corporation of India', # More specific name
        'screener_slug': 'POWERGRID', # Corrected
        'sector': 'Power',
        'bse_code': '532898',
        'ir_url': 'https://www.powergrid.in/investor-relations'
    },
    'ONGC': {
        'name': 'Oil & Natural Gas Corporation',
        'screener_slug': 'ONGC', # Corrected
        'sector': 'Oil & Gas',
        'bse_code': '500312',
        'ir_url': 'https://www.ongcindia.com/wps/wcm/connect/en/investor/'
    },
    'TECHM': {
        'name': 'Tech Mahindra',
        'screener_slug': 'TECHM', # Corrected
        'sector': 'Information Technology',
        'bse_code': '532755',
        'ir_url': 'https://www.techmahindra.com/investors.html'
    },
    'TATASTEEL': {
        'name': 'Tata Steel',
        'screener_slug': 'TATASTEEL', # Corrected
        'sector': 'Steel',
        'bse_code': '500470',
        'ir_url': 'https://www.tatasteel.com/investors/'
    },
    'ITC': {
        'name': 'ITC Limited',
        'screener_slug': 'ITC', # Corrected
        'sector': 'Consumer Goods', # Primarily, though conglomerate
        'bse_code': '500875',
        'ir_url': 'https://www.itcportal.com/about-itc/shareholder-value.aspx'
    },
    'TATAMOTORS': {
        'name': 'Tata Motors',
        'screener_slug': 'TATAMOTORS', # Corrected
        'sector': 'Automobile',
        'bse_code': '500570',
        'ir_url': 'https://www.tatamotors.com/investors/'
    },
    'ADANIPORTS': {
        'name': 'Adani Ports & SEZ', # More specific name
        'screener_slug': 'ADANIPORTS', # Corrected
        'sector': 'Infrastructure',
        'bse_code': '532921',
        'ir_url': 'https://www.adaniports.com/Investors'
    },
    'INDUSINDBK': {
        'name': 'IndusInd Bank',
        'screener_slug': 'INDUSINDBK', # Corrected
        'sector': 'Banking',
        'bse_code': '532187',
        'ir_url': 'https://www.indusind.com/in/en/personal/useful-links/investor-relations.html'
    },
    'CIPLA': {
        'name': 'Cipla',
        'screener_slug': 'CIPLA', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '500087',
        'ir_url': 'https://www.cipla.com/investor-information'
    },
    'DRREDDY': {
        'name': "Dr. Reddy's Laboratories",
        'screener_slug': 'DRREDDY', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '500124',
        'ir_url': 'https://www.drreddys.com/investors/'
    },
    'DIVISLAB': {
        'name': "Divi's Laboratories",
        'screener_slug': 'DIVISLAB', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '532488',
        'ir_url': 'https://www.divislabs.com/investors.html'
    },
    'BRITANNIA': {
        'name': 'Britannia Industries',
        'screener_slug': 'BRITANNIA', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500825',
        'ir_url': 'https://www.britannia.co.in/investors'
    },
    'BPCL': {
        'name': 'Bharat Petroleum Corporation', # More specific name
        'screener_slug': 'BPCL', # Corrected
        'sector': 'Oil & Gas',
        'bse_code': '500547',
        'ir_url': 'https://www.bharatpetroleum.in/Investor.aspx'
    },
    'APOLLOHOSP': {
        'name': 'Apollo Hospitals Enterprise', # More specific name
        'screener_slug': 'APOLLOHOSP', # Corrected
        'sector': 'Healthcare',
        'bse_code': '508869',
        'ir_url': 'https://www.apollohospitals.com/investor_relations/'
    },
    'BAJAJFINSV': {
        'name': 'Bajaj Finserv',
        'screener_slug': 'BAJAJFINSV', # Corrected
        'sector': 'Financial Services',
        'bse_code': '532978',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations'
    },
    'HDFCLIFE': {
        'name': 'HDFC Life Insurance Company', # More specific name
        'screener_slug': 'HDFCLIFE', # Corrected
        'sector': 'Insurance',
        'bse_code': '540777',
        'ir_url': 'https://www.hdfclife.com/investor-relations'
    },
    'SBILIFE': {
        'name': 'SBI Life Insurance Company', # More specific name
        'screener_slug': 'SBILIFE', # Corrected
        'sector': 'Insurance',
        'bse_code': '540719',
        'ir_url': 'https://www.sbilife.co.in/en/investor-information'
    },
    'ADANIENT': { # Moved here for consistency
        'name': 'Adani Enterprises',
        'screener_slug': 'ADANIENT', # Corrected
        'sector': 'Conglomerate',
        'bse_code': '512599',
        'ir_url': 'https://www.adanienterprises.com/investors'
    },
    'COALINDIA': {
        'name': 'Coal India',
        'screener_slug': 'COALINDIA', # Corrected
        'sector': 'Mining',
        'bse_code': '533278',
        'ir_url': 'https://www.coalindia.in/en-us/company/investors.aspx'
    },
    'HINDALCO': {
        'name': 'Hindalco Industries',
        'screener_slug': 'HINDALCO', # Corrected
        'sector': 'Metals',
        'bse_code': '500440',
        'ir_url': 'https://www.hindalco.com/investor-centre'
    },
    'JSWSTEEL': {
        'name': 'JSW Steel',
        'screener_slug': 'JSWSTEEL', # Corrected
        'sector': 'Steel',
        'bse_code': '500228',
        'ir_url': 'https://www.jsw.in/steel/investors/overview'
    },
    'BAJAJ-AUTO': { # Note: Fyers symbol might just be BAJAJAUTO
        'name': 'Bajaj Auto',
        'screener_slug': 'BAJAJ-AUTO', # This one might be hyphenated, verify
        'sector': 'Automobile',
        'bse_code': '532977',
        'ir_url': 'https://www.bajajauto.com/investors/'
    },
    'M&M': { # Note: Fyers symbol might be M&M or MAHM. Verify.
        'name': 'Mahindra & Mahindra',
        'screener_slug': 'M&M', # Screener often uses &
        'sector': 'Automobile',
        'bse_code': '500520',
        'ir_url': 'https://www.mahindra.com/investors'
    },
    'HEROMOTOCO': {
        'name': 'Hero MotoCorp',
        'screener_slug': 'HEROMOTOCO', # Corrected
        'sector': 'Automobile',
        'bse_code': '500182',
        'ir_url': 'https://www.heromotocorp.com/en-in/investor-relations/'
    },
    'GRASIM': {
        'name': 'Grasim Industries',
        'screener_slug': 'GRASIM', # Corrected
        'sector': 'Cement', # Also Diversified
        'bse_code': '500300',
        'ir_url': 'https://www.grasim.com/investors/'
    },
    'SHREECEM': {
        'name': 'Shree Cement',
        'screener_slug': 'SHREECEM', # Corrected
        'sector': 'Cement',
        'bse_code': '500387',
        'ir_url': 'https://www.shreecement.com/investors/stock-information/'
    },
    'EICHERMOT': {
        'name': 'Eicher Motors',
        'screener_slug': 'EICHERMOT', # Corrected
        'sector': 'Automobile',
        'bse_code': '505200',
        'ir_url': 'https://www.eichermotors.com/investors'
    },
    'UPL': {
        'name': 'UPL Limited',
        'screener_slug': 'UPL', # Corrected
        'sector': 'Chemicals',
        'bse_code': '512070',
        'ir_url': 'https://www.upl-ltd.com/investors'
    },
    'IOC': {
        'name': 'Indian Oil Corporation',
        'screener_slug': 'IOC', # Corrected
        'sector': 'Oil & Gas',
        'bse_code': '530965',
        'ir_url': 'https://www.iocl.com/investor-relations'
    },
    'TATACONSUM': {
        'name': 'Tata Consumer Products',
        'screener_slug': 'TATACONSUM', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500800',
        'ir_url': 'https://www.tataconsumer.com/investors'
    },

    # Mid & Small Cap Additions (Slugs Corrected)
    'GODREJCP': {
        'name': 'Godrej Consumer Products',
        'screener_slug': 'GODREJCP', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '532424',
        'ir_url': 'https://www.godrejcp.com/investors'
    },
    'COLPAL': {
        'name': 'Colgate-Palmolive (India)', # More specific name
        'screener_slug': 'COLPAL', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500830',
        'ir_url': 'https://www.colgatepalmolive.co.in/investor-relations'
    },
    'PIDILITIND': {
        'name': 'Pidilite Industries',
        'screener_slug': 'PIDILITIND', # Corrected
        'sector': 'Chemicals',
        'bse_code': '500331',
        'ir_url': 'https://www.pidilite.com/investor-relations/'
    },
    'MARICO': {
        'name': 'Marico Limited',
        'screener_slug': 'MARICO', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '531642',
        'ir_url': 'https://marico.com/india/investors/'
    },
    'DABUR': {
        'name': 'Dabur India',
        'screener_slug': 'DABUR', # Corrected
        'sector': 'Consumer Goods',
        'bse_code': '500096',
        'ir_url': 'https://www.dabur.com/investor-relations'
    },
    'LUPIN': {
        'name': 'Lupin Limited',
        'screener_slug': 'LUPIN', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '500257',
        'ir_url': 'https://www.lupin.com/investor-relations/'
    },
    'BIOCON': {
        'name': 'Biocon Limited',
        'screener_slug': 'BIOCON', # Corrected
        'sector': 'Pharmaceuticals',
        'bse_code': '532523',
        'ir_url': 'https://www.biocon.com/investor-relations/'
    },
    # MOTHERSUMI --> Replaced with MOTHERSON (Check Fyers symbol too)
    'MOTHERSON': {
        'name': 'Samvardhana Motherson International',
        'screener_slug': 'MOTHERSON', # Corrected (assuming ticker is slug)
        'sector': 'Automobile', # Auto Ancillaries
        'bse_code': '517334', # Same BSE code
        'ir_url': 'https://www.motherson.com/investors'
    },
    'TVSMOTOR': {
        'name': 'TVS Motor Company',
        'screener_slug': 'TVSMOTOR', # Corrected
        'sector': 'Automobile',
        'bse_code': '532343',
        'ir_url': 'https://www.tvsmotor.com/about-us/investors'
    },
    'MRF': {
        'name': 'MRF Limited',
        'screener_slug': 'MRF', # Corrected
        'sector': 'Automobile', # Tyres
        'bse_code': '500290',
        'ir_url': 'https://www.mrftyres.com/investors'
    },
    'DMART': {
        'name': 'Avenue Supermarts',
        'screener_slug': 'DMART', # Corrected
        'sector': 'Retail',
        'bse_code': '540376',
        'ir_url': 'https://www.dmartindia.com/investorrelation'
    },
    'TRENT': {
        'name': 'Trent Limited',
        'screener_slug': 'TRENT', # Corrected
        'sector': 'Retail',
        'bse_code': '500251',
        'ir_url': 'https://www.trentlimited.com/investor-relations.html'
    },
    'PAGEIND': {
        'name': 'Page Industries',
        'screener_slug': 'PAGEIND', # Corrected
        'sector': 'Textiles',
        'bse_code': '532827',
        'ir_url': 'https://www.jockeyindia.com/pages/investor-relations'
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

    def get_company_info(self, symbol: str) -> Dict[str, Optional[str]]:
        """
        Get complete company information for a symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')

        Returns:
            Dictionary with name, screener_slug, sector, bse_code, ir_url
            Returns default values (like None for codes/urls) if symbol not found.
        """
        # Clean symbol
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper().strip()

        # Handle potential Fyers variations like '&' vs 'and' or '-'
        # This requires more specific rules or a fuzzy matching approach if needed,
        # For now, we rely on the keys being the standard NSE/BSE ticker.
        # Example: Fyers might use 'M&M' but the key here is 'M&M'.

        if clean_symbol in self.symbol_map:
            # Return a copy to prevent modification of the original map
            return self.symbol_map[clean_symbol].copy()
        else:
            logger.warning(f"Symbol {clean_symbol} not found in mapping. Returning defaults.")
            # Return a default structure with the cleaned symbol as name/slug
            return {
                'name': clean_symbol,
                'screener_slug': clean_symbol.upper(), # Default to uppercase ticker for slug
                'sector': 'Unknown',
                'bse_code': None,
                'ir_url': None
            }

    def get_screener_slug(self, symbol: str) -> Optional[str]:
        """Get Screener.in slug for a symbol"""
        info = self.get_company_info(symbol)
        # Return the slug or default to the uppercase symbol if not found
        return info.get('screener_slug') or symbol.replace('.NS', '').replace('.BO', '').upper().strip()

    def get_bse_code(self, symbol: str) -> Optional[str]:
        """Get BSE script code for a symbol"""
        info = self.get_company_info(symbol)
        return info.get('bse_code')

    def get_ir_url(self, symbol: str) -> Optional[str]:
        """Get Investor Relations URL for a symbol"""
        info = self.get_company_info(symbol)
        return info.get('ir_url')

    def add_symbol(self, symbol: str, name: str, screener_slug: str,
                   sector: str = 'Unknown', bse_code: Optional[str] = None, ir_url: Optional[str] = None):
        """Add or update a symbol in the mapping"""
        clean_symbol = symbol.upper().strip()
        self.symbol_map[clean_symbol] = {
            'name': name,
            'screener_slug': screener_slug,
            'sector': sector,
            'bse_code': bse_code,
            'ir_url': ir_url
        }
        logger.info(f"Added/Updated symbol {clean_symbol} in mapper")

    def get_all_symbols(self) -> list:
        """Get list of all mapped symbols"""
        return list(self.symbol_map.keys())

    def get_symbols_by_sector(self, sector: str) -> list:
        """Get all symbols in a specific sector (case-insensitive)"""
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
        """Import mapping from dictionary, overwriting existing"""
        self.symbol_map = mapping.copy() # Replace instead of update? Or update? Let's update.
        # self.symbol_map.update(mapping)
        logger.info(f"Imported/Updated {len(mapping)} symbols from dict")

    def get_statistics(self) -> Dict:
        """Get statistics about the symbol mapping"""
        total = len(self.symbol_map)
        if total == 0:
            return {'total_symbols': 0, 'with_ir_url': 0, 'with_bse_code': 0, 'sectors': {}, 'coverage': {}}

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
                'ir_url_percent': round((with_ir / total * 100), 1),
                'bse_code_percent': round((with_bse / total * 100), 1)
            }
        }


# Helper function for quick access
def get_mapper() -> SymbolMapper:
    """Get a pre-initialized SymbolMapper instance"""
    # Ensures a singleton-like behavior if called multiple times within the same process
    # Note: In a multi-process setup (like gunicorn workers), each process gets its own instance.
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = SymbolMapper()
    return _default_mapper

_default_mapper = None # Initialize the global variable

# --- Remove Duplicate Definitions ---
# The 'STOCKS' dictionary and the second 'SymbolMapper' class definition have been removed.
