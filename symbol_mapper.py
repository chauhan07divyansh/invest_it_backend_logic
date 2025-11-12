"""
Symbol Mapping Module - EXPANDED VERSION
Maps NSE/BSE symbols to Screener.in company slugs + BSE codes + IR URLs
SYNCHRONIZED WITH 'swing_trading.py' STOCK UNIVERSE
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# COMPREHENSIVE Symbol Mapping
# Source: swing_trading.py (initialize_expanded_stock_database)
# Merged with: Original symbol_mapper.py for bse_code and ir_url
#
# Format: 'SYMBOL': {
#     'name': 'Company Name',
#     'screener_slug': 'company-slug-on-screener-in', # (Defaults to SYMBOL if not in original map)
#     'sector': 'Sector Name',
#     'bse_code': 'BSE Script Code', (None if not in original map)
#     'ir_url': 'Investor Relations URL' (None if not in original map)
# }

SYMBOL_MAP = {
    # ========== NIFTY 50 (50 stocks) ==========
    'RELIANCE': {
        'name': 'Reliance Industries',
        'sector': 'Oil & Gas',
        'screener_slug': 'RELIANCE',
        'bse_code': '500325',
        'ir_url': 'https://www.ril.com/InvestorRelations.aspx'
    },
    'TCS': {
        'name': 'Tata Consultancy Services',
        'sector': 'Information Technology',
        'screener_slug': 'TCS',
        'bse_code': '532540',
        'ir_url': 'https://www.tcs.com/investor-relations'
    },
    'HDFCBANK': {
        'name': 'HDFC Bank',
        'sector': 'Banking',
        'screener_slug': 'HDFCBANK',
        'bse_code': '500180',
        'ir_url': 'https://www.hdfcbank.com/personal/about-us/investor-relations'
    },
    'INFY': {
        'name': 'Infosys',
        'sector': 'Information Technology',
        'screener_slug': 'INFY',
        'bse_code': '500209',
        'ir_url': 'https://www.infosys.com/investors/reports-filings.html'
    },
    'HINDUNILVR': {
        'name': 'Hindustan Unilever',
        'sector': 'Consumer Goods',
        'screener_slug': 'HINDUNILVR',
        'bse_code': '500696',
        'ir_url': 'https://www.hul.co.in/investor-relations/'
    },
    'ICICIBANK': {
        'name': 'ICICI Bank',
        'sector': 'Banking',
        'screener_slug': 'ICICIBANK',
        'bse_code': '532174',
        'ir_url': 'https://www.icicibank.com/aboutus/annual_reports.page'
    },
    'KOTAKBANK': {
        'name': 'Kotak Mahindra Bank',
        'sector': 'Banking',
        'screener_slug': 'KOTAKBANK',
        'bse_code': '500247',
        'ir_url': 'https://www.kotak.com/en/investor-relations.html'
    },
    'BAJFINANCE': {
        'name': 'Bajaj Finance',
        'sector': 'Financial Services',
        'screener_slug': 'BAJFINANCE',
        'bse_code': '500034',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations'
    },
    'LT': {
        'name': 'Larsen & Toubro',
        'sector': 'Construction',
        'screener_slug': 'LT',
        'bse_code': '500510',
        'ir_url': 'https://www.larsentoubro.com/corporate/investors/'
    },
    'SBIN': {
        'name': 'State Bank of India',
        'sector': 'Banking',
        'screener_slug': 'SBIN',
        'bse_code': '500112',
        'ir_url': 'https://bank.sbi/web/investor-relations'
    },
    'BHARTIARTL': {
        'name': 'Bharti Airtel',
        'sector': 'Telecommunications',
        'screener_slug': 'BHARTIARTL',
        'bse_code': '532454',
        'ir_url': 'https://www.airtel.in/about-bharti/equity/investor-relations'
    },
    'ASIANPAINT': {
        'name': 'Asian Paints',
        'sector': 'Consumer Goods',
        'screener_slug': 'ASIANPAINT',
        'bse_code': '500820',
        'ir_url': 'https://www.asianpaints.com/investors.html'
    },
    'MARUTI': {
        'name': 'Maruti Suzuki',
        'sector': 'Automobile',
        'screener_slug': 'MARUTI',
        'bse_code': '532500',
        'ir_url': 'https://www.marutisuzuki.com/corporate/investor'
    },
    'TITAN': {
        'name': 'Titan Company',
        'sector': 'Consumer Goods',
        'screener_slug': 'TITAN',
        'bse_code': '500114',
        'ir_url': 'https://www.titancompany.in/investor-information'
    },
    'SUNPHARMA': {
        'name': 'Sun Pharmaceutical',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'SUNPHARMA',
        'bse_code': '524715',
        'ir_url': 'https://www.sunpharma.com/investors'
    },
    'ULTRACEMCO': {
        'name': 'UltraTech Cement',
        'sector': 'Cement',
        'screener_slug': 'ULTRACEMCO',
        'bse_code': '532538',
        'ir_url': 'https://www.ultratechcement.com/investors'
    },
    'NESTLEIND': {
        'name': 'Nestle India',
        'sector': 'Consumer Goods',
        'screener_slug': 'NESTLEIND',
        'bse_code': '500790',
        'ir_url': 'https://www.nestle.in/investors'
    },
    'HCLTECH': {
        'name': 'HCL Technologies',
        'sector': 'Information Technology',
        'screener_slug': 'HCLTECH',
        'bse_code': '532281',
        'ir_url': 'https://www.hcltech.com/investors'
    },
    'AXISBANK': {
        'name': 'Axis Bank',
        'sector': 'Banking',
        'screener_slug': 'AXISBANK',
        'bse_code': '532215',
        'ir_url': 'https://www.axisbank.com/shareholders-corner'
    },
    'WIPRO': {
        'name': 'Wipro',
        'sector': 'Information Technology',
        'screener_slug': 'WIPRO',
        'bse_code': '507685',
        'ir_url': 'https://www.wipro.com/investors/'
    },
    'NTPC': {
        'name': 'NTPC',
        'sector': 'Power',
        'screener_slug': 'NTPC',
        'bse_code': '532555',
        'ir_url': 'https://www.ntpc.co.in/investors'
    },
    'POWERGRID': {
        'name': 'Power Grid Corporation',
        'sector': 'Power',
        'screener_slug': 'POWERGRID',
        'bse_code': '532898',
        'ir_url': 'https://www.powergrid.in/investor-relations'
    },
    'ONGC': {
        'name': 'Oil & Natural Gas Corporation',
        'sector': 'Oil & Gas',
        'screener_slug': 'ONGC',
        'bse_code': '500312',
        'ir_url': 'https://www.ongcindia.com/wps/wcm/connect/en/investor/'
    },
    'TECHM': {
        'name': 'Tech Mahindra',
        'sector': 'Information Technology',
        'screener_slug': 'TECHM',
        'bse_code': '532755',
        'ir_url': 'https://www.techmahindra.com/investors.html'
    },
    'TATASTEEL': {
        'name': 'Tata Steel',
        'sector': 'Steel',
        'screener_slug': 'TATASTEEL',
        'bse_code': '500470',
        'ir_url': 'https://www.tatasteel.com/investors/'
    },
    'ADANIENT': {
        'name': 'Adani Enterprises',
        'sector': 'Conglomerate',
        'screener_slug': 'ADANIENT',
        'bse_code': '512599',
        'ir_url': 'https://www.adanienterprises.com/investors'
    },
    'COALINDIA': {
        'name': 'Coal India',
        'sector': 'Mining',
        'screener_slug': 'COALINDIA',
        'bse_code': '533278',
        'ir_url': 'https://www.coalindia.in/en-us/company/investors.aspx'
    },
    'HINDALCO': {
        'name': 'Hindalco Industries',
        'sector': 'Metals',
        'screener_slug': 'HINDALCO',
        'bse_code': '500440',
        'ir_url': 'https://www.hindalco.com/investor-centre'
    },
    'JSWSTEEL': {
        'name': 'JSW Steel',
        'sector': 'Steel',
        'screener_slug': 'JSWSTEEL',
        'bse_code': '500228',
        'ir_url': 'https://www.jsw.in/steel/investors/overview'
    },
    'BAJAJ-AUTO': {
        'name': 'Bajaj Auto',
        'sector': 'Automobile',
        'screener_slug': 'BAJAJ-AUTO',
        'bse_code': '532977',
        'ir_url': 'https://www.bajajauto.com/investors/'
    },
    'M&M': {
        'name': 'Mahindra & Mahindra',
        'sector': 'Automobile',
        'screener_slug': 'M&M',
        'bse_code': '500520',
        'ir_url': 'https://www.mahindra.com/investors'
    },
    'HEROMOTOCO': {
        'name': 'Hero MotoCorp',
        'sector': 'Automobile',
        'screener_slug': 'HEROMOTOCO',
        'bse_code': '500182',
        'ir_url': 'https://www.heromotocorp.com/en-in/investor-relations/'
    },
    'GRASIM': {
        'name': 'Grasim Industries',
        'sector': 'Cement',
        'screener_slug': 'GRASIM',
        'bse_code': '500300',
        'ir_url': 'https-://www.grasim.com/investors/'
    },
    'SHREECEM': {
        'name': 'Shree Cement',
        'sector': 'Cement',
        'screener_slug': 'SHREECEM',
        'bse_code': '500387',
        'ir_url': 'https://www.shreecement.com/investors/stock-information/'
    },
    'EICHERMOT': {
        'name': 'Eicher Motors',
        'sector': 'Automobile',
        'screener_slug': 'EICHERMOT',
        'bse_code': '505200',
        'ir_url': 'https://www.eichermotors.com/investors'
    },
    'UPL': {
        'name': 'UPL Limited',
        'sector': 'Chemicals',
        'screener_slug': 'UPL',
        'bse_code': '512070',
        'ir_url': 'https://www.upl-ltd.com/investors'
    },
    'BPCL': {
        'name': 'Bharat Petroleum',
        'sector': 'Oil & Gas',
        'screener_slug': 'BPCL',
        'bse_code': '500547',
        'ir_url': 'https://www.bharatpetroleum.in/Investor.aspx'
    },
    'DIVISLAB': {
        'name': "Divi's Laboratories",
        'sector': 'Pharmaceuticals',
        'screener_slug': 'DIVISLAB',
        'bse_code': '532488',
        'ir_url': 'https://www.divislabs.com/investors.html'
    },
    'DRREDDY': {
        'name': "Dr. Reddy's Laboratories",
        'sector': 'Pharmaceuticals',
        'screener_slug': 'DRREDDY',
        'bse_code': '500124',
        'ir_url': 'https://www.drreddys.com/investors/'
    },
    'CIPLA': {
        'name': 'Cipla',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'CIPLA',
        'bse_code': '500087',
        'ir_url': 'https://www.cipla.com/investor-information'
    },
    'BRITANNIA': {
        'name': 'Britannia Industries',
        'sector': 'Consumer Goods',
        'screener_slug': 'BRITANNIA',
        'bse_code': '500825',
        'ir_url': 'https://www.britannia.co.in/investors'
    },
    'TATACONSUM': {
        'name': 'Tata Consumer Products',
        'sector': 'Consumer Goods',
        'screener_slug': 'TATACONSUM',
        'bse_code': '500800',
        'ir_url': 'https://www.tataconsumer.com/investors'
    },
    'IOC': {
        'name': 'Indian Oil Corporation',
        'sector': 'Oil & Gas',
        'screener_slug': 'IOC',
        'bse_code': '530965',
        'ir_url': 'https://www.iocl.com/investor-relations'
    },
    'APOLLOHOSP': {
        'name': 'Apollo Hospitals',
        'sector': 'Healthcare',
        'screener_slug': 'APOLLOHOSP',
        'bse_code': '508869',
        'ir_url': 'https://www.apollohospitals.com/investor_relations/'
    },
    'BAJAJFINSV': {
        'name': 'Bajaj Finserv',
        'sector': 'Financial Services',
        'screener_slug': 'BAJAJFINSV',
        'bse_code': '532978',
        'ir_url': 'https://www.bajajfinserv.in/investor-relations'
    },
    'HDFCLIFE': {
        'name': 'HDFC Life Insurance',
        'sector': 'Insurance',
        'screener_slug': 'HDFCLIFE',
        'bse_code': '540777',
        'ir_url': 'https://www.hdfclife.com/investor-relations'
    },
    'SBILIFE': {
        'name': 'SBI Life Insurance',
        'sector': 'Insurance',
        'screener_slug': 'SBILIFE',
        'bse_code': '540719',
        'ir_url': 'https://www.sbilife.co.in/en/investor-information'
    },
    'INDUSINDBK': {
        'name': 'IndusInd Bank',
        'sector': 'Banking',
        'screener_slug': 'INDUSINDBK',
        'bse_code': '532187',
        'ir_url': 'https://www.indusind.com/in/en/personal/useful-links/investor-relations.html'
    },
    'ADANIPORTS': {
        'name': 'Adani Ports',
        'sector': 'Infrastructure',
        'screener_slug': 'ADANIPORTS',
        'bse_code': '532921',
        'ir_url': 'https://www.adaniports.com/Investors'
    },
    'TATAMOTORS': {
        'name': 'Tata Motors',
        'sector': 'Automobile',
        'screener_slug': 'TATAMOTORS',
        'bse_code': '500570',
        'ir_url': 'https://www.tatamotors.com/investors/'
    },
    'ITC': {
        'name': 'ITC Limited',
        'sector': 'Consumer Goods',
        'screener_slug': 'ITC',
        'bse_code': '500875',
        'ir_url': 'https://www.itcportal.com/about-itc/shareholder-value.aspx'
    },

    # ========== NIFTY NEXT 50 (50 stocks) ==========
    'SIEMENS': {
        'name': 'Siemens Limited',
        'sector': 'Capital Goods',
        'screener_slug': 'SIEMENS',
        'bse_code': None,
        'ir_url': None
    },
    'HAVELLS': {
        'name': 'Havells India',
        'sector': 'Electricals',
        'screener_slug': 'HAVELLS',
        'bse_code': None,
        'ir_url': None
    },
    'DLF': {
        'name': 'DLF Limited',
        'sector': 'Real Estate',
        'screener_slug': 'DLF',
        'bse_code': None,
        'ir_url': None
    },
    'GODREJCP': {
        'name': 'Godrej Consumer Products',
        'sector': 'Consumer Goods',
        'screener_slug': 'GODREJCP',
        'bse_code': '532424',
        'ir_url': 'https://www.godrejcp.com/investors'
    },
    'COLPAL': {
        'name': 'Colgate-Palmolive India',
        'sector': 'Consumer Goods',
        'screener_slug': 'COLPAL',
        'bse_code': '500830',
        'ir_url': 'https://www.colgatepalmolive.co.in/investor-relations'
    },
    'PIDILITIND': {
        'name': 'Pidilite Industries',
        'sector': 'Chemicals',
        'screener_slug': 'PIDILITIND',
        'bse_code': '500331',
        'ir_url': 'https://www.pidilite.com/investor-relations/'
    },
    'MARICO': {
        'name': 'Marico Limited',
        'sector': 'Consumer Goods',
        'screener_slug': 'MARICO',
        'bse_code': '531642',
        'ir_url': 'https://marico.com/india/investors/'
    },
    'DABUR': {
        'name': 'Dabur India',
        'sector': 'Consumer Goods',
        'screener_slug': 'DABUR',
        'bse_code': '500096',
        'ir_url': 'https://www.dabur.com/investor-relations'
    },
    'LUPIN': {
        'name': 'Lupin Limited',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'LUPIN',
        'bse_code': '500257',
        'ir_url': 'https://www.lupin.com/investor-relations/'
    },
    'BIOCON': {
        'name': 'Biocon Limited',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'BIOCON',
        'bse_code': '532523',
        'ir_url': 'https://www.biocon.com/investor-relations/'
    },
    'MOTHERSUMI': {
        'name': 'Motherson Sumi Systems',
        'sector': 'Automobile',
        'screener_slug': 'MOTHERSUMI',
        'bse_code': None,
        'ir_url': None
    },
    'BOSCHLTD': {
        'name': 'Bosch Limited',
        'sector': 'Automobile',
        'screener_slug': 'BOSCHLTD',
        'bse_code': None,
        'ir_url': None
    },
    'EXIDEIND': {
        'name': 'Exide Industries',
        'sector': 'Automobile',
        'screener_slug': 'EXIDEIND',
        'bse_code': None,
        'ir_url': None
    },
    'ASHOKLEY': {
        'name': 'Ashok Leyland',
        'sector': 'Automobile',
        'screener_slug': 'ASHOKLEY',
        'bse_code': None,
        'ir_url': None
    },
    'TVSMOTOR': {
        'name': 'TVS Motor Company',
        'sector': 'Automobile',
        'screener_slug': 'TVSMOTOR',
        'bse_code': '532343',
        'ir_url': 'https://www.tvsmotor.com/about-us/investors'
    },
    'BALKRISIND': {
        'name': 'Balkrishna Industries',
        'sector': 'Automobile',
        'screener_slug': 'BALKRISIND',
        'bse_code': None,
        'ir_url': None
    },
    'MRF': {
        'name': 'MRF Limited',
        'sector': 'Automobile',
        'screener_slug': 'MRF',
        'bse_code': '500290',
        'ir_url': 'https://www.mrftyres.com/investors'
    },
    'APOLLOTYRE': {
        'name': 'Apollo Tyres',
        'sector': 'Automobile',
        'screener_slug': 'APOLLOTYRE',
        'bse_code': None,
        'ir_url': None
    },
    'BHARATFORG': {
        'name': 'Bharat Forge',
        'sector': 'Automobile',
        'screener_slug': 'BHARATFORG',
        'bse_code': None,
        'ir_url': None
    },
    'CUMMINSIND': {
        'name': 'Cummins India',
        'sector': 'Automobile',
        'screener_slug': 'CUMMINSIND',
        'bse_code': None,
        'ir_url': None
    },
    'FEDERALBNK': {
        'name': 'Federal Bank',
        'sector': 'Banking',
        'screener_slug': 'FEDERALBNK',
        'bse_code': None,
        'ir_url': None
    },
    'BANDHANBNK': {
        'name': 'Bandhan Bank',
        'sector': 'Banking',
        'screener_slug': 'BANDHANBNK',
        'bse_code': None,
        'ir_url': None
    },
    'IDFCFIRSTB': {
        'name': 'IDFC First Bank',
        'sector': 'Banking',
        'screener_slug': 'IDFCFIRSTB',
        'bse_code': None,
        'ir_url': None
    },
    'PNB': {
        'name': 'Punjab National Bank',
        'sector': 'Banking',
        'screener_slug': 'PNB',
        'bse_code': None,
        'ir_url': None
    },
    'BANKBARODA': {
        'name': 'Bank of Baroda',
        'sector': 'Banking',
        'screener_slug': 'BANKBARODA',
        'bse_code': None,
        'ir_url': None
    },
    'CANBK': {
        'name': 'Canara Bank',
        'sector': 'Banking',
        'screener_slug': 'CANBK',
        'bse_code': None,
        'ir_url': None
    },
    'UNIONBANK': {
        'name': 'Union Bank of India',
        'sector': 'Banking',
        'screener_slug': 'UNIONBANK',
        'bse_code': None,
        'ir_url': None
    },
    'CHOLAFIN': {
        'name': 'Cholamandalam Investment',
        'sector': 'Financial Services',
        'screener_slug': 'CHOLAFIN',
        'bse_code': None,
        'ir_url': None
    },
    'LICHSGFIN': {
        'name': 'LIC Housing Finance',
        'sector': 'Financial Services',
        'screener_slug': 'LICHSGFIN',
        'bse_code': None,
        'ir_url': None
    },
    'SRTRANSFIN': {
        'name': 'Shriram Transport Finance',
        'sector': 'Financial Services',
        'screener_slug': 'SRTRANSFIN',
        'bse_code': None,
        'ir_url': None
    },
    'LTTS': {
        'name': 'L&T Technology Services',
        'sector': 'Information Technology',
        'screener_slug': 'LTTS',
        'bse_code': None,
        'ir_url': None
    },
    'PERSISTENT': {
        'name': 'Persistent Systems',
        'sector': 'Information Technology',
        'screener_slug': 'PERSISTENT',
        'bse_code': None,
        'ir_url': None
    },
    'COFORGE': {
        'name': 'Coforge Limited',
        'sector': 'Information Technology',
        'screener_slug': 'COFORGE',
        'bse_code': None,
        'ir_url': None
    },
    'MPHASIS': {
        'name': 'Mphasis Limited',
        'sector': 'Information Technology',
        'screener_slug': 'MPHASIS',
        'bse_code': None,
        'ir_url': None
    },
    'DMART': {
        'name': 'Avenue Supermarts',
        'sector': 'Retail',
        'screener_slug': 'DMART',
        'bse_code': '540376',
        'ir_url': 'https://www.dmartindia.com/investorrelation'
    },
    'TRENT': {
        'name': 'Trent Limited',
        'sector': 'Retail',
        'screener_slug': 'TRENT',
        'bse_code': '500251',
        'ir_url': 'https://www.trentlimited.com/investor-relations.html'
    },
    'PAGEIND': {
        'name': 'Page Industries',
        'sector': 'Textiles',
        'screener_slug': 'PAGEIND',
        'bse_code': '532827',
        'ir_url': 'https://www.jockeyindia.com/pages/investor-relations'
    },
    'RAYMOND': {
        'name': 'Raymond Limited',
        'sector': 'Textiles',
        'screener_slug': 'RAYMOND',
        'bse_code': None,
        'ir_url': None
    },
    'BERGEPAINT': {
        'name': 'Berger Paints',
        'sector': 'Consumer Goods',
        'screener_slug': 'BERGEPAINT',
        'bse_code': None,
        'ir_url': None
    },
    'VOLTAS': {
        'name': 'Voltas Limited',
        'sector': 'Consumer Durables',
        'screener_slug': 'VOLTAS',
        'bse_code': None,
        'ir_url': None
    },
    'WHIRLPOOL': {
        'name': 'Whirlpool of India',
        'sector': 'Consumer Durables',
        'screener_slug': 'WHIRLPOOL',
        'bse_code': None,
        'ir_url': None
    },
    'CROMPTON': {
        'name': 'Crompton Greaves',
        'sector': 'Electricals',
        'screener_slug': 'CROMPTON',
        'bse_code': None,
        'ir_url': None
    },
    'TORNTPHARM': {
        'name': 'Torrent Pharmaceuticals',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'TORNTPHARM',
        'bse_code': None,
        'ir_url': None
    },
    'AUROPHARMA': {
        'name': 'Aurobindo Pharma',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'AUROPHARMA',
        'bse_code': None,
        'ir_url': None
    },
    'ALKEM': {
        'name': 'Alkem Laboratories',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'ALKEM',
        'bse_code': None,
        'ir_url': None
    },
    'JUBLFOOD': {
        'name': 'Jubilant FoodWorks',
        'sector': 'Consumer Services',
        'screener_slug': 'JUBLFOOD',
        'bse_code': None,
        'ir_url': None
    },
    'VBL': {
        'name': 'Varun Beverages',
        'sector': 'Beverages',
        'screener_slug': 'VBL',
        'bse_code': None,
        'ir_url': None
    },
    'EMAMILTD': {
        'name': 'Emami Limited',
        'sector': 'Consumer Goods',
        'screener_slug': 'EMAMILTD',
        'bse_code': None,
        'ir_url': None
    },
    'GODREJPROP': {
        'name': 'Godrej Properties',
        'sector': 'Real Estate',
        'screener_slug': 'GODREJPROP',
        'bse_code': None,
        'ir_url': None
    },
    'OBEROIRLTY': {
        'name': 'Oberoi Realty',
        'sector': 'Real Estate',
        'screener_slug': 'OBEROIRLTY',
        'bse_code': None,
        'ir_url': None
    },

    # ========== NIFTY MIDCAP 100 (Top 100 liquid stocks) ==========
    'ABCAPITAL': {
        'name': 'Aditya Birla Capital',
        'sector': 'Financial Services',
        'screener_slug': 'ABCAPITAL',
        'bse_code': None,
        'ir_url': None
    },
    'ABFRL': {
        'name': 'Aditya Birla Fashion',
        'sector': 'Retail',
        'screener_slug': 'ABFRL',
        'bse_code': None,
        'ir_url': None
    },
    'ACC': {
        'name': 'ACC Limited',
        'sector': 'Cement',
        'screener_slug': 'ACC',
        'bse_code': None,
        'ir_url': None
    },
    'ADANIGREEN': {
        'name': 'Adani Green Energy',
        'sector': 'Power',
        'screener_slug': 'ADANIGREEN',
        'bse_code': None,
        'ir_url': None
    },
    'ADANIPOWER': {
        'name': 'Adani Power',
        'sector': 'Power',
        'screener_slug': 'ADANIPOWER',
        'bse_code': None,
        'ir_url': None
    },
    'AFFLE': {
        'name': 'Affle India',
        'sector': 'Technology',
        'screener_slug': 'AFFLE',
        'bse_code': None,
        'ir_url': None
    },
    'AIAENG': {
        'name': 'AIA Engineering',
        'sector': 'Capital Goods',
        'screener_slug': 'AIAENG',
        'bse_code': None,
        'ir_url': None
    },
    'AJANTPHARM': {
        'name': 'Ajanta Pharma',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'AJANTPHARM',
        'bse_code': None,
        'ir_url': None
    },
    'AKUMS': {
        'name': 'Akums Drugs',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'AKUMS',
        'bse_code': None,
        'ir_url': None
    },
    'AMBER': {
        'name': 'Amber Enterprises',
        'sector': 'Consumer Durables',
        'screener_slug': 'AMBER',
        'bse_code': None,
        'ir_url': None
    },
    'AMBUJACEM': {
        'name': 'Ambuja Cements',
        'sector': 'Cement',
        'screener_slug': 'AMBUJACEM',
        'bse_code': None,
        'ir_url': None
    },
    'ASTRAL': {
        'name': 'Astral Limited',
        'sector': 'Building Materials',
        'screener_slug': 'ASTRAL',
        'bse_code': None,
        'ir_url': None
    },
    'ATUL': {
        'name': 'Atul Limited',
        'sector': 'Chemicals',
        'screener_slug': 'ATUL',
        'bse_code': None,
        'ir_url': None
    },
    'AUBANK': {
        'name': 'AU Small Finance Bank',
        'sector': 'Banking',
        'screener_slug': 'AUBANK',
        'bse_code': None,
        'ir_url': None
    },
    'BAJAJELEC': {
        'name': 'Bajaj Electricals',
        'sector': 'Electricals',
        'screener_slug': 'BAJAJELEC',
        'bse_code': None,
        'ir_url': None
    },
    'BALAMINES': {
        'name': 'Balaji Amines',
        'sector': 'Chemicals',
        'screener_slug': 'BALAMINES',
        'bse_code': None,
        'ir_url': None
    },
    'BATAINDIA': {
        'name': 'Bata India',
        'sector': 'Footwear',
        'screener_slug': 'BATAINDIA',
        'bse_code': None,
        'ir_url': None
    },
    'BEL': {
        'name': 'Bharat Electronics',
        'sector': 'Defence',
        'screener_slug': 'BEL',
        'bse_code': None,
        'ir_url': None
    },
    'BHEL': {
        'name': 'Bharat Heavy Electricals',
        'sector': 'Capital Goods',
        'screener_slug': 'BHEL',
        'bse_code': None,
        'ir_url': None
    },
    'BRIGADE': {
        'name': 'Brigade Enterprises',
        'sector': 'Real Estate',
        'screener_slug': 'BRIGADE',
        'bse_code': None,
        'ir_url': None
    },
    'CESC': {
        'name': 'CESC Limited',
        'sector': 'Power',
        'screener_slug': 'CESC',
        'bse_code': None,
        'ir_url': None
    },
    'CHAMBLFERT': {
        'name': 'Chambal Fertilizers',
        'sector': 'Fertilizers',
        'screener_slug': 'CHAMBLFERT',
        'bse_code': None,
        'ir_url': None
    },
    'CONCOR': {
        'name': 'Container Corporation',
        'sector': 'Logistics',
        'screener_slug': 'CONCOR',
        'bse_code': None,
        'ir_url': None
    },
    'COROMANDEL': {
        'name': 'Coromandel International',
        'sector': 'Fertilizers',
        'screener_slug': 'COROMANDEL',
        'bse_code': None,
        'ir_url': None
    },
    'CRISIL': {
        'name': 'CRISIL Limited',
        'sector': 'Financial Services',
        'screener_slug': 'CRISIL',
        'bse_code': None,
        'ir_url': None
    },
    'CUB': {
        'name': 'City Union Bank',
        'sector': 'Banking',
        'screener_slug': 'CUB',
        'bse_code': None,
        'ir_url': None
    },
    'CYIENT': {
        'name': 'Cyient Limited',
        'sector': 'Information Technology',
        'screener_slug': 'CYIENT',
        'bse_code': None,
        'ir_url': None
    },
    'DEEPAKNTR': {
        'name': 'Deepak Nitrite',
        'sector': 'Chemicals',
        'screener_slug': 'DEEPAKNTR',
        'bse_code': None,
        'ir_url': None
    },
    'DIXON': {
        'name': 'Dixon Technologies',
        'sector': 'Electronics',
        'screener_slug': 'DIXON',
        'bse_code': None,
        'ir_url': None
    },
    'ESCORTS': {
        'name': 'Escorts Kubota',
        'sector': 'Automobile',
        'screener_slug': 'ESCORTS',
        'bse_code': None,
        'ir_url': None
    },
    'FACT': {
        'name': 'Fertilizers And Chemicals',
        'sector': 'Fertilizers',
        'screener_slug': 'FACT',
        'bse_code': None,
        'ir_url': None
    },
    'GAIL': {
        'name': 'GAIL India',
        'sector': 'Oil & Gas',
        'screener_slug': 'GAIL',
        'bse_code': None,
        'ir_url': None
    },
    'GLENMARK': {
        'name': 'Glenmark Pharmaceuticals',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'GLENMARK',
        'bse_code': None,
        'ir_url': None
    },
    'GRANULES': {
        'name': 'Granules India',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'GRANULES',
        'bse_code': None,
        'ir_url': None
    },
    'GRAPHITE': {
        'name': 'Graphite India',
        'sector': 'Capital Goods',
        'screener_slug': 'GRAPHITE',
        'bse_code': None,
        'ir_url': None
    },
    'GUJGASLTD': {
        'name': 'Gujarat Gas',
        'sector': 'Oil & Gas',
        'screener_slug': 'GUJGASLTD',
        'bse_code': None,
        'ir_url': None
    },
    'HFCL': {
        'name': 'HFCL Limited',
        'sector': 'Telecommunications',
        'screener_slug': 'HFCL',
        'bse_code': None,
        'ir_url': None
    },
    'HINDCOPPER': {
        'name': 'Hindustan Copper',
        'sector': 'Metals',
        'screener_slug': 'HINDCOPPER',
        'bse_code': None,
        'ir_url': None
    },
    'HINDPETRO': {
        'name': 'Hindustan Petroleum',
        'sector': 'Oil & Gas',
        'screener_slug': 'HINDPETRO',
        'bse_code': None,
        'ir_url': None
    },
    'HONAUT': {
        'name': 'Honeywell Automation',
        'sector': 'Capital Goods',
        'screener_slug': 'HONAUT',
        'bse_code': None,
        'ir_url': None
    },
    'IEX': {
        'name': 'Indian Energy Exchange',
        'sector': 'Financial Services',
        'screener_slug': 'IEX',
        'bse_code': None,
        'ir_url': None
    },
    'IGL': {
        'name': 'Indraprastha Gas',
        'sector': 'Oil & Gas',
        'screener_slug': 'IGL',
        'bse_code': None,
        'ir_url': None
    },
    'INDHOTEL': {
        'name': 'Indian Hotels',
        'sector': 'Hotels',
        'screener_slug': 'INDHOTEL',
        'bse_code': None,
        'ir_url': None
    },
    'INDUSTOWER': {
        'name': 'Indus Towers',
        'sector': 'Telecommunications',
        'screener_slug': 'INDUSTOWER',
        'bse_code': None,
        'ir_url': None
    },
    'INTELLECT': {
        'name': 'Intellect Design Arena',
        'sector': 'Technology',
        'screener_slug': 'INTELLECT',
        'bse_code': None,
        'ir_url': None
    },
    'IRCTC': {
        'name': 'Indian Railway Catering',
        'sector': 'Services',
        'screener_slug': 'IRCTC',
        'bse_code': None,
        'ir_url': None
    },
    'ISEC': {
        'name': 'ICICI Securities',
        'sector': 'Financial Services',
        'screener_slug': 'ISEC',
        'bse_code': None,
        'ir_url': None
    },
    'JINDALSTEL': {
        'name': 'Jindal Steel & Power',
        'sector': 'Steel',
        'screener_slug': 'JINDALSTEL',
        'bse_code': None,
        'ir_url': None
    },
    'JKCEMENT': {
        'name': 'JK Cement',
        'sector': 'Cement',
        'screener_slug': 'JKCEMENT',
        'bse_code': None,
        'ir_url': None
    },
    'JSWENERGY': {
        'name': 'JSW Energy',
        'sector': 'Power',
        'screener_slug': 'JSWENERGY',
        'bse_code': None,
        'ir_url': None
    },
    'KAJARIACER': {
        'name': 'Kajaria Ceramics',
        'sector': 'Building Materials',
        'screener_slug': 'KAJARIACER',
        'bse_code': None,
        'ir_url': None
    },
    'KEI': {
        'name': 'KEI Industries',
        'sector': 'Electricals',
        'screener_slug': 'KEI',
        'bse_code': None,
        'ir_url': None
    },
    'L&TFH': {
        'name': 'L&T Finance Holdings',
        'sector': 'Financial Services',
        'screener_slug': 'L&TFH',
        'bse_code': None,
        'ir_url': None
    },
    'LALPATHLAB': {
        'name': 'Dr Lal PathLabs',
        'sector': 'Healthcare',
        'screener_slug': 'LALPATHLAB',
        'bse_code': None,
        'ir_url': None
    },
    'LAURUSLABS': {
        'name': 'Laurus Labs',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'LAURUSLABS',
        'bse_code': None,
        'ir_url': None
    },
    'MANAPPURAM': {
        'name': 'Manappuram Finance',
        'sector': 'Financial Services',
        'screener_slug': 'MANAPPURAM',
        'bse_code': None,
        'ir_url': None
    },
    'MCX': {
        'name': 'Multi Commodity Exchange',
        'sector': 'Financial Services',
        'screener_slug': 'MCX',
        'bse_code': None,
        'ir_url': None
    },
    'METROBRAND': {
        'name': 'Metro Brands',
        'sector': 'Footwear',
        'screener_slug': 'METROBRAND',
        'bse_code': None,
        'ir_url': None
    },
    'MFSL': {
        'name': 'Max Financial Services',
        'sector': 'Insurance',
        'screener_slug': 'MFSL',
        'bse_code': None,
        'ir_url': None
    },
    'MGL': {
        'name': 'Mahanagar Gas',
        'sector': 'Oil & Gas',
        'screener_slug': 'MGL',
        'bse_code': None,
        'ir_url': None
    },
    'MINDTREE': {
        'name': 'Mindtree Limited',
        'sector': 'Information Technology',
        'screener_slug': 'MINDTREE',
        'bse_code': None,
        'ir_url': None
    },
    'MOTHERSON': {
        'name': 'Samvardhana Motherson',
        'sector': 'Automobile',
        'screener_slug': 'MOTHERSON',
        'bse_code': '517334',
        'ir_url': 'https://www.motherson.com/investors'
    },
    'MUTHOOTFIN': {
        'name': 'Muthoot Finance',
        'sector': 'Financial Services',
        'screener_slug': 'MUTHOOTFIN',
        'bse_code': None,
        'ir_url': None
    },
    'NATIONALUM': {
        'name': 'National Aluminium',
        'sector': 'Metals',
        'screener_slug': 'NATIONALUM',
        'bse_code': None,
        'ir_url': None
    },
    'NAUKRI': {
        'name': 'Info Edge India',
        'sector': 'Internet',
        'screener_slug': 'NAUKRI',
        'bse_code': None,
        'ir_url': None
    },
    'NAVINFLUOR': {
        'name': 'Navin Fluorine',
        'sector': 'Chemicals',
        'screener_slug': 'NAVINFLUOR',
        'bse_code': None,
        'ir_url': None
    },
    'NMDC': {
        'name': 'NMDC Limited',
        'sector': 'Mining',
        'screener_slug': 'NMDC',
        'bse_code': None,
        'ir_url': None
    },
    'OIL': {
        'name': 'Oil India',
        'sector': 'Oil & Gas',
        'screener_slug': 'OIL',
        'bse_code': None,
        'ir_url': None
    },
    'PAYTM': {
        'name': 'One 97 Communications',
        'sector': 'Financial Services',
        'screener_slug': 'PAYTM',
        'bse_code': None,
        'ir_url': None
    },
    'PEL': {
        'name': 'Piramal Enterprises',
        'sector': 'Financial Services',
        'screener_slug': 'PEL',
        'bse_code': None,
        'ir_url': None
    },
    'PETRONET': {
        'name': 'Petronet LNG',
        'sector': 'Oil & Gas',
        'screener_slug': 'PETRONET',
        'bse_code': None,
        'ir_url': None
    },
    'PFC': {
        'name': 'Power Finance Corporation',
        'sector': 'Financial Services',
        'screener_slug': 'PFC',
        'bse_code': None,
        'ir_url': None
    },
    'PHOENIXLTD': {
        'name': 'Phoenix Mills',
        'sector': 'Real Estate',
        'screener_slug': 'PHOENIXLTD',
        'bse_code': None,
        'ir_url': None
    },
    'PIIND': {
        'name': 'PI Industries',
        'sector': 'Chemicals',
        'screener_slug': 'PIIND',
        'bse_code': None,
        'ir_url': None
    },
    'POLYCAB': {
        'name': 'Polycab India',
        'sector': 'Electricals',
        'screener_slug': 'POLYCAB',
        'bse_code': None,
        'ir_url': None
    },
    'PRESTIGE': {
        'name': 'Prestige Estates',
        'sector': 'Real Estate',
        'screener_slug': 'PRESTIGE',
        'bse_code': None,
        'ir_url': None
    },
    'RECLTD': {
        'name': 'REC Limited',
        'sector': 'Financial Services',
        'screener_slug': 'RECLTD',
        'bse_code': None,
        'ir_url': None
    },
    'SBICARD': {
        'name': 'SBI Cards',
        'sector': 'Financial Services',
        'screener_slug': 'SBICARD',
        'bse_code': None,
        'ir_url': None
    },
    'SOLARINDS': {
        'name': 'Solar Industries',
        'sector': 'Chemicals',
        'screener_slug': 'SOLARINDS',
        'bse_code': None,
        'ir_url': None
    },
    'SONACOMS': {
        'name': 'Sona BLW Precision',
        'sector': 'Automobile',
        'screener_slug': 'SONACOMS',
        'bse_code': None,
        'ir_url': None
    },
    'SRF': {
        'name': 'SRF Limited',
        'sector': 'Chemicals',
        'screener_slug': 'SRF',
        'bse_code': None,
        'ir_url': None
    },
    'STAR': {
        'name': 'Sterlite Technologies',
        'sector': 'Telecommunications',
        'screener_slug': 'STAR',
        'bse_code': None,
        'ir_url': None
    },
    'TATACOMM': {
        'name': 'Tata Communications',
        'sector': 'Telecommunications',
        'screener_slug': 'TATACOMM',
        'bse_code': None,
        'ir_url': None
    },
    'TATAELXSI': {
        'name': 'Tata Elxsi',
        'sector': 'Information Technology',
        'screener_slug': 'TATAELXSI',
        'bse_code': None,
        'ir_url': None
    },
    'TATAPOWER': {
        'name': 'Tata Power',
        'sector': 'Power',
        'screener_slug': 'TATAPOWER',
        'bse_code': None,
        'ir_url': None
    },
    'THERMAX': {
        'name': 'Thermax Limited',
        'sector': 'Capital Goods',
        'screener_slug': 'THERMAX',
        'bse_code': None,
        'ir_url': None
    },
    'TORNTPOWER': {
        'name': 'Torrent Power',
        'sector': 'Power',
        'screener_slug': 'TORNTPOWER',
        'bse_code': None,
        'ir_url': None
    },
    'UBL': {
        'name': 'United Breweries',
        'sector': 'Beverages',
        'screener_slug': 'UBL',
        'bse_code': None,
        'ir_url': None
    },
    'VEDL': {
        'name': 'Vedanta Limited',
        'sector': 'Metals',
        'screener_slug': 'VEDL',
        'bse_code': None,
        'ir_url': None
    },
    'ZOMATO': {
        'name': 'Zomato Limited',
        'sector': 'Consumer Services',
        'screener_slug': 'ZOMATO',
        'bse_code': None,
        'ir_url': None
    },
    'ZYDUSLIFE': {
        'name': 'Zydus Lifesciences',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'ZYDUSLIFE',
        'bse_code': None,
        'ir_url': None
    },

    # ========== HIGH-GROWTH SMALLCAP PICKS (50 stocks) ==========
    'AAVAS': {
        'name': 'Aavas Financiers',
        'sector': 'Financial Services',
        'screener_slug': 'AAVAS',
        'bse_code': None,
        'ir_url': None
    },
    'ANANDRATHI': {
        'name': 'Anand Rathi Wealth',
        'sector': 'Financial Services',
        'screener_slug': 'ANANDRATHI',
        'bse_code': None,
        'ir_url': None
    },
    'ANGELONE': {
        'name': 'Angel One',
        'sector': 'Financial Services',
        'screener_slug': 'ANGELONE',
        'bse_code': None,
        'ir_url': None
    },
    'ASIANHOTNR': {
        'name': 'Asian Hotels (North)',
        'sector': 'Hotels',
        'screener_slug': 'ASIANHOTNR',
        'bse_code': None,
        'ir_url': None
    },
    'BASF': {
        'name': 'BASF India',
        'sector': 'Chemicals',
        'screener_slug': 'BASF',
        'bse_code': None,
        'ir_url': None
    },
    'BLUESTARCO': {
        'name': 'Blue Star',
        'sector': 'Consumer Durables',
        'screener_slug': 'BLUESTARCO',
        'bse_code': None,
        'ir_url': None
    },
    'CAMS': {
        'name': 'CAMS',
        'sector': 'Financial Services',
        'screener_slug': 'CAMS',
        'bse_code': None,
        'ir_url': None
    },
    'CDSL': {
        'name': 'Central Depository Services',
        'sector': 'Financial Services',
        'screener_slug': 'CDSL',
        'bse_code': None,
        'ir_url': None
    },
    'CENTRALBK': {
        'name': 'Central Bank of India',
        'sector': 'Banking',
        'screener_slug': 'CENTRALBK',
        'bse_code': None,
        'ir_url': None
    },
    'CENTURYPLY': {
        'name': 'Century Plyboards',
        'sector': 'Building Materials',
        'screener_slug': 'CENTURYPLY',
        'bse_code': None,
        'ir_url': None
    },
    'CLEAN': {
        'name': 'Clean Science',
        'sector': 'Chemicals',
        'screener_slug': 'CLEAN',
        'bse_code': None,
        'ir_url': None
    },
    'CREDITACC': {
        'name': 'CreditAccess Grameen',
        'sector': 'Financial Services',
        'screener_slug': 'CREDITACC',
        'bse_code': None,
        'ir_url': None
    },
    'CSBBANK': {
        'name': 'CSB Bank',
        'sector': 'Banking',
        'screener_slug': 'CSBBANK',
        'bse_code': None,
        'ir_url': None
    },
    'DELTACORP': {
        'name': 'Delta Corp',
        'sector': 'Gaming',
        'screener_slug': 'DELTACORP',
        'bse_code': None,
        'ir_url': None
    },
    'DEVYANI': {
        'name': 'Devyani International',
        'sector': 'Consumer Services',
        'screener_slug': 'DEVYANI',
        'bse_code': None,
        'ir_url': None
    },
    'EQUITAS': {
        'name': 'Equitas Small Finance Bank',
        'sector': 'Banking',
        'screener_slug': 'EQUITAS',
        'bse_code': None,
        'ir_url': None
    },
    'FINPIPE': {
        'name': 'Fine Organic Industries',
        'sector': 'Chemicals',
        'screener_slug': 'FINPIPE',
        'bse_code': None,
        'ir_url': None
    },
    'FLUOROCHEM': {
        'name': 'Gujarat Fluorochemicals',
        'sector': 'Chemicals',
        'screener_slug': 'FLUOROCHEM',
        'bse_code': None,
        'ir_url': None
    },
    'GRINDWELL': {
        'name': 'Grindwell Norton',
        'sector': 'Capital Goods',
        'screener_slug': 'GRINDWELL',
        'bse_code': None,
        'ir_url': None
    },
    'HAPPSTMNDS': {
        'name': 'Happiest Minds',
        'sector': 'Information Technology',
        'screener_slug': 'HAPPSTMNDS',
        'bse_code': None,
        'ir_url': None
    },
    'HEMHINDUS': {
        'name': 'HEG Limited',
        'sector': 'Capital Goods',
        'screener_slug': 'HEMHINDUS',
        'bse_code': None,
        'ir_url': None
    },
    'IIFLWAM': {
        'name': 'IIFL Wealth Management',
        'sector': 'Financial Services',
        'screener_slug': 'IIFLWAM',
        'bse_code': None,
        'ir_url': None
    },
    'INDIAMART': {
        'name': 'IndiaMART InterMESH',
        'sector': 'Internet',
        'screener_slug': 'INDIAMART',
        'bse_code': None,
        'ir_url': None
    },
    'INDIANB': {
        'name': 'Indian Bank',
        'sector': 'Banking',
        'screener_slug': 'INDIANB',
        'bse_code': None,
        'ir_url': None
    },
    'JUBLPHARMA': {
        'name': 'Jubilant Pharmova',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'JUBLPHARMA',
        'bse_code': None,
        'ir_url': None
    },
    'JUSTDIAL': {
        'name': 'Just Dial',
        'sector': 'Internet',
        'screener_slug': 'JUSTDIAL',
        'bse_code': None,
        'ir_url': None
    },
    'KPITTECH': {
        'name': 'KPIT Technologies',
        'sector': 'Information Technology',
        'screener_slug': 'KPITTECH',
        'bse_code': None,
        'ir_url': None
    },
    'LATENTVIEW': {
        'name': 'Latent View Analytics',
        'sector': 'Information Technology',
        'screener_slug': 'LATENTVIEW',
        'bse_code': None,
        'ir_url': None
    },
    'LEMONTREE': {
        'name': 'Lemon Tree Hotels',
        'sector': 'Hotels',
        'screener_slug': 'LEMONTREE',
        'bse_code': None,
        'ir_url': None
    },
    'MAZDOCK': {
        'name': 'Mazagon Dock Shipbuilders',
        'sector': 'Defence',
        'screener_slug': 'MAZDOCK',
        'bse_code': None,
        'ir_url': None
    },
    'METROPOLIS': {
        'name': 'Metropolis Healthcare',
        'sector': 'Healthcare',
        'screener_slug': 'METROPOLIS',
        'bse_code': None,
        'ir_url': None
    },
    'MIDHANI': {
        'name': 'Mishra Dhatu Nigam',
        'sector': 'Defence',
        'screener_slug': 'MIDHANI',
        'bse_code': None,
        'ir_url': None
    },
    'NAZARA': {
        'name': 'Nazara Technologies',
        'sector': 'Gaming',
        'screener_slug': 'NAZARA',
        'bse_code': None,
        'ir_url': None
    },
    'NIACL': {
        'name': 'New India Assurance',
        'sector': 'Insurance',
        'screener_slug': 'NIACL',
        'bse_code': None,
        'ir_url': None
    },
    'NYKAA': {
        'name': 'FSN E-Commerce (Nykaa)',
        'sector': 'Retail',
        'screener_slug': 'NYKAA',
        'bse_code': None,
        'ir_url': None
    },
    'ORIENTELEC': {
        'name': 'Orient Electric',
        'sector': 'Electricals',
        'screener_slug': 'ORIENTELEC',
        'bse_code': None,
        'ir_url': None
    },
    'PARAS': {
        'name': 'Paras Defence',
        'sector': 'Defence',
        'screener_slug': 'PARAS',
        'bse_code': None,
        'ir_url': None
    },
    'PNBHOUSING': {
        'name': 'PNB Housing Finance',
        'sector': 'Financial Services',
        'screener_slug': 'PNBHOUSING',
        'bse_code': None,
        'ir_url': None
    },
    'POLICYBZR': {
        'name': 'PB Fintech',
        'sector': 'Financial Services',
        'screener_slug': 'POLICYBZR',
        'bse_code': None,
        'ir_url': None
    },
    'POONAWALLA': {
        'name': 'Poonawalla Fincorp',
        'sector': 'Financial Services',
        'screener_slug': 'POONAWALLA',
        'bse_code': None,
        'ir_url': None
    },
    'RAILTEL': {
        'name': 'RailTel Corporation',
        'sector': 'Telecommunications',
        'screener_slug': 'RAILTEL',
        'bse_code': None,
        'ir_url': None
    },
    'RATNAMANI': {
        'name': 'Ratnamani Metals',
        'sector': 'Metals',
        'screener_slug': 'RATNAMANI',
        'bse_code': None,
        'ir_url': None
    },
    'ROUTE': {
        'name': 'Route Mobile',
        'sector': 'Telecommunications',
        'screener_slug': 'ROUTE',
        'bse_code': None,
        'ir_url': None
    },
    'SAFARI': {
        'name': 'Safari Industries',
        'sector': 'Consumer Goods',
        'screener_slug': 'SAFARI',
        'bse_code': None,
        'ir_url': None
    },
    'SHYAMMETL': {
        'name': 'Shyam Metalics',
        'sector': 'Metals',
        'screener_slug': 'SHYAMMETL',
        'bse_code': None,
        'ir_url': None
    },
    'SIGNATURE': {
        'name': 'Signature Global',
        'sector': 'Real Estate',
        'screener_slug': 'SIGNATURE',
        'bse_code': None,
        'ir_url': None
    },
    'SYNGENE': {
        'name': 'Syngene International',
        'sector': 'Pharmaceuticals',
        'screener_slug': 'SYNGENE',
        'bse_code': None,
        'ir_url': None
    },
    'TANLA': {
        'name': 'Tanla Platforms',
        'sector': 'Telecommunications',
        'screener_slug': 'TANLA',
        'bse_code': None,
        'ir_url': None
    },
    'UCOBANK': {
        'name': 'UCO Bank',
        'sector': 'Banking',
        'screener_slug': 'UCOBANK',
        'bse_code': None,
        'ir_url': None
    },
    'UJJIVAN': {
        'name': 'Ujjivan Small Finance Bank',
        'sector': 'Banking',
        'screener_slug': 'UJJIVAN',
        'bse_code': None,
        'ir_url': None
    },
    'UTIAMC': {
        'name': 'UTI Asset Management',
        'sector': 'Financial Services',
        'screener_slug': 'UTIAMC',
        'bse_code': None,
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
                'screener_slug': clean_symbol.upper(),  # Default to uppercase ticker for slug
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
        self.symbol_map = mapping.copy()  # Replace instead of update? Or update? Let's update.
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


_default_mapper = None  # Initialize the global variable
