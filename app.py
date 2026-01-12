import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import wraps
import math

from flask import Flask, request, jsonify
from flask_cors import CORS
from data_providers import StockDataProvider
from symbol_mapper import SymbolMapper


# --- SEBI COMPLIANCE DISCLAIMER ---
SEBI_DISCLAIMER = {
    "text": (
        "The information, analysis, scores, recommendations, and signals provided by this application "
        "are generated algorithmically for informational and educational purposes only. They do NOT constitute "
        "investment advice, financial advice, portfolio management services, or research analysis as defined under "
        "SEBI regulations. We are NOT registered with SEBI as an Investment Adviser, Research Analyst, Portfolio "
        "Manager, or Stock Broker. Users must NOT rely solely on this information for making investment decisions. "
        "Stock market investments are subject to market risks, including the potential loss of principal. Users are "
        "strongly advised to conduct independent research and consult a SEBI-registered Investment Adviser or financial "
        "professional before making any investment decisions. Past performance is not indicative of future results. "
        "The application and its creators assume no liability for any financial losses incurred based on the use of "
        "this information."
    ),
    "version": "1.0",
    "last_updated": "2024-11-19"
}


# --- Graceful Import of Trading Systems ---
SYSTEMS_AVAILABLE = False
try:
  from systems.position_trading import EnhancedPositionTradingSystem
  from systems.swing_trading import EnhancedSwingTradingSystem

  SYSTEMS_AVAILABLE = True
  logging.info("Successfully imported trading system modules.")
except ImportError as e:
  logging.critical(f"Could not import trading systems: {e}. API will run in a degraded mode.")

app = Flask(__name__)

# --- Secure CORS Configuration ---
ALLOWED_ORIGINS = [
    "https://sentiquant.org",
    "https://www.sentiquant.org",
    "https://sentiquant-frontend.onrender.com"
]

CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Validation Functions ---
def validate_budget(budget):
  try:
    budget = float(budget)
    if not 10000 <= budget <= 10000000:
      raise ValueError("Budget must be between ₹10,000 and ₹10,000,000")
    return budget
  except (TypeError, ValueError):
    raise ValueError("Budget must be a valid number")


def validate_risk_appetite(risk):
  if not isinstance(risk, str) or risk.upper() not in ['LOW', 'MEDIUM', 'HIGH']:
    raise ValueError("Risk appetite must be LOW, MEDIUM, or HIGH")
  return risk.upper()


def validate_symbol(symbol):
  if not isinstance(symbol, str) or not symbol.strip():
    raise ValueError("Symbol must be a non-empty string")
  return symbol.upper().strip()


def validate_time_period(time_period):
  try:
    time_period = int(time_period)
    if time_period not in [9, 18, 36, 60]:
      raise ValueError("Time period must be 9, 18, 36, or 60 months")
    return time_period
  except (TypeError, ValueError):
    raise ValueError("Time period must be a valid integer")


# --- Global Disclaimer Injection ---
@app.after_request
def inject_disclaimer(response):
  """Automatically inject SEBI disclaimer into all JSON API responses"""
  if response.content_type == 'application/json' and response.status_code == 200:
    try:
      data = response.get_json()
      if isinstance(data, dict) and 'disclaimer' not in data:
        data['disclaimer'] = SEBI_DISCLAIMER
        response.data = jsonify(data).data
    except Exception as e:
      logger.debug(f"Could not inject disclaimer: {e}")
  return response


class TradingAPI:
  """Handles all trading logic and system interactions."""

  def __init__(self):
    # Initialize data provider FIRST
    self.data_provider = None
    self.symbol_mapper = None
    self._initialize_data_provider()
   
    # Then initialize trading systems
    self.swing_system = None
    self.position_system = None
    if SYSTEMS_AVAILABLE:
      self.initialize_systems()
    else:
      logger.warning("Trading systems not imported. API is in a degraded state.")
 
  def _initialize_data_provider(self):
    """Initialize the unified data provider with Fyers + Screener.in"""
    try:
      # Get credentials from environment
      fyers_app_id = os.getenv('FYERS_APP_ID')
      fyers_access_token = os.getenv('FYERS_ACCESS_TOKEN')
      redis_url = os.getenv('REDIS_URL')
     
      if not fyers_app_id or not fyers_access_token:
        logger.error("❌ FYERS_APP_ID or FYERS_ACCESS_TOKEN not configured!")
        logger.warning("Data fetching will fail. Please configure Fyers credentials.")
        return
     
      # Initialize symbol mapper
      self.symbol_mapper = SymbolMapper()
      logger.info(f"✅ SymbolMapper initialized with {len(self.symbol_mapper.get_all_symbols())} symbols")
     
      # Initialize unified data provider
      self.data_provider = StockDataProvider(
        fyers_app_id=fyers_app_id,
        fyers_access_token=fyers_access_token,
        symbol_mapper=self.symbol_mapper,
        redis_url=redis_url
      )
      logger.info("✅ StockDataProvider initialized successfully")
     
    except Exception as e:
      logger.error(f"❌ Failed to initialize data provider: {e}")
      import traceback
      traceback.print_exc()


  def initialize_systems(self):
    """Initializes each trading system individually with data provider injection."""
    logger.info("Initializing trading systems...")
   
    if not self.data_provider:
      logger.error("❌ Cannot initialize trading systems: Data provider not available")
      return
   
    try:
      self.swing_system = EnhancedSwingTradingSystem(data_provider=self.data_provider)
      logger.info("✅ EnhancedSwingTradingSystem initialized successfully.")
    except Exception:
      logger.critical("❌ FAILED to initialize EnhancedSwingTradingSystem", exc_info=True)

    try:
      self.position_system = EnhancedPositionTradingSystem(data_provider=self.data_provider)
      logger.info("✅ EnhancedPositionTradingSystem initialized successfully.")
    except Exception:
      logger.critical("❌ FAILED to initialize EnhancedPositionTradingSystem", exc_info=True)

  def _clean_fundamental_data(self, fundamentals):
    """Sanitizes fundamental data points for display."""
    if not isinstance(fundamentals, dict):
      return {}

    cleaned_data = fundamentals.copy()

    # Format Market Cap for readability
    key = 'market_cap'
    if key in cleaned_data and isinstance(cleaned_data[key], (int, float)):
      cleaned_data[key] = f"{cleaned_data[key]:,}"
   
    return cleaned_data

  def extract_targets_from_backend(self, result):
    """Extract all three targets from backend trading plan."""
    try:
      backend_plan = result.get('trading_plan', {})
      targets = backend_plan.get('targets', {})

      return {
        'target_1': float(targets.get('target_1', 0)) if targets.get('target_1') else 0,
        'target_2': float(targets.get('target_2', 0)) if targets.get('target_2') else 0,
        'target_3': float(targets.get('target_3', 0)) if targets.get('target_3') else 0,
      }
    except Exception as e:
      logger.warning(f"Error extracting targets from backend: {e}")
      return {'target_1': 0, 'target_2': 0, 'target_3': 0}

  def generate_trading_plan(self, result, system_type):
    """Generate trading plan using backend logic with all targets and trailing stops."""
    try:
      backend_plan = result.get('trading_plan', {})

      if not backend_plan:
        logger.warning("No backend trading plan found")
        return self._fallback_trading_plan(result)

      entry_signal = backend_plan.get('entry_signal', 'HOLD/WATCH')
      entry_strategy = backend_plan.get('entry_strategy', 'Wait for clearer signals')
      stop_loss = backend_plan.get('stop_loss', 0)
      targets = backend_plan.get('targets', {})
      trade_management_note = backend_plan.get('trade_management_note', '')

      current_price = result.get('current_price', 0)

      formatted_plan = {
        'signal': entry_signal,
        'strategy': self._enhance_strategy_description(entry_strategy, entry_signal, system_type),
        'entry_price': f"Around {current_price:.2f}" if current_price > 0 else 'N/A',
        'stop_loss': f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) and stop_loss > 0 else 'N/A',
        'target_1': f"{targets.get('target_1', 0):.2f}" if targets.get('target_1') else 'N/A',
        'target_2': f"{targets.get('target_2', 0):.2f}" if targets.get('target_2') else 'N/A',
        'target_3': f"{targets.get('target_3', 0):.2f}" if targets.get('target_3') else 'N/A',
        'trailing_stop_advice': trade_management_note if trade_management_note else 'Consider moving stop loss to breakeven after hitting Target 1.',
      }

      return formatted_plan

    except Exception as e:
      logger.error(f"Error generating trading plan: {e}")
      return self._fallback_trading_plan(result)

  def _enhance_strategy_description(self, base_strategy, signal, system_type):
    """Enhance the strategy description with context."""
    sentiment = signal.lower()

    if "strong buy" in sentiment:
      return f"A high-conviction BUY signal for {system_type.lower()} trading. {base_strategy}"
    elif "buy" in sentiment:
      return f"A solid BUY opportunity for {system_type.lower()} trading. {base_strategy}"
    elif "hold" in sentiment:
      return f"Neutral stance recommended. {base_strategy}"
    elif "sell" in sentiment or "avoid" in sentiment:
      return f"Caution advised. {base_strategy}"
    else:
      return base_strategy

  def _fallback_trading_plan(self, result):
    """Fallback trading plan if backend plan is unavailable."""
    current_price = result.get('current_price', 0)
    return {
      'signal': 'UNAVAILABLE',
      'strategy': 'Trading plan not available from backend system.',
      'entry_price': f"Around {current_price:.2f}" if current_price > 0 else 'N/A',
      'stop_loss': 'N/A',
      'target_price': 'N/A',
    }

  def format_analysis_response(self, result, system_type):
    """Format analysis response using backend data."""
    if not result:
      return None

    score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
    score = result.get(score_key, 0)
    current_price = result.get('current_price', 0)

    all_targets = self.extract_targets_from_backend(result)
    target_price = all_targets.get('target_2', 0)
    if target_price == 0:
      target_price = all_targets.get('target_1', current_price)

    potential_return = ((target_price - current_price) / current_price) * 100 if current_price > 0 else 0

    grade = "D (Poor)"
    if score >= 80:
      grade = "A+ (Excellent)"
    elif score >= 70:
      grade = "A (Good)"
    elif score >= 60:
      grade = "B (Average)"
    elif score >= 50:
      grade = "C (Below Average)"

    trading_plan = self.generate_trading_plan(result, system_type)

    sentiment_data = result.get('sentiment', {})
    if system_type == 'Position' and result.get('mda_analysis'):
      sentiment_data['mda_tone'] = result['mda_analysis'].get('tone')
      sentiment_data['mda_score'] = result['mda_analysis'].get('score')
   
    cleaned_fundamentals = self._clean_fundamental_data(result.get('fundamentals', {}))

    system_technicals = result.get('technical_indicators', {})
    final_technicals = {}
    for key, value in system_technicals.items():
      if isinstance(value, (int, float)):
        final_technicals[key] = round(value, 2)
      else:
        final_technicals[key] = value

    return {
      'symbol': result.get('symbol', 'N/A'),
      'company_name': result.get('company_name', 'N/A'),
      'analysis_timestamp': datetime.now().isoformat(),
      'system_type': system_type,
      'overall_score': score,
      'investment_grade': grade,
      'current_price': current_price,
      'target_price': target_price,
      'potential_return': potential_return,
      'trading_plan': trading_plan,
      'technical_indicators': final_technicals,
      'fundamentals': cleaned_fundamentals,
      'sentiment': sentiment_data,
      'time_horizon': "1-4 weeks" if system_type == "Swing" else "6-18 months"
    }

  def _standardize_portfolio_keys(self, portfolio_list):
    if not portfolio_list:
      return []

    standardized_list = []
    for item in portfolio_list:
      price = item.get('price') or item.get('current_price') or item.get('entry_price') or item.get(
        'avg_price') or item.get('ltp') or 0
      stop_loss = item.get('stop_loss') or item.get('stoploss') or item.get('sl') or item.get('stop') or 0
      alloc = item.get('percentage_allocation') or item.get('allocation_pct') or item.get(
        'alloc_percent') or item.get('allocation') or 0
      shares = item.get('number_of_shares') or item.get('shares') or item.get('qty') or item.get('quantity') or 0
      risk = item.get('risk') or item.get('risk_amount') or item.get('max_risk') or 0

      new_item = {
        'symbol': item.get('symbol') or item.get('ticker'),
        'company': item.get('company') or item.get('name') or item.get('company_name'),
        'score': item.get('score'),
        'price': price,
        'stop_loss': stop_loss,
        'risk': risk,
        'investment_amount': item.get('amount') or item.get('investment_amount'),
        'number_of_shares': shares,
        'percentage_allocation': alloc
      }
      standardized_list.append(new_item)
    return standardized_list

  def generate_swing_portfolio(self, budget, risk_appetite):
    if not self.swing_system:
      raise ConnectionAbortedError('Swing trading system not available')
    all_stocks = self.swing_system.get_all_stock_symbols()
    all_results = self.swing_system.analyze_stocks_parallel(all_stocks)
    filtered = self.swing_system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
    portfolio_list = self.swing_system.generate_portfolio_allocation(filtered, budget, risk_appetite)

    standardized_portfolio = self._standardize_portfolio_keys(portfolio_list)

    total_allocated = sum(item.get('investment_amount', 0) for item in standardized_portfolio)
    avg_score = sum(item.get('score', 0) for item in standardized_portfolio) / len(
      standardized_portfolio) if standardized_portfolio else 0

    return {
      'portfolio': standardized_portfolio,
      'summary': {
        'total_budget': budget,
        'total_allocated': total_allocated,
        'remaining_cash': budget - total_allocated,
        'diversification': len(standardized_portfolio),
        'average_score': avg_score,
      }
    }

  def generate_position_portfolio(self, budget, risk_appetite, time_period):
    if not self.position_system:
      raise ConnectionAbortedError('Position trading system not available')
    results = self.position_system.create_personalized_portfolio(risk_appetite, time_period, budget)
    portfolio_data = results.get('portfolio', {})
    portfolio_list = []
    if isinstance(portfolio_data, dict):
      for symbol, details in portfolio_data.items():
        details['symbol'] = symbol
        portfolio_list.append(details)
    else:
      portfolio_list = portfolio_data

    standardized_portfolio = self._standardize_portfolio_keys(portfolio_list)

    total_allocated = sum(item.get('investment_amount', 0) for item in standardized_portfolio)
    avg_score = sum(item.get('score', 0) for item in standardized_portfolio) / len(
      standardized_portfolio) if standardized_portfolio else 0
    return {
      'portfolio': standardized_portfolio,
      'summary': {
        'total_budget': budget,
        'total_allocated': total_allocated,
        'remaining_cash': budget - total_allocated,
        'diversification': len(standardized_portfolio),
        'average_score': avg_score,
      }
    }


trading_api = TradingAPI()


# --- API Endpoints ---
@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
  if not trading_api.swing_system:
    return jsonify({'success': False, 'error': 'Trading system not available'}), 503
  stocks = trading_api.swing_system.get_all_stock_symbols()
  result = {'stocks': stocks, 'total_count': len(stocks)}
  return jsonify({'success': True, 'data': result})


def analyze_stock(system_type, symbol):
  try:
    symbol = validate_symbol(symbol)
   
    system = getattr(trading_api, f"{system_type}_system")
    if not system:
      return jsonify(
        {'success': False, 'error': f'{system_type.capitalize()} trading system not available'}), 503
   
    analysis_func_name = f"analyze_{system_type}_trading_stock"
    if not hasattr(system, analysis_func_name):
      analysis_func_name = "analyze_swing_trading_stock"
      if not hasattr(system, analysis_func_name):
        raise AttributeError(f"Could not find analysis function for {system_type}")

    analysis_func = getattr(system, analysis_func_name)
   
    result = analysis_func(symbol)
   
    if not result:
      return jsonify({'success': False, 'error': f'Could not analyze stock {symbol}'}), 404
   
    formatted_result = trading_api.format_analysis_response(result, system_type.capitalize())
   
    return jsonify({'success': True, 'data': formatted_result})
 
  except ValueError as e:
    return jsonify({'success': False, 'error': str(e)}), 400
  except Exception as e:
    logger.error(f"Error in analyze/{system_type}/{symbol}: {e}\n{traceback.format_exc()}")
    return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/analyze/swing/<symbol>', methods=['GET'])
def analyze_swing_stock_endpoint(symbol):
  return analyze_stock('swing', symbol)


@app.route('/api/analyze/position/<symbol>', methods=['GET'])
def analyze_position_stock_endpoint(symbol):
  return analyze_stock('position', symbol)


@app.route('/api/portfolio/swing', methods=['POST'])
def create_swing_portfolio_endpoint():
  try:
    data = request.get_json()
    if not data:
      return jsonify({'success': False, 'error': 'Request body cannot be empty'}), 400
    budget = validate_budget(data.get('budget'))
    risk = validate_risk_appetite(data.get('risk_appetite'))
    result = trading_api.generate_swing_portfolio(budget, risk)
    return jsonify({'success': True, 'data': result})
  except (ValueError, KeyError) as e:
    return jsonify({'success': False, 'error': str(e)}), 400
  except Exception as e:
    logger.error(f"Error in portfolio/swing: {e}\n{traceback.format_exc()}")
    return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/portfolio/position', methods=['POST'])
def create_position_portfolio_endpoint():
  try:
    data = request.get_json()
    if not data:
      return jsonify({'success': False, 'error': 'Request body cannot be empty'}), 400
    budget = validate_budget(data.get('budget'))
    risk = validate_risk_appetite(data.get('risk_appetite'))
    time_period = validate_time_period(data.get('time_period'))
    result = trading_api.generate_position_portfolio(budget, risk, time_period)
    return jsonify({'success': True, 'data': result})
  except (ValueError, KeyError) as e:
    return jsonify({'success': False, 'error': str(e)}), 400
  except Exception as e:
    logger.error(f"Error in portfolio/position: {e}\n{traceback.format_exc()}")
    return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/compare/<symbol>', methods=['GET'])
def compare_strategies_endpoint(symbol):
        try:
                symbol = validate_symbol(symbol)

                if not trading_api.swing_system or not trading_api.position_system:
                        return jsonify(
                                {'success': False, 'error': 'One or more trading systems are unavailable'}), 503

                swing_result = trading_api.swing_system.analyze_swing_trading_stock(symbol)
                position_result = trading_api.position_system.analyze_position_trading_stock(symbol)

                if not swing_result or not position_result:
                        return jsonify(
                                {'success': False, 'error': f'Could not complete comparison for {symbol}'}), 404

                swing_formatted = trading_api.format_analysis_response(swing_result, 'Swing')
                position_formatted = trading_api.format_analysis_response(position_result, 'Position')
                result = {'swing_analysis': swing_formatted, 'position_analysis': position_formatted}

                return jsonify({'success': True, 'data': result})
        except ValueError as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
                logger.error(f"Error in compare/{symbol}: {e}\n{traceback.format_exc()}")
                return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
        """Health check endpoint to verify data provider status"""
        status = {
                'api': 'healthy',
                'data_provider': trading_api.data_provider is not None,
                'fyers_configured': bool(os.getenv('FYERS_APP_ID') and os.getenv('FYERS_ACCESS_TOKEN')),
                'redis_configured': bool(os.getenv('REDIS_URL')),
                'swing_system': trading_api.swing_system is not None,
                'position_system': trading_api.position_system is not None
        }

        all_healthy = all([
                status['data_provider'],
                status['fyers_configured'],
                status['swing_system'],
                status['position_system']
        ])

        return jsonify({
                'success': True,
                'status': 'healthy' if all_healthy else 'degraded',
                'components': status
        }), 200 if all_healthy else 503


@app.route('/api/disclaimer', methods=['GET'])
def get_disclaimer():
        """Dedicated endpoint to fetch the SEBI disclaimer"""
        return jsonify({
                'success': True,
                'data': SEBI_DISCLAIMER
        })


# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
         return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))

        try:
                from waitress import serve
                logging.getLogger('waitress').setLevel(logging.WARNING)
                logger.info(f"Starting production server with Waitress on port {port}...")
                serve(app, host="0.0.0.0", port=port)
        except ImportError:
                logger.warning("Waitress not found. Falling back to Flask dev server.")
                logger.warning("DO NOT use the Flask dev server in production.")
                app.run(host="0.0.0.0", port=port, debug=False)
