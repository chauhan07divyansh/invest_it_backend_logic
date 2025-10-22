from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from functools import wraps
import traceback
import math

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
# Define the live URL of your React frontend
FRONTEND_URL = "https://sentiquant-frontend.onrender.com" 

# Allow requests *only* from your frontend URL
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URL}})
# --- End of CORS Setup ---


# --- Logging Configuration ---

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


# --- Simple In-Memory Cache ---
simple_cache = {}
CACHE_TIMEOUT = 300  # 5 minutes


def get_from_cache(key):
    if key in simple_cache:
        data, timestamp = simple_cache[key]
        if (datetime.now().timestamp() - timestamp) < CACHE_TIMEOUT:
            return data
        del simple_cache[key]
    return None


def set_cache(key, value):
    simple_cache[key] = (value, datetime.now().timestamp())


class TradingAPI:
    """Handles all trading logic and system interactions."""

    def __init__(self):
        self.swing_system = None
        self.position_system = None
        if SYSTEMS_AVAILABLE:
            self.initialize_systems()
        else:
            logger.warning("Trading systems not imported. API is in a degraded state.")

    def initialize_systems(self):
        """Initializes each trading system individually for better resilience."""
        logger.info("Initializing trading systems...")
        try:
            self.swing_system = EnhancedSwingTradingSystem()
            logger.info("✅ EnhancedSwingTradingSystem initialized successfully.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedSwingTradingSystem", exc_info=True)

        try:
            self.position_system = EnhancedPositionTradingSystem()
            logger.info("✅ EnhancedPositionTradingSystem initialized successfully.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedPositionTradingSystem", exc_info=True)

    def _clean_fundamental_data(self, fundamentals):
        """Sanitizes and corrects fundamental data points known to have issues."""
        if not isinstance(fundamentals, dict):
            return {}

        cleaned_data = fundamentals.copy()

        if 'debt_to_equity' in cleaned_data and isinstance(cleaned_data['debt_to_equity'], (int, float)) and \
                cleaned_data['debt_to_equity'] > 3.0:
            cleaned_data['debt_to_equity'] = round(cleaned_data['debt_to_equity'] / 100, 3)

        for key in ['market_cap', 'enterprise_value']:
            if key in cleaned_data and isinstance(cleaned_data[key], (int, float)):
                cleaned_data[key] = f"{cleaned_data[key]:,}"

        for key in ['operating_margin', 'profit_margin', 'revenue_growth', 'earnings_growth', 'dividend_yield']:
            if key in cleaned_data and isinstance(cleaned_data[key], float) and abs(cleaned_data[key]) < 1:
                cleaned_data[key] = f"{cleaned_data[key] * 100:.2f}%"

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

            # Format the complete trading plan
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
            # More robustly extract values with multiple fallbacks, mirroring frontend logic
            price = item.get('price') or item.get('current_price') or item.get('entry_price') or item.get(
                'avg_price') or item.get('ltp') or 0
            stop_loss = item.get('stop_loss') or item.get('stoploss') or item.get('sl') or item.get('stop') or 0
            alloc = item.get('percentage_allocation') or item.get('allocation_pct') or item.get(
                'alloc_percent') or item.get('allocation') or 0
            shares = item.get('number_of_shares') or item.get('shares') or item.get('qty') or item.get('quantity') or 0
            risk = item.get('risk') or item.get('risk_amount') or item.get('max_risk') or 0

            new_item = {
                'symbol': item.get('symbol') or item.get('ticker'),
                'company': item.get('company') or item.get('name'),
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
        all_results = self.swing_system.analyze_multiple_stocks(all_stocks)
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
    cache_key = "all_stocks"
    if cached := get_from_cache(cache_key):
        return jsonify({'success': True, 'data': cached})
    if not trading_api.swing_system:
        return jsonify({'success': False, 'error': 'Trading system not available'}), 503
    stocks = trading_api.swing_system.get_all_stock_symbols()
    result = {'stocks': stocks, 'total_count': len(stocks)}
    set_cache(cache_key, result)
    return jsonify({'success': True, 'data': result})


def analyze_stock(system_type, symbol):
    try:
        symbol = validate_symbol(symbol)
        cache_key = f"{system_type}_analysis_{symbol}"
        # if cached := get_from_cache(cache_key):
        #     return jsonify({'success': True, 'data': cached})
        system = getattr(trading_api, f"{system_type}_system")
        if not system:
            return jsonify(
                {'success': False, 'error': f'{system_type.capitalize()} trading system not available'}), 503
        analysis_func = getattr(system, f"analyze_{system_type}_trading_stock")
        result = analysis_func(symbol)
        if not result:
            return jsonify({'success': False, 'error': f'Could not analyze stock {symbol}'}), 404
        formatted_result = trading_api.format_analysis_response(result, system_type.capitalize())
        set_cache(cache_key, formatted_result)
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
        cache_key = f"compare_{symbol}"
        if cached := get_from_cache(cache_key):
            return jsonify({'success': True, 'data': cached})
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
        set_cache(cache_key, result)
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in compare/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == "__main__":
    from waitress import serve
    import os

    # Prevent duplicate logs in Render logs
    logging.getLogger('waitress').setLevel(logging.WARNING)

    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)

