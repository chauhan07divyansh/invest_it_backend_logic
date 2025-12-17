import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import wraps
import math

# --- Integration Imports ---
import torch
import joblib
from hf_utils import get_all_models

from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

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
FRONTEND_URL = "https://sentiquant-frontend.onrender.com"  
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URL}})

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
        # 1. Initialize data provider FIRST
        self.data_provider = None
        self.symbol_mapper = None
        self._initialize_data_provider()
        
        # 2. INTEGRATION: Load Hugging Face Models
        self.models = {}
        self._load_hf_models()
        
        # 3. Then initialize trading systems
        self.swing_system = None
        self.position_system = None
        if SYSTEMS_AVAILABLE:
            self.initialize_systems()
        else:
            logger.warning("Trading systems not imported. API is in a degraded state.")
    
    def _initialize_data_provider(self):
        """Initialize the unified data provider with Fyers + Screener.in"""
        try:
            fyers_app_id = os.getenv('FYERS_APP_ID')
            fyers_access_token = os.getenv('FYERS_ACCESS_TOKEN')
            redis_url = os.getenv('REDIS_URL')
            
            if not fyers_app_id or not fyers_access_token:
                logger.error("❌ FYERS_APP_ID or FYERS_ACCESS_TOKEN not configured!")
                return
            
            self.symbol_mapper = SymbolMapper()
            logger.info(f"✅ SymbolMapper initialized with {len(self.symbol_mapper.get_all_symbols())} symbols")
            
            self.data_provider = StockDataProvider(
                fyers_app_id=fyers_app_id,
                fyers_access_token=fyers_access_token,
                symbol_mapper=self.symbol_mapper,
                redis_url=redis_url
            )
            logger.info("✅ StockDataProvider initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data provider: {e}")
            traceback.print_exc()

    def _load_hf_models(self):
        """Downloads and loads models from Hugging Face into memory."""
        try:
            logger.info("--- Starting Model Integration from Hugging Face ---")
            model_paths = get_all_models() 

            # Load PyTorch Model (.pth)
            if model_paths.get("best_model_fold_1.pth"):
                self.models['pytorch_model'] = torch.load(
                    model_paths["best_model_fold_1.pth"], 
                    map_location=torch.device('cpu')
                )
                self.models['pytorch_model'].eval()
                logger.info("✅ PyTorch model loaded into memory.")

            # Load Scikit-Learn/NLP Pipelines (.pkl / .joblib)
            if model_paths.get("sbert_rf_pipeline.pkl"):
                self.models['rf_pipeline'] = joblib.load(model_paths["sbert_rf_pipeline.pkl"])
                logger.info("✅ SBERT RF Pipeline loaded into memory.")

            if model_paths.get("sentiment_pipeline_chunking.joblib"):
                self.models['sentiment_pipeline'] = joblib.load(model_paths["sentiment_pipeline_chunking.joblib"])
                logger.info("✅ Sentiment pipeline loaded into memory.")

        except Exception as e:
            logger.error(f"❌ Critical Error loading models: {e}")

    def initialize_systems(self):
        """Initializes trading systems with data provider and loaded models."""
        logger.info("Initializing trading systems...")
        
        if not self.data_provider:
            logger.error("❌ Cannot initialize systems: Data provider not available")
            return
        
        try:
            # You can now pass self.models to these systems if they are set up to accept them
            self.swing_system = EnhancedSwingTradingSystem(
                data_provider=self.data_provider,
                models=self.models
            )
            logger.info("✅ EnhancedSwingTradingSystem initialized.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedSwingTradingSystem", exc_info=True)

        try:
            self.position_system = EnhancedPositionTradingSystem(
                data_provider=self.data_provider,
                models=self.models
            )
            logger.info("✅ EnhancedPositionTradingSystem initialized.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedPositionTradingSystem", exc_info=True)

    def _clean_fundamental_data(self, fundamentals):
        if not isinstance(fundamentals, dict): return {}
        cleaned_data = fundamentals.copy()
        if 'market_cap' in cleaned_data and isinstance(cleaned_data['market_cap'], (int, float)):
            cleaned_data['market_cap'] = f"{cleaned_data['market_cap']:,}"
        return cleaned_data

    def extract_targets_from_backend(self, result):
        try:
            targets = result.get('trading_plan', {}).get('targets', {})
            return {
                'target_1': float(targets.get('target_1', 0)) if targets.get('target_1') else 0,
                'target_2': float(targets.get('target_2', 0)) if targets.get('target_2') else 0,
                'target_3': float(targets.get('target_3', 0)) if targets.get('target_3') else 0,
            }
        except Exception:
            return {'target_1': 0, 'target_2': 0, 'target_3': 0}

    def generate_trading_plan(self, result, system_type):
        try:
            backend_plan = result.get('trading_plan', {})
            if not backend_plan: return self._fallback_trading_plan(result)

            targets = backend_plan.get('targets', {})
            current_price = result.get('current_price', 0)

            return {
                'signal': backend_plan.get('entry_signal', 'HOLD/WATCH'),
                'strategy': self._enhance_strategy_description(backend_plan.get('entry_strategy', ''), backend_plan.get('entry_signal', ''), system_type),
                'entry_price': f"Around {current_price:.2f}" if current_price > 0 else 'N/A',
                'stop_loss': f"{backend_plan.get('stop_loss', 0):.2f}" if backend_plan.get('stop_loss') else 'N/A',
                'target_1': f"{targets.get('target_1', 0):.2f}" if targets.get('target_1') else 'N/A',
                'target_2': f"{targets.get('target_2', 0):.2f}" if targets.get('target_2') else 'N/A',
                'target_3': f"{targets.get('target_3', 0):.2f}" if targets.get('target_3') else 'N/A',
                'trailing_stop_advice': backend_plan.get('trade_management_note', 'Move SL to breakeven after Target 1.'),
            }
        except Exception:
            return self._fallback_trading_plan(result)

    def _enhance_strategy_description(self, base_strategy, signal, system_type):
        sentiment = signal.lower()
        if "strong buy" in sentiment: prefix = f"High-conviction BUY for {system_type.lower()} trading."
        elif "buy" in sentiment: prefix = f"Solid BUY opportunity for {system_type.lower()} trading."
        else: prefix = f"Stance for {system_type.lower()} trading."
        return f"{prefix} {base_strategy}"

    def _fallback_trading_plan(self, result):
        return {'signal': 'UNAVAILABLE', 'strategy': 'Plan not available.', 'entry_price': 'N/A', 'stop_loss': 'N/A', 'target_price': 'N/A'}

    def format_analysis_response(self, result, system_type):
        if not result: return None
        score = result.get('swing_score' if system_type == 'Swing' else 'position_score', 0)
        current_price = result.get('current_price', 0)
        targets = self.extract_targets_from_backend(result)
        target_price = targets['target_2'] or targets['target_1'] or current_price
        
        grade = "D (Poor)"
        if score >= 80: grade = "A+ (Excellent)"
        elif score >= 70: grade = "A (Good)"
        elif score >= 60: grade = "B (Average)"
        elif score >= 50: grade = "C (Below Average)"

        return {
            'symbol': result.get('symbol', 'N/A'),
            'company_name': result.get('company_name', 'N/A'),
            'analysis_timestamp': datetime.now().isoformat(),
            'system_type': system_type,
            'overall_score': score,
            'investment_grade': grade,
            'current_price': current_price,
            'target_price': target_price,
            'potential_return': ((target_price - current_price) / current_price) * 100 if current_price > 0 else 0,
            'trading_plan': self.generate_trading_plan(result, system_type),
            'technical_indicators': {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in result.get('technical_indicators', {}).items()},
            'fundamentals': self._clean_fundamental_data(result.get('fundamentals', {})),
            'sentiment': result.get('sentiment', {}),
            'time_horizon': "1-4 weeks" if system_type == "Swing" else "6-18 months"
        }

    def _standardize_portfolio_keys(self, portfolio_list):
        if not portfolio_list: return []
        standardized = []
        for item in portfolio_list:
            standardized.append({
                'symbol': item.get('symbol') or item.get('ticker'),
                'company': item.get('company') or item.get('name') or item.get('company_name'),
                'score': item.get('score'),
                'price': item.get('price') or item.get('current_price') or item.get('ltp') or 0,
                'stop_loss': item.get('stop_loss') or item.get('sl') or 0,
                'risk': item.get('risk') or 0,
                'investment_amount': item.get('amount') or item.get('investment_amount'),
                'number_of_shares': item.get('shares') or item.get('qty') or 0,
                'percentage_allocation': item.get('percentage_allocation') or item.get('allocation') or 0
            })
        return standardized

    def generate_swing_portfolio(self, budget, risk_appetite):
        if not self.swing_system: raise ConnectionAbortedError('System not available')
        all_stocks = self.swing_system.get_all_stock_symbols()
        all_results = self.swing_system.analyze_stocks_parallel(all_stocks)
        filtered = self.swing_system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
        portfolio = self._standardize_portfolio_keys(self.swing_system.generate_portfolio_allocation(filtered, budget, risk_appetite))
        return self._wrap_portfolio_result(portfolio, budget)

    def generate_position_portfolio(self, budget, risk_appetite, time_period):
        if not self.position_system: raise ConnectionAbortedError('System not available')
        results = self.position_system.create_personalized_portfolio(risk_appetite, time_period, budget)
        p_data = results.get('portfolio', {})
        p_list = [{'symbol': k, **v} for k, v in p_data.items()] if isinstance(p_data, dict) else p_data
        portfolio = self._standardize_portfolio_keys(p_list)
        return self._wrap_portfolio_result(portfolio, budget)

    def _wrap_portfolio_result(self, portfolio, budget):
        total_allocated = sum(item.get('investment_amount', 0) for item in portfolio)
        avg_score = sum(item.get('score', 0) for item in portfolio) / len(portfolio) if portfolio else 0
        return {
            'portfolio': portfolio,
            'summary': {
                'total_budget': budget, 'total_allocated': total_allocated,
                'remaining_cash': budget - total_allocated, 'diversification': len(portfolio),
                'average_score': avg_score,
            }
        }

trading_api = TradingAPI()

# --- API Endpoints ---

@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    if not trading_api.swing_system: return jsonify({'success': False, 'error': 'System not available'}), 503
    stocks = trading_api.swing_system.get_all_stock_symbols()
    return jsonify({'success': True, 'data': {'stocks': stocks, 'total_count': len(stocks)}})

def analyze_stock(system_type, symbol):
    try:
        symbol = validate_symbol(symbol)
        system = getattr(trading_api, f"{system_type}_system")
        if not system: return jsonify({'success': False, 'error': 'System not available'}), 503
        
        func = getattr(system, f"analyze_{system_type}_trading_stock", None) or getattr(system, "analyze_swing_trading_stock", None)
        if not func: raise AttributeError(f"Analysis function not found for {system_type}")

        result = func(symbol)
        if not result: return jsonify({'success': False, 'error': f'Could not analyze {symbol}'}), 404
        
        return jsonify({'success': True, 'data': trading_api.format_analysis_response(result, system_type.capitalize())})
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/swing/<symbol>', methods=['GET'])
def analyze_swing_stock_endpoint(symbol): return analyze_stock('swing', symbol)

@app.route('/api/analyze/position/<symbol>', methods=['GET'])
def analyze_position_stock_endpoint(symbol): return analyze_stock('position', symbol)

@app.route('/api/portfolio/swing', methods=['POST'])
def create_swing_portfolio_endpoint():
    try:
        data = request.get_json()
        result = trading_api.generate_swing_portfolio(validate_budget(data.get('budget')), validate_risk_appetite(data.get('risk_appetite')))
        return jsonify({'success': True, 'data': result})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/portfolio/position', methods=['POST'])
def create_position_portfolio_endpoint():
    try:
        data = request.get_json()
        result = trading_api.generate_position_portfolio(validate_budget(data.get('budget')), validate_risk_appetite(data.get('risk_appetite')), validate_time_period(data.get('time_period')))
        return jsonify({'success': True, 'data': result})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        'api': 'healthy',
        'data_provider': trading_api.data_provider is not None,
        'models_loaded': len(trading_api.models) > 0,
        'fyers_configured': bool(os.getenv('FYERS_APP_ID')),
        'swing_system': trading_api.swing_system is not None,
        'position_system': trading_api.position_system is not None
    }
    all_healthy = all(status.values())
    return jsonify({'success': True, 'status': 'healthy' if all_healthy else 'degraded', 'components': status}), 200 if all_healthy else 503

@app.route('/api/disclaimer', methods=['GET'])
def get_disclaimer(): return jsonify({'success': True, 'data': SEBI_DISCLAIMER})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.getLogger('waitress').setLevel(logging.WARNING)
    logger.info(f"Starting production server on port {port}...")
    serve(app, host="0.0.0.0", port=port)
