from dotenv import load_dotenv
load_dotenv()
import os
import backoff
import redis
import json
import uuid
import smtplib
import concurrent.futures
from email.mime.text import MIMEText
from flask import Flask, request, jsonify, g, Blueprint
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import logging
from datetime import datetime, timedelta
import traceback

import jwt

# NEW-5: passlib with argon2 (bcrypt kept as upgrade fallback)
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(
        schemes=["argon2", "bcrypt"],
        deprecated="auto",          # bcrypt hashes auto-upgrade to argon2 on next login
        argon2__rounds=4,
        argon2__memory_cost=65536,
    )
    PASSLIB_AVAILABLE = True
except ImportError:
    import bcrypt
    PASSLIB_AVAILABLE = False

from services.data_providers.stock_data_provider import StockDataProvider

# ── Trading system import ─────────────────────────────────────────────────────
SYSTEMS_AVAILABLE = False
try:
    from services.trading.position_trading import EnhancedPositionTradingSystem
    from services.trading.swing_trading import EnhancedSwingTradingSystem
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Could not import trading systems: {e}. API running in degraded mode.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Database ──────────────────────────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///sentiquant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# ── DB Models ─────────────────────────────────────────────────────────────────

class User(db.Model):
    """Core user account."""
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(512), nullable=False)   # NEW-5: string (argon2 hash)
    # NEW-3: subscription plan
    plan          = db.Column(db.String(20), nullable=False, default='FREE')   # FREE | PRO | ENTERPRISE
    # NEW-6: email verification
    is_verified   = db.Column(db.Boolean, nullable=False, default=False)
    verify_token  = db.Column(db.String(64), nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    usages        = db.relationship('UserUsage', backref='user', lazy=True)


# NEW-2: Usage tracking table
class UserUsage(db.Model):
    """Per-request usage log for billing & analytics."""
    __tablename__ = 'user_usage'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    endpoint       = db.Column(db.String(120), nullable=False)
    symbol         = db.Column(db.String(20), nullable=True)
    cost_estimate  = db.Column(db.Float, nullable=False, default=0.0)
    response_ms    = db.Column(db.Integer, nullable=True)
    cache_hit      = db.Column(db.Boolean, nullable=False, default=False)
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow, index=True)


with app.app_context():
    db.create_all()


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "60 per hour"],
    storage_uri=os.getenv("LIMITER_STORAGE", "memory://")
)

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "https://sentiquant.org",
    "https://www.sentiquant.org",
    "https://sentiquant-frontend.onrender.com"
]
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# ── SEBI Disclaimer ───────────────────────────────────────────────────────────
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

# ── Plan limits (NEW-3) ───────────────────────────────────────────────────────
PLAN_LIMITS = {
    'FREE':       {'analyze': True,  'portfolio': True,  'compare': False, 'daily_calls': 10,  'portfolio_per_day': 1},
    'PRO':        {'analyze': True,  'portfolio': True,  'compare': True,  'daily_calls': 500, 'portfolio_per_day': 999},
    'ENTERPRISE': {'analyze': True,  'portfolio': True,  'compare': True,  'daily_calls': 999999, 'portfolio_per_day': 999},
}

# ── Redis cache ───────────────────────────────────────────────────────────────
CACHE_TIMEOUT  = 300    # 5 min for live requests
PRECOMPUTE_TTL = 86400  # 24 h for background-worker results

redis_client = None
simple_cache = {}

if os.getenv("REDIS_URL"):
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connected.")
    except Exception as e:
        logger.warning(f"Redis failed, using in-memory cache: {e}")


def get_from_cache(key: str):
    if redis_client:
        try:
            val = redis_client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning(f"Redis GET '{key}': {e}")
            return None
    if key in simple_cache:
        data, ts = simple_cache[key]
        if (datetime.now().timestamp() - ts) < CACHE_TIMEOUT:
            return data
        del simple_cache[key]
    return None


def set_cache(key: str, value, ttl: int = CACHE_TIMEOUT):
    if redis_client:
        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return
        except Exception as e:
            logger.warning(f"Redis SET '{key}': {e}")
    simple_cache[key] = (value, datetime.now().timestamp())


# ─────────────────────────────────────────────────────────────────────────────
# NEW-1: Background Precompute
# ─────────────────────────────────────────────────────────────────────────────
# The API checks the precompute cache first (key: "pre_{type}_{symbol}").
# If present → instant response, zero Fyers cost.
# worker.py refreshes these keys on a cron schedule.
# ThreadPoolExecutor provides on-demand warming for uncached symbols.
# ─────────────────────────────────────────────────────────────────────────────

# Executor used only for async email sending (register endpoint)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def get_precomputed(system_type: str, symbol: str):
    """Read precomputed result written by worker.py. Returns None if not yet available."""
    return get_from_cache(f"pre_{system_type}_{symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW-4: @backoff retry wrapper for external API calls
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_with_retry(func, *args, **kwargs):
    """Wraps any external call with exponential backoff (max 3 tries)."""
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"[backoff] Retry #{d['tries']} for {func.__name__} "
            f"after {d['wait']:.1f}s — {d['exception']}"
        )
    )
    def _inner():
        return func(*args, **kwargs)
    return _inner()


# ─────────────────────────────────────────────────────────────────────────────
# NEW-5: Password helpers (passlib argon2 / bcrypt fallback)
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    if PASSLIB_AVAILABLE:
        return pwd_context.hash(plain)
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if PASSLIB_AVAILABLE:
        return pwd_context.verify(plain, hashed)
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def password_needs_rehash(hashed: str) -> bool:
    if PASSLIB_AVAILABLE:
        return pwd_context.needs_update(hashed)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# NEW-6: Email verification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _send_verification_email(to_email: str, token: str):
    """
    Sends a verification link via SMTP.
    Required .env vars: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, APP_BASE_URL
    Falls back to logging the link if SMTP is not configured (dev mode).
    """
    base_url = os.getenv("APP_BASE_URL", "http://localhost:5000")
    link = f"{base_url}/api/v1/auth/verify-email?token={token}"

    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        logger.info(f"[DEV] Email verification link for {to_email}: {link}")
        return

    msg = MIMEText(
        f"Welcome to SentiQuant!\n\nVerify your email by clicking:\n{link}\n\n"
        f"This link expires in 24 hours.",
        "plain"
    )
    msg["Subject"] = "Verify your SentiQuant account"
    msg["From"]    = os.getenv("SMTP_USER")
    msg["To"]      = to_email

    try:
        with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            server.sendmail(msg["From"], [to_email], msg.as_string())
        logger.info(f"Verification email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send verification email to {to_email}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Request lifecycle hooks
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def attach_request_id():
    g.request_id    = str(uuid.uuid4())[:8]
    g.user_email    = "anon"
    g.user_id       = None
    g.user_plan     = "FREE"
    g.cache_hit     = False
    g.current_symbol= None
    g.request_start = datetime.now()


def log_context():
    return f"[{getattr(g, 'request_id', '?')}] [{getattr(g, 'user_email', 'anon')}]"


# NEW-2: Usage tracking + disclaimer injection on every response
@app.after_request
def track_usage_and_disclaimer(response):
    # Inject SEBI disclaimer into successful JSON responses
    if response.content_type == 'application/json' and response.status_code == 200:
        try:
            data = response.get_json()
            if isinstance(data, dict) and 'disclaimer' not in data:
                data['disclaimer'] = SEBI_DISCLAIMER
                response.data = jsonify(data).data
        except Exception:
            pass

    # Log usage for all /api/ routes
    if request.path.startswith('/api/') and request.method in ('GET', 'POST'):
        try:
            elapsed_ms = int((datetime.now() - g.request_start).total_seconds() * 1000) \
                         if hasattr(g, 'request_start') else None
            cost = 0.0 if getattr(g, 'cache_hit', False) else _estimate_cost(request.path)
            usage = UserUsage(
                user_id      = getattr(g, 'user_id', None),
                endpoint     = request.path,
                symbol       = getattr(g, 'current_symbol', None),
                cost_estimate= cost,
                response_ms  = elapsed_ms,
                cache_hit    = getattr(g, 'cache_hit', False),
            )
            db.session.add(usage)
            db.session.commit()
        except Exception as e:
            logger.debug(f"Usage tracking failed: {e}")

    return response


def _estimate_cost(path: str) -> float:
    if '/analyze/' in path or '/compare/' in path:
        return 1.0
    if '/portfolio/' in path:
        return 2.0
    return 0.1


# ─────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_token(token):
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("Server misconfigured: JWT_SECRET is missing")
    return jwt.decode(token, secret, algorithms=["HS256"],
                      options={"verify_aud": False, "verify_iss": False})


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(" ")[1]
            except IndexError:
                return jsonify({'success': False, 'error': 'Malformed authorization header'}), 401
        if not token:
            return jsonify({'success': False, 'error': 'Token is missing'}), 401
        try:
            payload = decode_token(token)
            if payload.get('type') == 'refresh':
                return jsonify({'success': False, 'error': 'Refresh token cannot be used here'}), 401
            request.user = payload
            g.user_email = payload.get('email', 'anon')
            g.user_id    = payload.get('user_id')
            g.user_plan  = payload.get('plan', 'FREE')
        except RuntimeError as e:
            return jsonify({'success': False, 'error': str(e)}), 500
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        except Exception as e:
            logger.warning(f"{log_context()} JWT ERROR: {e}")
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# NEW-3: Subscription enforcement decorator
# ─────────────────────────────────────────────────────────────────────────────

def plan_required(feature: str):
    """
    @plan_required('portfolio') blocks FREE users from portfolio endpoints.
    Must be placed AFTER @token_required.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            plan  = getattr(g, 'user_plan', 'FREE')
            perms = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE'])
            if not perms.get(feature, False):
                return jsonify({
                    'success': False,
                    'error':   f"Your {plan} plan does not include '{feature}'. "
                               f"Upgrade at sentiquant.org/pricing."
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Systems guard decorator
# ─────────────────────────────────────────────────────────────────────────────

def require_systems(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not SYSTEMS_AVAILABLE:
            return jsonify({
                'success': False,
                'error':   'Trading systems unavailable. See /api/v1/system-status.'
            }), 503
        return f(*args, **kwargs)
    return decorated


def check_portfolio_daily_limit(f):
    """
    Enforces per-day portfolio generation limit based on plan.
    FREE: 1 portfolio/day   PRO/ENTERPRISE: unlimited
    Must be placed AFTER @token_required.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        plan      = getattr(g, 'user_plan', 'FREE')
        user_id   = getattr(g, 'user_id', None)
        limit     = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE']).get('portfolio_per_day', 1)

        if limit < 999 and user_id:
            try:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                count = db.session.query(UserUsage).filter(
                    UserUsage.user_id   == user_id,
                    UserUsage.endpoint.like('%/portfolio/%'),
                    UserUsage.timestamp >= today_start,
                    UserUsage.cache_hit == False
                ).count()

                if count >= limit:
                    return jsonify({
                        'success': False,
                        'error':   f"FREE plan allows {limit} portfolio generation per day. "
                                   f"You have used {count}/{limit} today. "
                                   f"Upgrade to PRO at sentiquant.org/pricing for unlimited portfolios."
                    }), 429
            except Exception as e:
                logger.warning(f"{log_context()} Portfolio limit check failed: {e}")

        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

NSE_SYMBOLS = set()
try:
    with open("nse_symbols.txt") as f:
        NSE_SYMBOLS = {l.strip().upper() for l in f if l.strip()}
    logger.info(f"✅ Loaded {len(NSE_SYMBOLS)} NSE symbols.")
except FileNotFoundError:
    logger.warning("nse_symbols.txt missing — symbol allowlist DISABLED.")


def validate_symbol(symbol):
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.upper().strip()
    if NSE_SYMBOLS and symbol not in NSE_SYMBOLS:
        raise ValueError(f"'{symbol}' is not a recognised NSE symbol")
    return symbol


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


def validate_time_period(tp):
    try:
        tp = int(tp)
        if tp not in [9, 18, 36, 60]:
            raise ValueError("Time period must be 9, 18, 36, or 60 months")
        return tp
    except (TypeError, ValueError):
        raise ValueError("Time period must be a valid integer")


# ─────────────────────────────────────────────────────────────────────────────
# TradingAPI class
# ─────────────────────────────────────────────────────────────────────────────

class TradingAPI:
    def __init__(self):
        self.swing_system    = None
        self.position_system = None
        if SYSTEMS_AVAILABLE:
            self.initialize_systems()
        else:
            logger.warning("Trading systems not imported. API is in degraded state.")

    def initialize_systems(self):
        logger.info("Initializing trading systems...")
        from services.data_providers.symbol_mapper import SymbolMapper
        symbol_mapper = SymbolMapper()
        data_provider = StockDataProvider(
            fyers_app_id=os.getenv("FYERS_APP_ID", "YOUR_APP_ID"),
            fyers_access_token=os.getenv("FYERS_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN"),
            symbol_mapper=symbol_mapper,
            redis_url=os.getenv("REDIS_URL", None)
        )
        try:
            self.swing_system = EnhancedSwingTradingSystem(data_provider=data_provider)
            logger.info("✅ SwingTradingSystem ready.")
        except Exception:
            logger.critical("❌ SwingTradingSystem init failed", exc_info=True)
        try:
            self.position_system = EnhancedPositionTradingSystem(data_provider=data_provider)
            logger.info("✅ PositionTradingSystem ready.")
        except Exception:
            logger.critical("❌ PositionTradingSystem init failed", exc_info=True)

    def _clean_fundamental_data(self, fundamentals):
        if not isinstance(fundamentals, dict):
            return {}
        cleaned = fundamentals.copy()
        if 'debt_to_equity' in cleaned and isinstance(cleaned['debt_to_equity'], (int, float)) \
                and cleaned['debt_to_equity'] > 3.0:
            cleaned['debt_to_equity'] = round(cleaned['debt_to_equity'] / 100, 3)
        for key in ['market_cap', 'enterprise_value']:
            if key in cleaned and isinstance(cleaned[key], (int, float)):
                cleaned[key] = f"{cleaned[key]:,}"
        for key in ['operating_margin', 'profit_margin', 'revenue_growth', 'earnings_growth', 'dividend_yield']:
            if key in cleaned and isinstance(cleaned[key], float) and abs(cleaned[key]) < 1:
                cleaned[key] = f"{cleaned[key] * 100:.2f}%"
        return cleaned

    def extract_targets_from_backend(self, result):
        try:
            targets = result.get('trading_plan', {}).get('targets', {})
            return {
                'target_1': float(targets.get('target_1', 0)) if targets.get('target_1') else 0,
                'target_2': float(targets.get('target_2', 0)) if targets.get('target_2') else 0,
                'target_3': float(targets.get('target_3', 0)) if targets.get('target_3') else 0,
            }
        except Exception as e:
            logger.warning(f"{log_context()} Error extracting targets: {e}")
            return {'target_1': 0, 'target_2': 0, 'target_3': 0}

    def generate_trading_plan(self, result, system_type):
        try:
            backend_plan = result.get('trading_plan', {})
            if not backend_plan:
                return self._fallback_trading_plan(result)
            entry_signal   = backend_plan.get('entry_signal', 'HOLD/WATCH')
            entry_strategy = backend_plan.get('entry_strategy', 'Wait for clearer signals')
            stop_loss      = backend_plan.get('stop_loss', 0)
            targets        = backend_plan.get('targets', {})
            mgmt_note      = backend_plan.get('trade_management_note', '')
            current_price  = result.get('current_price', 0)
            return {
                'signal':              entry_signal,
                'strategy':            self._enhance_strategy_description(entry_strategy, entry_signal, system_type),
                'entry_price':         f"Around {current_price:.2f}" if current_price > 0 else 'N/A',
                'stop_loss':           f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) and stop_loss > 0 else 'N/A',
                'target_1':            f"{targets.get('target_1', 0):.2f}" if targets.get('target_1') else 'N/A',
                'target_2':            f"{targets.get('target_2', 0):.2f}" if targets.get('target_2') else 'N/A',
                'target_3':            f"{targets.get('target_3', 0):.2f}" if targets.get('target_3') else 'N/A',
                'trailing_stop_advice': mgmt_note or 'Consider moving stop loss to breakeven after hitting Target 1.',
            }
        except Exception as e:
            logger.error(f"{log_context()} Error generating trading plan: {e}")
            return self._fallback_trading_plan(result)

    def _enhance_strategy_description(self, base_strategy, signal, system_type):
        s = signal.lower()
        if "strong buy" in s:
            return f"A high-conviction BUY signal for {system_type.lower()} trading. {base_strategy}"
        elif "buy" in s:
            return f"A solid BUY opportunity for {system_type.lower()} trading. {base_strategy}"
        elif "hold" in s:
            return f"Neutral stance recommended. {base_strategy}"
        elif "sell" in s or "avoid" in s:
            return f"Caution advised. {base_strategy}"
        return base_strategy

    def _fallback_trading_plan(self, result):
        cp = result.get('current_price', 0)
        return {
            'signal':      'UNAVAILABLE',
            'strategy':    'Trading plan not available from backend system.',
            'entry_price': f"Around {cp:.2f}" if cp > 0 else 'N/A',
            'stop_loss':   'N/A',
            'target_price':'N/A',
        }

    def format_analysis_response(self, result, system_type):
        if not result:
            return None
        score_key    = 'swing_score' if system_type == 'Swing' else 'position_score'
        score        = result.get(score_key, 0)
        current_price= result.get('current_price', 0)
        all_targets  = self.extract_targets_from_backend(result)
        target_price = all_targets.get('target_2', 0) or all_targets.get('target_1', current_price)
        potential_ret= ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
        grade = ("A+ (Excellent)" if score >= 80 else "A (Good)" if score >= 70
                 else "B (Average)" if score >= 60 else "C (Below Average)" if score >= 50 else "D (Poor)")
        trading_plan   = self.generate_trading_plan(result, system_type)
        sentiment_data = result.get('sentiment', {})
        if system_type == 'Position' and result.get('mda_analysis'):
            sentiment_data['mda_tone']  = result['mda_analysis'].get('tone')
            sentiment_data['mda_score'] = result['mda_analysis'].get('score')
        cleaned_fundamentals = self._clean_fundamental_data(result.get('fundamentals', {}))
        final_technicals = {
            k: (round(v, 2) if isinstance(v, (int, float)) else v)
            for k, v in result.get('technical_indicators', {}).items()
        }
        return {
            'symbol':               result.get('symbol', 'N/A'),
            'company_name':         result.get('company_name', 'N/A'),
            'analysis_timestamp':   datetime.now().isoformat(),
            'system_type':          system_type,
            'overall_score':        score,
            'investment_grade':     grade,
            'current_price':        current_price,
            'target_price':         target_price,
            'potential_return':     potential_ret,
            'trading_plan':         trading_plan,
            'technical_indicators': final_technicals,
            'fundamentals':         cleaned_fundamentals,
            'sentiment':            sentiment_data,
            'time_horizon':         "1-4 weeks" if system_type == "Swing" else "6-18 months",
        }

    def _standardize_portfolio_keys(self, portfolio_list):
        if not portfolio_list:
            return []
        out = []
        for item in portfolio_list:
            price  = (item.get('price') or item.get('current_price') or item.get('entry_price')
                      or item.get('avg_price') or item.get('ltp') or 0)
            sl     = item.get('stop_loss') or item.get('stoploss') or item.get('sl') or item.get('stop') or 0
            alloc  = (item.get('percentage_allocation') or item.get('allocation_pct')
                      or item.get('alloc_percent') or item.get('allocation') or 0)
            shares = (item.get('number_of_shares') or item.get('shares')
                      or item.get('qty') or item.get('quantity') or 0)
            risk   = item.get('risk') or item.get('risk_amount') or item.get('max_risk') or 0
            out.append({
                'symbol':                item.get('symbol') or item.get('ticker'),
                'company':               item.get('company') or item.get('name') or item.get('company_name'),
                'score':                 item.get('score'),
                'price':                 price,
                'stop_loss':             sl,
                'risk':                  risk,
                'investment_amount':     item.get('amount') or item.get('investment_amount'),
                'number_of_shares':      shares,
                'percentage_allocation': alloc,
            })
        return out

    def generate_swing_portfolio(self, budget, risk_appetite):
        if not self.swing_system:
            raise ConnectionAbortedError('Swing trading system not available')
        all_stocks     = _fetch_with_retry(self.swing_system.get_all_stock_symbols)   # NEW-4
        all_results = self.swing_system.analyze_stocks_parallel(all_stocks)
        filtered       = self.swing_system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
        portfolio_list = self.swing_system.generate_portfolio_allocation(filtered, budget, risk_appetite)
        std        = self._standardize_portfolio_keys(portfolio_list)
        total_alloc= sum(i.get('investment_amount', 0) for i in std)
        avg_score  = sum(i.get('score', 0) for i in std) / len(std) if std else 0
        return {'portfolio': std, 'summary': {
            'total_budget': budget, 'total_allocated': total_alloc,
            'remaining_cash': budget - total_alloc, 'diversification': len(std), 'average_score': avg_score}}

    def generate_position_portfolio(self, budget, risk_appetite, time_period):
        if not self.position_system:
            raise ConnectionAbortedError('Position trading system not available')
        results        = _fetch_with_retry(                                            # NEW-4
            self.position_system.create_personalized_portfolio, risk_appetite, time_period, budget)
        portfolio_data = results.get('portfolio', {})
        portfolio_list = ([{**v, 'symbol': k} for k, v in portfolio_data.items()]
                          if isinstance(portfolio_data, dict) else portfolio_data)
        std        = self._standardize_portfolio_keys(portfolio_list)
        total_alloc= sum(i.get('investment_amount', 0) for i in std)
        avg_score  = sum(i.get('score', 0) for i in std) / len(std) if std else 0
        return {'portfolio': std, 'summary': {
            'total_budget': budget, 'total_allocated': total_alloc,
            'remaining_cash': budget - total_alloc, 'diversification': len(std), 'average_score': avg_score}}


trading_api = TradingAPI()


# ─────────────────────────────────────────────────────────────────────────────
# Blueprint v1
# ─────────────────────────────────────────────────────────────────────────────

v1 = Blueprint('v1', __name__, url_prefix='/api/v1')


# ── Auth ──────────────────────────────────────────────────────────────────────

@v1.route('/auth/register', methods=['POST'])
@limiter.limit("10 per hour")
def register():
    data     = request.get_json() or {}
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password required'}), 400
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'User already exists'}), 409

    hashed       = hash_password(password)                  # NEW-5: argon2
    verify_token = uuid.uuid4().hex                         # NEW-6
    user = User(email=email, password_hash=hashed, verify_token=verify_token)
    db.session.add(user)
    db.session.commit()

    _executor.submit(_send_verification_email, email, verify_token)  # NEW-6: async email

    logger.info(f"{log_context()} Registered: {email}")
    return jsonify({
        'success': True,
        'message': 'Registered successfully. Please check your email to verify your account.'
    }), 201


@v1.route('/auth/verify-email', methods=['GET'])
def verify_email():
    """NEW-6: Handles the link clicked in the verification email."""
    token = request.args.get('token', '')
    if not token:
        return jsonify({'success': False, 'error': 'Token required'}), 400
    user = User.query.filter_by(verify_token=token).first()
    if not user:
        return jsonify({'success': False, 'error': 'Invalid or expired token'}), 400
    user.is_verified  = True
    user.verify_token = None
    db.session.commit()
    return jsonify({'success': True, 'message': 'Email verified. You can now log in.'})


@v1.route('/auth/login', methods=['POST'])
@limiter.limit("20 per hour")
def login():
    data     = request.get_json() or {}
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')
    secret   = os.getenv("JWT_SECRET")
    if not secret:
        logger.critical("JWT_SECRET not set!")
        return jsonify({'success': False, 'error': 'Server misconfigured'}), 500

    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):   # NEW-5
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    if not user.is_verified:                                             # NEW-6
        return jsonify({
            'success': False,
            'error':   'Please verify your email before logging in.'
        }), 403

    # NEW-5: transparent argon2 upgrade on login
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        db.session.commit()
        logger.info(f"Password upgraded to argon2 for {email}")

    # NEW-3: embed plan in JWT so decorators can read it without a DB query
    token_payload = {
        'email':   email,
        'user_id': user.id,
        'plan':    user.plan,
    }
    access_token  = jwt.encode({**token_payload, 'type': 'access',
                                 'exp': datetime.utcnow() + timedelta(minutes=15)},
                                secret, algorithm="HS256")
    refresh_token = jwt.encode({**token_payload, 'type': 'refresh',
                                 'exp': datetime.utcnow() + timedelta(days=30)},
                                secret, algorithm="HS256")

    logger.info(f"{log_context()} Login: {email} (plan={user.plan})")
    return jsonify({
        'success':       True,
        'access_token':  access_token,
        'refresh_token': refresh_token,
        'expires_in':    900,
        'plan':          user.plan,
    })


@v1.route('/auth/refresh', methods=['POST'])
@limiter.limit("30 per hour")
def refresh_token_endpoint():
    data  = request.get_json() or {}
    token = data.get('refresh_token')
    if not token:
        return jsonify({'success': False, 'error': 'Refresh token required'}), 400
    secret = os.getenv("JWT_SECRET")
    if not secret:
        return jsonify({'success': False, 'error': 'Server misconfigured'}), 500
    try:
        payload = decode_token(token)
        if payload.get('type') != 'refresh':
            return jsonify({'success': False, 'error': 'Invalid token type'}), 401
        new_token = jwt.encode({
            'email': payload['email'], 'user_id': payload.get('user_id'),
            'plan': payload.get('plan', 'FREE'), 'type': 'access',
            'exp': datetime.utcnow() + timedelta(minutes=15)
        }, secret, algorithm="HS256")
        return jsonify({'success': True, 'access_token': new_token, 'expires_in': 900})
    except jwt.ExpiredSignatureError:
        return jsonify({'success': False, 'error': 'Refresh token expired. Please log in again.'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'success': False, 'error': 'Invalid refresh token'}), 401


# ── Trading endpoints ─────────────────────────────────────────────────────────

@v1.route('/stocks', methods=['GET'])
def get_all_stocks():
    cache_key = "all_stocks"
    if cached := get_from_cache(cache_key):
        g.cache_hit = True
        return jsonify({'success': True, 'data': cached})
    if not trading_api.swing_system:
        return jsonify({'success': False, 'error': 'Trading system not available'}), 503
    stocks = trading_api.swing_system.get_all_stock_symbols()
    result = {'stocks': stocks, 'total_count': len(stocks)}
    set_cache(cache_key, result)
    return jsonify({'success': True, 'data': result})


def analyze_stock(system_type: str, symbol: str):
    """
    3-tier cache lookup:
      1. Precomputed (pre_{type}_{symbol}) — written by worker.py 1×/day.
         After the morning run every symbol is in here → zero Fyers cost all day.
      2. Short-term cache (5 min) — serves repeated hits between worker runs.
      3. Live Fyers call — only if worker hasn't run yet today (e.g. first boot,
         or a symbol added to nse_symbols.txt after today's run).
         Result is cached for 5 min so the next request is also free.
    """
    try:
        symbol = validate_symbol(symbol)
        g.current_symbol = symbol

        # 1. Precomputed by worker (fastest path — no Fyers call)
        precomputed = get_precomputed(system_type, symbol)
        if precomputed:
            g.cache_hit = True
            return jsonify({'success': True, 'data': precomputed, 'source': 'precomputed'})

        # 2. Short-term cache (e.g. same symbol requested twice within 5 min)
        cache_key = f"{system_type}_analysis_{symbol}"
        if cached := get_from_cache(cache_key):
            g.cache_hit = True
            return jsonify({'success': True, 'data': cached, 'source': 'cache'})

        # 3. Live Fyers call (fallback — worker covers all symbols so this is rare)
        system = getattr(trading_api, f"{system_type}_system")
        if not system:
            return jsonify({'success': False, 'error': f'{system_type.capitalize()} system unavailable'}), 503

        analysis_func = getattr(system, f"analyze_{system_type}_trading_stock")
        result        = _fetch_with_retry(analysis_func, symbol)

        if not result:
            return jsonify({'success': False, 'error': f'Could not analyze {symbol}'}), 404

        formatted = trading_api.format_analysis_response(result, system_type.capitalize())
        set_cache(cache_key, formatted)   # 5-min cache so repeated hits are free
        return jsonify({'success': True, 'data': formatted, 'source': 'live'})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} analyze/{system_type}/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@v1.route('/analyze/swing/<symbol>', methods=['GET'])
@require_systems
@limiter.limit("10 per minute")
def analyze_swing_stock_endpoint(symbol):
    return analyze_stock('swing', symbol)


@v1.route('/analyze/position/<symbol>', methods=['GET'])
@require_systems
@limiter.limit("10 per minute")
def analyze_position_stock_endpoint(symbol):
    return analyze_stock('position', symbol)


@v1.route('/portfolio/swing', methods=['POST'])
@token_required
@plan_required('portfolio')
@check_portfolio_daily_limit       # FREE: 1/day limit
@require_systems
@limiter.limit("5 per minute")
def create_swing_portfolio_endpoint():
    try:
        data   = request.get_json() or {}
        budget = validate_budget(data.get('budget'))
        risk   = validate_risk_appetite(data.get('risk_appetite'))
        result = trading_api.generate_swing_portfolio(budget, risk)
        return jsonify({'success': True, 'data': result})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} portfolio/swing: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@v1.route('/portfolio/position', methods=['POST'])
@token_required
@plan_required('portfolio')
@check_portfolio_daily_limit       # FREE: 1/day limit
@require_systems
@limiter.limit("5 per minute")
def create_position_portfolio_endpoint():
    try:
        data        = request.get_json() or {}
        budget      = validate_budget(data.get('budget'))
        risk        = validate_risk_appetite(data.get('risk_appetite'))
        time_period = validate_time_period(data.get('time_period'))
        result      = trading_api.generate_position_portfolio(budget, risk, time_period)
        return jsonify({'success': True, 'data': result})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} portfolio/position: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@v1.route('/compare/<symbol>', methods=['GET'])
@token_required
@plan_required('compare')          # NEW-3
@require_systems
@limiter.limit("10 per minute")
def compare_strategies_endpoint(symbol):
    try:
        symbol = validate_symbol(symbol)
        g.current_symbol = symbol
        cache_key = f"compare_{symbol}"
        if cached := get_from_cache(cache_key):
            g.cache_hit = True
            return jsonify({'success': True, 'data': cached})
        if not trading_api.swing_system or not trading_api.position_system:
            return jsonify({'success': False, 'error': 'One or more systems unavailable'}), 503
        swing_result    = _fetch_with_retry(trading_api.swing_system.analyze_swing_trading_stock, symbol)      # NEW-4
        position_result = _fetch_with_retry(trading_api.position_system.analyze_position_trading_stock, symbol)
        if not swing_result or not position_result:
            return jsonify({'success': False, 'error': f'Could not complete comparison for {symbol}'}), 404
        result = {
            'swing_analysis':    trading_api.format_analysis_response(swing_result,    'Swing'),
            'position_analysis': trading_api.format_analysis_response(position_result, 'Position'),
        }
        set_cache(cache_key, result)
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} compare/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ── Observability ─────────────────────────────────────────────────────────────

@v1.route('/system-status', methods=['GET'])
def system_status():
    status = {
        'trading_systems_imported': SYSTEMS_AVAILABLE,
        'swing_system_ready':       trading_api.swing_system is not None,
        'position_system_ready':    trading_api.position_system is not None,
        'fyers_configured':         bool(os.getenv('FYERS_APP_ID') and os.getenv('FYERS_ACCESS_TOKEN')),
        'redis_connected':          redis_client is not None,
        'nse_symbol_list_loaded':   len(NSE_SYMBOLS) > 0,
        'passlib_argon2':           PASSLIB_AVAILABLE,
        'smtp_configured':          bool(os.getenv('SMTP_HOST')),
        'database':                 'connected',
    }
    try:
        db.session.execute(db.text('SELECT 1'))
    except Exception:
        status['database'] = 'error'
    all_ready = all([status['trading_systems_imported'], status['swing_system_ready'],
                     status['position_system_ready'], status['fyers_configured'],
                     status['database'] == 'connected'])
    return jsonify({'success': True, 'overall': 'ready' if all_ready else 'degraded',
                    'components': status}), 200 if all_ready else 503


@v1.route('/health', methods=['GET'])
def health_check():
    return jsonify({'success': True, 'status': 'alive'}), 200


@v1.route('/disclaimer', methods=['GET'])
def get_disclaimer():
    return jsonify({'success': True, 'data': SEBI_DISCLAIMER})


# NEW-2: Admin usage analytics endpoint
@v1.route('/admin/usage', methods=['GET'])
@token_required
def usage_stats():
    since_days = int(request.args.get('days', 7))
    since      = datetime.utcnow() - timedelta(days=since_days)
    rows = (db.session.query(
                UserUsage.endpoint,
                db.func.count(UserUsage.id).label('calls'),
                db.func.sum(UserUsage.cost_estimate).label('total_cost'),
                db.func.avg(UserUsage.response_ms).label('avg_ms'),
                db.func.sum(db.cast(UserUsage.cache_hit, db.Integer)).label('cache_hits'))
            .filter(UserUsage.timestamp >= since)
            .group_by(UserUsage.endpoint)
            .order_by(db.desc('calls'))
            .all())
    data = [{
        'endpoint':       r.endpoint,
        'calls':          r.calls,
        'total_cost':     round(r.total_cost or 0, 2),
        'avg_response_ms':round(r.avg_ms or 0, 1),
        'cache_hit_rate': round((r.cache_hits or 0) / r.calls * 100, 1),
    } for r in rows]
    return jsonify({'success': True, 'data': data, 'period_days': since_days})


# ── Register blueprint + legacy aliases ──────────────────────────────────────

app.register_blueprint(v1)

@app.route('/api/stocks',                     methods=['GET'])
def legacy_stocks():         return get_all_stocks()

@app.route('/api/analyze/swing/<symbol>',    methods=['GET'])
def legacy_swing(symbol):    return analyze_stock('swing', symbol)

@app.route('/api/analyze/position/<symbol>', methods=['GET'])
def legacy_position(symbol): return analyze_stock('position', symbol)

@app.route('/api/health',                    methods=['GET'])
def legacy_health():         return jsonify({'success': True, 'status': 'alive'}), 200

@app.route('/api/disclaimer',               methods=['GET'])
def legacy_disclaimer():     return jsonify({'success': True, 'data': SEBI_DISCLAIMER})


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def rate_limited(_): return jsonify({'success': False, 'error': 'Rate limit exceeded. Please slow down.'}), 429

@app.errorhandler(500)
def server_error(_): return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from waitress import serve
    logging.getLogger('waitress').setLevel(logging.WARNING)
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Waitress on port {port}...")
    serve(app, host="0.0.0.0", port=port)