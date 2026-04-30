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
        deprecated="auto",
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
    __tablename__ = 'users'
    id                   = db.Column(db.Integer, primary_key=True)
    email                = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash        = db.Column(db.String(512), nullable=False)
    plan                 = db.Column(db.String(20), nullable=False, default='FREE')
    is_verified          = db.Column(db.Boolean, nullable=False, default=False)
    verify_token         = db.Column(db.String(64), nullable=True)
    # ── Account lockout (ported from Node.js auth) ────────────────────────────
    failed_login_attempts = db.Column(db.Integer, nullable=False, default=0)
    locked_until         = db.Column(db.DateTime, nullable=True)
    is_active            = db.Column(db.Boolean, nullable=False, default=True)
    created_at           = db.Column(db.DateTime, default=datetime.utcnow)
    usages               = db.relationship('UserUsage', backref='user', lazy=True)
    audit_logs           = db.relationship('LoginAudit', backref='user', lazy=True)


class UserUsage(db.Model):
    __tablename__ = 'user_usage'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    endpoint       = db.Column(db.String(120), nullable=False)
    symbol         = db.Column(db.String(20), nullable=True)
    cost_estimate  = db.Column(db.Float, nullable=False, default=0.0)
    response_ms    = db.Column(db.Integer, nullable=True)
    cache_hit      = db.Column(db.Boolean, nullable=False, default=False)
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow, index=True)


# ── Login audit log (ported from Node.js auth) ────────────────────────────────
class LoginAudit(db.Model):
    __tablename__ = 'login_audit'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    email          = db.Column(db.String(255), nullable=False)
    ip_address     = db.Column(db.String(64), nullable=True)
    user_agent     = db.Column(db.Text, nullable=True)
    status         = db.Column(db.String(20), nullable=False)   # success | failed | locked
    failure_reason = db.Column(db.String(100), nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow, index=True)


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
        "IMPORTANT: This platform provides AI-generated technical analysis for educational and informational purposes ONLY. "
        "Sentiquant is NOT registered with SEBI as an Investment Adviser, Research Analyst, Portfolio Manager, or Stock Broker. "
        "Nothing on this platform constitutes investment advice, a recommendation to buy or sell securities, portfolio management "
        "services, or research analysis as defined under SEBI (Investment Advisers) Regulations, 2013 or SEBI (Research Analysts) "
        "Regulations, 2014. All signals, scores, and technical reference levels shown are algorithmic outputs for informational "
        "purposes only and must NOT be construed as buy/sell recommendations. Stock market investments are subject to market risks "
        "including the potential loss of principal. Users are strongly advised to conduct independent due diligence and consult a "
        "SEBI-registered Investment Adviser before making any investment decisions. Past performance is not indicative of future results. "
        "The platform and its creators assume no liability for any financial losses incurred based on the use of this information."
    ),
    "version": "2.0",
    "last_updated": "2026-04-30"
}

# ── Plan limits ───────────────────────────────────────────────────────────────
PLAN_LIMITS = {
    'FREE':       {'analyze': True,  'portfolio': True,  'compare': False, 'daily_calls': 10,  'portfolio_per_day': 1},
    'PRO':        {'analyze': True,  'portfolio': True,  'compare': True,  'daily_calls': 500, 'portfolio_per_day': 999},
    'ENTERPRISE': {'analyze': True,  'portfolio': True,  'compare': True,  'daily_calls': 999999, 'portfolio_per_day': 999},
}

# ── Account lockout config ────────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS     = 5
LOCKOUT_DURATION_MINUTES = 15

# ── Redis cache ───────────────────────────────────────────────────────────────
CACHE_TIMEOUT  = 300
PRECOMPUTE_TTL = 86400

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


# ── Token blacklist helpers (ported from Node.js auth) ────────────────────────
BLACKLIST_PREFIX = "blacklist:token:"
BLACKLIST_TTL    = 15 * 60  # matches JWT access token lifetime (15 min)

def blacklist_token(token: str):
    """Add an access token to the blacklist so it can't be reused after logout."""
    if redis_client:
        try:
            redis_client.setex(f"{BLACKLIST_PREFIX}{token}", BLACKLIST_TTL, "1")
        except Exception as e:
            logger.warning(f"Token blacklist SET failed: {e}")
    else:
        # In-memory fallback (single instance only)
        set_cache(f"{BLACKLIST_PREFIX}{token}", "1", BLACKLIST_TTL)

def is_token_blacklisted(token: str) -> bool:
    """Returns True if the token has been blacklisted (user logged out)."""
    if redis_client:
        try:
            return bool(redis_client.exists(f"{BLACKLIST_PREFIX}{token}"))
        except Exception as e:
            logger.warning(f"Token blacklist GET failed: {e}")
            return False
    return get_from_cache(f"{BLACKLIST_PREFIX}{token}") is not None


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def get_precomputed(system_type: str, symbol: str):
    return get_from_cache(f"pre_{system_type}_{symbol}")


def _fetch_with_retry(func, *args, **kwargs):
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


# ── Password helpers ──────────────────────────────────────────────────────────

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


# ── Account lockout helpers (ported from Node.js auth) ───────────────────────

def is_account_locked(user) -> bool:
    if not user.locked_until:
        return False
    return user.locked_until > datetime.utcnow()

def record_failed_attempt(user):
    """Increment failed attempts and lock account if threshold reached."""
    user.failed_login_attempts += 1
    if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        logger.warning(f"Account locked: {user.email} — {user.failed_login_attempts} failed attempts")
    db.session.commit()
    return user.failed_login_attempts

def reset_failed_attempts(user):
    """Reset lockout state after successful login."""
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()


# ── Audit log helper (ported from Node.js auth) ───────────────────────────────

def log_audit(email, status, failure_reason=None, user_id=None):
    """Record every login attempt — success or failure."""
    try:
        ip         = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
        user_agent = request.headers.get('User-Agent', 'unknown')
        entry = LoginAudit(
            user_id        = user_id,
            email          = email,
            ip_address     = ip,
            user_agent     = user_agent,
            status         = status,
            failure_reason = failure_reason,
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


# ── Email verification helpers ────────────────────────────────────────────────

def _send_verification_email(to_email: str, token: str):
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


def _send_login_alert(admin_email: str, user_email: str, ip: str, user_agent: str):
    """Send a login alert email to the admin inbox on every successful login."""
    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host or not admin_email:
        return

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;background:#0f1117;color:#fff;border-radius:12px;overflow:hidden">
      <div style="background:#0f1117;padding:32px;text-align:center;border-bottom:1px solid #1e2130">
        <span style="font-size:24px;font-weight:bold">⚡ SentiQuant</span>
        <p style="color:#8b8fa8;margin:4px 0 0">Security &amp; Authentication System</p>
        <span style="background:#00c896;color:#000;padding:4px 16px;border-radius:4px;font-size:12px;font-weight:bold;display:inline-block;margin-top:12px">LOGIN DETECTED</span>
      </div>
      <div style="padding:32px">
        <h2 style="margin:0 0 24px">New login to your platform</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="color:#8b8fa8;padding:8px 0;width:140px">User</td><td style="color:#fff">{user_email}</td></tr>
          <tr><td style="color:#8b8fa8;padding:8px 0">IP Address</td><td style="color:#fff">{ip}</td></tr>
          <tr><td style="color:#8b8fa8;padding:8px 0">Device</td><td style="color:#fff;font-size:12px">{user_agent[:120]}</td></tr>
          <tr><td style="color:#8b8fa8;padding:8px 0">Time</td><td style="color:#fff">{timestamp}</td></tr>
        </table>
        <p style="color:#8b8fa8;font-size:13px;margin-top:24px">
          If this was not you, investigate immediately at your Render dashboard.
        </p>
      </div>
      <div style="padding:16px 32px;border-top:1px solid #1e2130;text-align:center">
        <p style="color:#8b8fa8;font-size:12px;margin:0">SentiQuant &mdash; AI Stock Analysis Platform</p>
      </div>
    </div>
    """
    msg = MIMEText(html, 'html')
    msg['Subject'] = f'🔐 Login Alert: {user_email} signed in'
    msg['From']    = os.getenv("SMTP_USER")
    msg['To']      = admin_email

    try:
        with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            server.sendmail(msg["From"], [admin_email], msg.as_string())
        logger.info(f"Login alert sent for {user_email}")
    except Exception as e:
        logger.error(f"Failed to send login alert for {user_email}: {e}")


# ── Request lifecycle hooks ───────────────────────────────────────────────────

@app.before_request
def attach_request_id():
    g.request_id     = str(uuid.uuid4())[:8]
    g.user_email     = "anon"
    g.user_id        = None
    g.user_plan      = "FREE"
    g.cache_hit      = False
    g.current_symbol = None
    g.request_start  = datetime.now()


def log_context():
    return f"[{getattr(g, 'request_id', '?')}] [{getattr(g, 'user_email', 'anon')}]"


@app.after_request
def track_usage_and_disclaimer(response):
    if response.content_type == 'application/json' and response.status_code == 200:
        try:
            data = response.get_json()
            if isinstance(data, dict) and 'disclaimer' not in data:
                data['disclaimer'] = SEBI_DISCLAIMER
                response.set_data(json.dumps(data))
                response.content_type = 'application/json'
        except Exception:
            pass

    TRACKED_PATHS = ('/analyze/', '/portfolio/', '/compare/')
    if any(p in request.path for p in TRACKED_PATHS) and request.method in ('GET', 'POST'):
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


# ── JWT helpers ───────────────────────────────────────────────────────────────

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

        # AUTH: Check blacklist first (handles logout-before-expiry)
        if is_token_blacklisted(token):
            return jsonify({'success': False, 'error': 'Token has been revoked. Please log in again.'}), 401

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


def plan_required(feature: str):
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
    @wraps(f)
    def decorated(*args, **kwargs):
        plan    = getattr(g, 'user_plan', 'FREE')
        user_id = getattr(g, 'user_id', None)
        limit   = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE']).get('portfolio_per_day', 1)

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


def check_daily_api_limit(f):
    """Blocks requests when the user has exceeded their daily API call limit."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = getattr(g, 'user_id', None)
        plan    = getattr(g, 'user_plan', 'FREE')
        limit   = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE']).get('daily_calls', 10)

        if limit < 999999 and user_id:
            try:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                count = db.session.query(UserUsage).filter(
                    UserUsage.user_id   == user_id,
                    UserUsage.timestamp >= today_start,
                    UserUsage.cache_hit == False
                ).count()

                if count >= limit:
                    return jsonify({
                        'success': False,
                        'error':   f'{plan} plan allows {limit} API calls per day. '
                                   f'You have used {count}/{limit} today. '
                                   f'Upgrade to PRO at sentiquant.org/pricing for 500 calls/day.',
                        'code':    'DAILY_LIMIT_EXCEEDED',
                    }), 429
            except Exception as e:
                logger.warning(f"{log_context()} Daily limit check failed: {e}")

        return f(*args, **kwargs)
    return decorated


# ── Validation helpers ────────────────────────────────────────────────────────

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


# ── TradingAPI class ──────────────────────────────────────────────────────────

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
            self.swing_system = EnhancedSwingTradingSystem(data_provider=data_provider, redis_client=redis_client)
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
                'signal':               self._normalize_signal(entry_signal),
                'strategy':             self._enhance_strategy_description(entry_strategy, entry_signal, system_type),
                'entry_price':          f"Around {current_price:.2f}" if current_price > 0 else 'N/A',
                'stop_loss':            f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) and stop_loss > 0 else 'N/A',
                'target_1':             f"{targets.get('target_1', 0):.2f}" if targets.get('target_1') else 'N/A',
                'target_2':             f"{targets.get('target_2', 0):.2f}" if targets.get('target_2') else 'N/A',
                'target_3':             f"{targets.get('target_3', 0):.2f}" if targets.get('target_3') else 'N/A',
                'trailing_stop_advice': mgmt_note or 'Consider moving stop loss to breakeven after hitting Target 1.',
            }
        except Exception as e:
            logger.error(f"{log_context()} Error generating trading plan: {e}")
            return self._fallback_trading_plan(result)

    def _normalize_signal(self, signal: str) -> str:
        """Map raw BUY/SELL/HOLD signals to SEBI-compliant neutral terms."""
        s = signal.upper().strip()
        if 'STRONG BUY' in s or 'STRONG_BUY' in s: return 'STRONG BULLISH'
        if 'BUY' in s:    return 'BULLISH'
        if 'STRONG SELL' in s or 'STRONG_SELL' in s: return 'STRONG BEARISH'
        if 'SELL' in s:   return 'BEARISH'
        if 'HOLD' in s or 'WATCH' in s: return 'NEUTRAL'
        if 'AVOID' in s:  return 'CAUTIOUS'
        return signal  # pass through if already compliant

    def _enhance_strategy_description(self, base_strategy, signal, system_type):
        s = signal.lower()
        if "strong buy" in s or "strong bullish" in s:
            return f"A high-conviction bullish technical setup for {system_type.lower()} analysis. {base_strategy}"
        elif "buy" in s or "bullish" in s:
            return f"A positive technical setup for {system_type.lower()} analysis. {base_strategy}"
        elif "hold" in s or "neutral" in s or "watch" in s:
            return f"Neutral technical stance — wait for clearer signals. {base_strategy}"
        elif "sell" in s or "bearish" in s or "avoid" in s or "cautious" in s:
            return f"Caution indicated by technical indicators. {base_strategy}"
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
        score_key     = 'swing_score' if system_type == 'Swing' else 'position_score'
        score         = result.get(score_key, 0)
        current_price = result.get('current_price', 0)
        all_targets   = self.extract_targets_from_backend(result)
        target_price  = all_targets.get('target_2', 0) or all_targets.get('target_1', current_price)
        potential_ret = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
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
        all_stocks  = _fetch_with_retry(self.swing_system.get_all_stock_symbols)
        all_results = self.swing_system.analyze_stocks_parallel(all_stocks)
        filtered    = self.swing_system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
        portfolio_list = self.swing_system.generate_portfolio_allocation(filtered, budget, risk_appetite)
        std         = self._standardize_portfolio_keys(portfolio_list)
        total_alloc = sum(i.get('investment_amount', 0) for i in std)
        avg_score   = sum(i.get('score', 0) for i in std) / len(std) if std else 0
        return {'portfolio': std, 'summary': {
            'total_budget': budget, 'total_allocated': total_alloc,
            'remaining_cash': budget - total_alloc, 'diversification': len(std), 'average_score': avg_score}}

    def generate_position_portfolio(self, budget, risk_appetite, time_period):
        if not self.position_system:
            raise ConnectionAbortedError('Position trading system not available')
        results        = _fetch_with_retry(
            self.position_system.create_personalized_portfolio, risk_appetite, time_period, budget)
        portfolio_data = results.get('portfolio', {})
        portfolio_list = ([{**v, 'symbol': k} for k, v in portfolio_data.items()]
                          if isinstance(portfolio_data, dict) else portfolio_data)
        std         = self._standardize_portfolio_keys(portfolio_list)
        total_alloc = sum(i.get('investment_amount', 0) for i in std)
        avg_score   = sum(i.get('score', 0) for i in std) / len(std) if std else 0
        return {'portfolio': std, 'summary': {
            'total_budget': budget, 'total_allocated': total_alloc,
            'remaining_cash': budget - total_alloc, 'diversification': len(std), 'average_score': avg_score}}


trading_api = TradingAPI()


# ── Blueprint v1 ──────────────────────────────────────────────────────────────

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
    existing = User.query.filter_by(email=email).first()
    if existing:
        if existing.is_verified:
            return jsonify({'success': False, 'error': 'User already exists'}), 409
        # Unverified account — resend verification with a fresh token
        existing.verify_token = uuid.uuid4().hex
        db.session.commit()
        _executor.submit(_send_verification_email, email, existing.verify_token)
        return jsonify({
            'success': True,
            'message': 'Registered successfully. Please check your email to verify your account.'
        }), 201

    hashed       = hash_password(password)
    verify_token = uuid.uuid4().hex
    user = User(email=email, password_hash=hashed, verify_token=verify_token)
    db.session.add(user)
    db.session.commit()

    _executor.submit(_send_verification_email, email, verify_token)

    logger.info(f"{log_context()} Registered: {email}")
    return jsonify({
        'success': True,
        'message': 'Registered successfully. Please check your email to verify your account.'
    }), 201


@v1.route('/auth/verify-email', methods=['GET'])
def verify_email():
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


# ── TEMPORARY: Direct verify endpoint (remove after initial setup) ─────────────
@v1.route('/auth/verify-direct', methods=['POST'])
def verify_direct():
    """TEMPORARY: Direct email verification without token. Remove after setup."""
    admin_key = request.headers.get('X-Admin-Key', '')
    if admin_key != os.getenv('ADMIN_KEY', ''):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    data  = request.get_json() or {}
    email = data.get('email', '').lower().strip()
    user  = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    user.is_verified  = True
    user.verify_token = None
    db.session.commit()
    logger.info(f"[admin] Manually verified: {email}")
    return jsonify({'success': True, 'message': f'{email} verified.'})


@v1.route('/auth/login', methods=['POST'])
@limiter.limit("10 per minute", key_func=lambda: (request.get_json(silent=True) or {}).get('email', get_remote_address()))
def login():
    data     = request.get_json() or {}
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')
    secret   = os.getenv("JWT_SECRET")
    if not secret:
        logger.critical("JWT_SECRET not set!")
        return jsonify({'success': False, 'error': 'Server misconfigured'}), 500

    user = User.query.filter_by(email=email).first()

    # AUTH: User not found — same error as wrong password (prevent email enumeration)
    if not user:
        log_audit(email, status='failed', failure_reason='user_not_found')
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    # AUTH: Account locked?
    if is_account_locked(user):
        minutes_left = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
        log_audit(email, status='locked', failure_reason='account_locked', user_id=user.id)
        logger.warning(f"{log_context()} Login blocked — account locked: {email}")
        return jsonify({
            'success': False,
            'error':   f'Account temporarily locked due to too many failed attempts. '
                       f'Try again in {minutes_left} minute(s).',
            'code':    'ACCOUNT_LOCKED',
            'locked_until': user.locked_until.isoformat(),
        }), 423

    # AUTH: Account disabled?
    if not user.is_active:
        log_audit(email, status='failed', failure_reason='account_disabled', user_id=user.id)
        return jsonify({'success': False, 'error': 'Account is disabled. Please contact support.'}), 403

    # AUTH: Wrong password?
    if not verify_password(password, user.password_hash):
        attempts = record_failed_attempt(user)
        remaining = max(0, MAX_FAILED_ATTEMPTS - attempts)
        log_audit(email, status='failed', failure_reason='wrong_password', user_id=user.id)
        logger.warning(f"{log_context()} Failed login: {email} — attempt {attempts}/{MAX_FAILED_ATTEMPTS}")
        return jsonify({
            'success': False,
            'error':   f'Invalid credentials. {remaining} attempt(s) remaining before lockout.'
                       if remaining > 0 else 'Account locked due to too many failed attempts.',
            'attempts_remaining': remaining,
        }), 401

    # AUTH: Email not verified?
    if not user.is_verified:
        log_audit(email, status='failed', failure_reason='email_not_verified', user_id=user.id)
        return jsonify({
            'success': False,
            'error':   'Please verify your email before logging in.'
        }), 403

    # ✅ Login success
    reset_failed_attempts(user)

    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        db.session.commit()
        logger.info(f"Password upgraded to argon2 for {email}")

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

    log_audit(email, status='success', user_id=user.id)
    logger.info(f"{log_context()} Login: {email} (plan={user.plan})")
    # Send admin login alert (non-blocking)
    _login_ip         = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
    _login_user_agent = request.headers.get('User-Agent', 'unknown')
    _admin_email      = os.getenv('ADMIN_EMAIL') or os.getenv('SMTP_USER')
    if _admin_email:
        _executor.submit(_send_login_alert, _admin_email, email, _login_ip, _login_user_agent)
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
            'email':   payload['email'],
            'user_id': payload.get('user_id'),
            'plan':    payload.get('plan', 'FREE'),
            'type':    'access',
            'exp':     datetime.utcnow() + timedelta(minutes=15)
        }, secret, algorithm="HS256")
        return jsonify({'success': True, 'access_token': new_token, 'expires_in': 900})
    except jwt.ExpiredSignatureError:
        return jsonify({'success': False, 'error': 'Refresh token expired. Please log in again.'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'success': False, 'error': 'Invalid refresh token'}), 401


@v1.route('/auth/logout', methods=['POST'])
@token_required
def logout():
    """
    AUTH: Blacklist the current access token so it can't be reused.
    Client must also delete tokens from storage.
    """
    try:
        raw_token = request.headers['Authorization'].split(" ")[1]
        blacklist_token(raw_token)
        logger.info(f"{log_context()} Logout: {g.user_email}")
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'success': False, 'error': 'Logout failed'}), 500


@v1.route('/auth/usage', methods=['GET'])
@token_required
def get_usage():
    """Returns today's real usage from DB for the authenticated user."""
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Total trading API calls today (analyze + portfolio + compare, excluding cache hits)
        total_calls = db.session.query(UserUsage).filter(
            UserUsage.user_id   == g.user_id,
            UserUsage.timestamp >= today_start,
            UserUsage.cache_hit == False,
            db.or_(
                UserUsage.endpoint.like('%/analyze/%'),
                UserUsage.endpoint.like('%/portfolio/%'),
                UserUsage.endpoint.like('%/compare/%'),
            )
        ).count()

        # Portfolio calls today
        portfolio_calls = db.session.query(UserUsage).filter(
            UserUsage.user_id   == g.user_id,
            UserUsage.endpoint.like('%/portfolio/%'),
            UserUsage.timestamp >= today_start,
            UserUsage.cache_hit == False
        ).count()

        plan   = g.user_plan
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE'])

        return jsonify({
            'success': True,
            'data': {
                'plan':                  plan,
                'daily_api_calls_used':  total_calls,
                'daily_api_calls_limit': limits['daily_calls'],
                'portfolios_used_today': portfolio_calls,
                'portfolio_limit':       limits['portfolio_per_day'],
                'reset_at':              'midnight UTC',
            }
        })
    except Exception as e:
        logger.error(f"{log_context()} usage endpoint error: {e}")
        return jsonify({'success': False, 'error': 'Could not fetch usage'}), 500


# ── Trading endpoints ─────────────────────────────────────────────────────────

# ── Portfolio job store ───────────────────────────────────────────────────────
# Jobs are stored in Redis (production) or in-memory dict (fallback).
# Each job: { status, progress, result, error, created_at }

JOB_TTL = 3600  # jobs expire after 1 hour

def _job_key(job_id: str) -> str:
    return f"portfolio_job:{job_id}"

def _set_job(job_id: str, data: dict):
    if redis_client:
        try:
            redis_client.setex(_job_key(job_id), JOB_TTL, json.dumps(data))
            return
        except Exception as e:
            logger.warning(f"Redis job SET failed: {e}")
    set_cache(_job_key(job_id), data, JOB_TTL)

def _get_job(job_id: str) -> dict | None:
    if redis_client:
        try:
            val = redis_client.get(_job_key(job_id))
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning(f"Redis job GET failed: {e}")
    return get_from_cache(_job_key(job_id))

def _run_swing_job(job_id: str, budget: float, risk: str, user_id: int, plan: str):
    """Background thread: run swing portfolio and store result."""
    try:
        _set_job(job_id, {'status': 'processing', 'progress': 5, 'result': None, 'error': None})
        result = trading_api.generate_swing_portfolio(budget, risk)
        _set_job(job_id, {'status': 'complete', 'progress': 100, 'result': result, 'error': None})
        # Track usage
        with app.app_context():
            try:
                usage = UserUsage(user_id=user_id, endpoint='/api/v1/portfolio/swing',
                                  cost_estimate=2.0, cache_hit=False)
                db.session.add(usage)
                db.session.commit()
            except Exception as e:
                logger.warning(f"Usage tracking failed for job {job_id}: {e}")
    except Exception as e:
        logger.error(f"Portfolio job {job_id} failed: {e}")
        _set_job(job_id, {'status': 'failed', 'progress': 0, 'result': None, 'error': str(e)})

def _run_position_job(job_id: str, budget: float, risk: str, time_period: int, user_id: int, plan: str):
    """Background thread: run position portfolio and store result."""
    try:
        _set_job(job_id, {'status': 'processing', 'progress': 5, 'result': None, 'error': None})
        result = trading_api.generate_position_portfolio(budget, risk, time_period)
        _set_job(job_id, {'status': 'complete', 'progress': 100, 'result': result, 'error': None})
        with app.app_context():
            try:
                usage = UserUsage(user_id=user_id, endpoint='/api/v1/portfolio/position',
                                  cost_estimate=2.0, cache_hit=False)
                db.session.add(usage)
                db.session.commit()
            except Exception as e:
                logger.warning(f"Usage tracking failed for job {job_id}: {e}")
    except Exception as e:
        logger.error(f"Portfolio job {job_id} failed: {e}")
        _set_job(job_id, {'status': 'failed', 'progress': 0, 'result': None, 'error': str(e)})

@v1.route('/portfolio/swing/start', methods=['POST'])
@token_required
@plan_required('portfolio')
@check_portfolio_daily_limit
@require_systems
@limiter.limit("5 per minute")
def start_swing_portfolio():
    """Start a portfolio generation job and return job_id immediately."""
    try:
        data   = request.get_json() or {}
        budget = validate_budget(data.get('budget'))
        risk   = validate_risk_appetite(data.get('risk_appetite'))
        job_id = uuid.uuid4().hex
        _set_job(job_id, {'status': 'queued', 'progress': 0, 'result': None, 'error': None})
        _executor.submit(_run_swing_job, job_id, budget, risk, g.user_id, g.user_plan)
        logger.info(f"{log_context()} Started swing portfolio job {job_id}")
        return jsonify({'success': True, 'job_id': job_id})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} portfolio/swing/start: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@v1.route('/portfolio/position/start', methods=['POST'])
@token_required
@plan_required('portfolio')
@check_portfolio_daily_limit
@require_systems
@limiter.limit("5 per minute")
def start_position_portfolio():
    """Start a position portfolio generation job and return job_id immediately."""
    try:
        data        = request.get_json() or {}
        budget      = validate_budget(data.get('budget'))
        risk        = validate_risk_appetite(data.get('risk_appetite'))
        time_period = validate_time_period(data.get('time_period'))
        job_id = uuid.uuid4().hex
        _set_job(job_id, {'status': 'queued', 'progress': 0, 'result': None, 'error': None})
        _executor.submit(_run_position_job, job_id, budget, risk, time_period, g.user_id, g.user_plan)
        logger.info(f"{log_context()} Started position portfolio job {job_id}")
        return jsonify({'success': True, 'job_id': job_id})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} portfolio/position/start: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@v1.route('/portfolio/job/<job_id>', methods=['GET'])
@token_required
@limiter.limit("30 per minute")
def get_portfolio_job(job_id: str):
    """Poll job status. Returns status, progress (0-100), and result when complete."""
    job = _get_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found or expired'}), 404
    return jsonify({'success': True, 'data': job})



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
    try:
        symbol = validate_symbol(symbol)
        g.current_symbol = symbol

        # AUTH: Check daily limit before serving even cached results.
        # Cached responses are still counted against the user's quota.
        user_id = getattr(g, 'user_id', None)
        plan    = getattr(g, 'user_plan', 'FREE')
        limit   = PLAN_LIMITS.get(plan, PLAN_LIMITS['FREE']).get('daily_calls', 10)

        if limit < 999999 and user_id:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            count = db.session.query(UserUsage).filter(
                UserUsage.user_id   == user_id,
                UserUsage.timestamp >= today_start,
                UserUsage.cache_hit == False,
                db.or_(
                    UserUsage.endpoint.like('%/analyze/%'),
                    UserUsage.endpoint.like('%/portfolio/%'),
                    UserUsage.endpoint.like('%/compare/%'),
                )
            ).count()
            if count >= limit:
                return jsonify({
                    'success': False,
                    'error':   f'{plan} plan allows {limit} API calls per day. '
                               f'You have used {count}/{limit} today. '
                               f'Upgrade to PRO at sentiquant.org/pricing for 500 calls/day.',
                    'code':    'DAILY_LIMIT_EXCEEDED',
                }), 429

        precomputed = get_precomputed(system_type, symbol)
        if precomputed:
            g.cache_hit = True
            return jsonify({'success': True, 'data': precomputed, 'source': 'precomputed'})

        cache_key = f"{system_type}_analysis_{symbol}"
        if cached := get_from_cache(cache_key):
            g.cache_hit = True
            return jsonify({'success': True, 'data': cached, 'source': 'cache'})

        system = getattr(trading_api, f"{system_type}_system")
        if not system:
            return jsonify({'success': False, 'error': f'{system_type.capitalize()} system unavailable'}), 503

        analysis_func = getattr(system, f"analyze_{system_type}_trading_stock")
        result        = _fetch_with_retry(analysis_func, symbol)

        if not result:
            return jsonify({'success': False, 'error': f'Could not analyze {symbol}'}), 404

        formatted = trading_api.format_analysis_response(result, system_type.capitalize())
        set_cache(cache_key, formatted)
        return jsonify({'success': True, 'data': formatted, 'source': 'live'})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"{log_context()} analyze/{system_type}/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@v1.route('/analyze/swing/<symbol>', methods=['GET'])
@token_required
@check_daily_api_limit
@require_systems
@limiter.limit("10 per minute")
def analyze_swing_stock_endpoint(symbol):
    return analyze_stock('swing', symbol)


@v1.route('/analyze/position/<symbol>', methods=['GET'])
@token_required
@check_daily_api_limit
@require_systems
@limiter.limit("10 per minute")
def analyze_position_stock_endpoint(symbol):
    return analyze_stock('position', symbol)


@v1.route('/portfolio/swing', methods=['POST'])
@token_required
@plan_required('portfolio')
@check_portfolio_daily_limit
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
@check_portfolio_daily_limit
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
@plan_required('compare')
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
        swing_result    = _fetch_with_retry(trading_api.swing_system.analyze_swing_trading_stock, symbol)
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


@v1.route('/contact', methods=['POST'])
@limiter.limit("5 per hour")
def contact():
    """Contact form — sends email to admin via Zoho SMTP."""
    data    = request.get_json() or {}
    name    = data.get('name', '').strip()
    email   = data.get('email', '').strip()
    subject = data.get('subject', '').strip()
    message = data.get('message', '').strip()

    if not all([name, email, subject, message]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    if len(message) < 20:
        return jsonify({'success': False, 'error': 'Message too short'}), 400

    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        logger.warning("Contact form submitted but SMTP not configured")
        return jsonify({'success': False, 'error': 'service_unavailable'}), 503

    try:
        recipient = os.getenv("ADMIN_EMAIL") or os.getenv("SMTP_USER")
        html_body = f"""
        <div style="font-family:sans-serif;max-width:600px;margin:0 auto;background:#0f1117;color:#fff;border-radius:12px;overflow:hidden">
          <div style="padding:24px 32px;border-bottom:1px solid #1e2130">
            <span style="font-size:20px;font-weight:bold">⚡ SentiQuant</span>
            <p style="color:#8b8fa8;margin:4px 0 0;font-size:13px">Contact Form Submission</p>
          </div>
          <div style="padding:32px">
            <table style="width:100%;border-collapse:collapse">
              <tr><td style="color:#8b8fa8;padding:8px 0;width:120px;font-size:13px">Name</td><td style="color:#fff;font-size:13px">{name}</td></tr>
              <tr><td style="color:#8b8fa8;padding:8px 0;font-size:13px">Email</td><td style="font-size:13px"><a href="mailto:{email}" style="color:#06b6d4">{email}</a></td></tr>
              <tr><td style="color:#8b8fa8;padding:8px 0;font-size:13px">Subject</td><td style="color:#fff;font-size:13px">{subject}</td></tr>
            </table>
            <div style="margin-top:24px;padding:16px;background:#1e2130;border-radius:8px">
              <p style="color:#8b8fa8;font-size:11px;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.05em">Message</p>
              <p style="color:#fff;font-size:14px;line-height:1.6;margin:0;white-space:pre-wrap">{message}</p>
            </div>
          </div>
          <div style="padding:16px 32px;border-top:1px solid #1e2130;text-align:center">
            <p style="color:#8b8fa8;font-size:12px;margin:0">SentiQuant — AI Stock Analysis Platform</p>
          </div>
        </div>
        """
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText as _MIMEText
        msg = MIMEMultipart('alternative')
        msg['Subject']  = f"[Contact] {subject} — from {name}"
        msg['From']     = os.getenv("SMTP_USER")
        msg['To']       = recipient
        msg['Reply-To'] = email
        msg.attach(_MIMEText(f"Name: {name}\nEmail: {email}\nSubject: {subject}\n\nMessage:\n{message}", 'plain'))
        msg.attach(_MIMEText(html_body, 'html'))

        with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            server.sendmail(msg['From'], [recipient], msg.as_string())

        logger.info(f"Contact form email sent from {email} ({name})")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Contact form email failed: {e}")
        return jsonify({'success': False, 'error': 'Failed to send message'}), 500


@v1.route('/disclaimer', methods=['GET'])
def get_disclaimer():
    return jsonify({'success': True, 'data': SEBI_DISCLAIMER})


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
        'endpoint':        r.endpoint,
        'calls':           r.calls,
        'total_cost':      round(r.total_cost or 0, 2),
        'avg_response_ms': round(r.avg_ms or 0, 1),
        'cache_hit_rate':  round((r.cache_hits or 0) / r.calls * 100, 1),
    } for r in rows]
    return jsonify({'success': True, 'data': data, 'period_days': since_days})


@v1.route('/admin/audit', methods=['GET'])
@token_required
def audit_log():
    """AUTH: View recent login audit entries. Token required."""
    since_days = int(request.args.get('days', 1))
    since      = datetime.utcnow() - timedelta(days=since_days)
    rows = (db.session.query(LoginAudit)
            .filter(LoginAudit.created_at >= since)
            .order_by(LoginAudit.created_at.desc())
            .limit(200)
            .all())
    data = [{
        'email':          r.email,
        'status':         r.status,
        'failure_reason': r.failure_reason,
        'ip_address':     r.ip_address,
        'created_at':     r.created_at.isoformat(),
    } for r in rows]
    return jsonify({'success': True, 'data': data, 'period_days': since_days})


# ── Register blueprint + legacy aliases ──────────────────────────────────────

app.register_blueprint(v1)

@app.route('/api/stocks',                     methods=['GET'])
def legacy_stocks():         return get_all_stocks()

@app.route('/api/analyze/swing/<symbol>',    methods=['GET'])
@token_required
def legacy_swing(symbol):    return analyze_stock('swing', symbol)

@app.route('/api/analyze/position/<symbol>', methods=['GET'])
@token_required
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


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from waitress import serve
    logging.getLogger('waitress').setLevel(logging.WARNING)
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Waitress on port {port}...")
    serve(app, host="0.0.0.0", port=port)
