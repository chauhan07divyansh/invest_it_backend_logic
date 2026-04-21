import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
#  API KEYS & SECRETS
# ==============================================================================
NEWS_API_KEY           = os.getenv("NEWS_API_KEY", "")
HF_API_KEY             = os.getenv("HF_API_KEY", "")
EVENT_REGISTRY_API_KEY = os.getenv("EVENT_REGISTRY_API_KEY", "")

# ==============================================================================
#  EXTERNAL API ENDPOINTS
# ==============================================================================
EVENT_REGISTRY_ENDPOINT = os.getenv(
    "EVENT_REGISTRY_ENDPOINT",
    "https://eventregistry.org/api/v1/article/getArticles"
)
HF_SENTIMENT_API_URL = os.getenv(
    "HF_SENTIMENT_API_URL",
    "https://brosoverhoes07-financial-model.hf.space/predict/sentiment"
)
HF_MDA_API_URL = os.getenv(
    "HF_MDA_API_URL",
    "https://brosoverhoes07-financial-model.hf.space/predict/mda"
)

# ==============================================================================
#  TRADING LOGIC PARAMETERS
# ==============================================================================
POSITION_TRADING_PARAMS = {
    'min_holding_period':       90,
    'max_holding_period':       1095,
    'risk_per_trade':           0.01,
    'max_portfolio_risk':       0.05,
    'profit_target_multiplier': 4.0,
    'max_positions':            12,
    'fundamental_weight':       0.45,
    'technical_weight':         0.35,
    'sentiment_weight':         0.10,
    'mda_weight':               0.10,
}

SWING_TRADING_PARAMS = {
    'min_holding_period':       3,
    'max_holding_period':       30,
    'risk_per_trade':           0.02,
    'max_portfolio_risk':       0.10,
    'profit_target_multiplier': 2.5,
}
