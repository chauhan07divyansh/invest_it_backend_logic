import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development.
# On Render, these variables will be set directly in the dashboard.
load_dotenv()

# ==============================================================================
#  API KEYS & SECRETS
# ==============================================================================
# These MUST be set as environment variables in your Render dashboard.
# DO NOT hardcode secret keys in your code.

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Your Hugging Face User Access Token (read or write)


# ==============================================================================
#  HUGGING FACE API ENDPOINTS
# ==============================================================================
# These are the URLs for your deployed Hugging Face Spaces.
# You can find these URLs under the "Use via API" section on each Space page.
# These MUST be set as environment variables in your Render dashboard.

HF_SENTIMENT_API_URL = os.getenv("HF_SENTIMENT_API_URL")
HF_MDA_API_URL = os.getenv("HF_MDA_API_URL")


# ==============================================================================
#  TRADING LOGIC PARAMETERS
# ==============================================================================
# These values configure the core behavior of your trading algorithms.

POSITION_TRADING_PARAMS = {
    'min_holding_period': 90,
    'max_holding_period': 1095,
    'risk_per_trade': 0.01,
    'max_portfolio_risk': 0.05,
    'profit_target_multiplier': 4.0,
    'max_positions': 12,
    'fundamental_weight': 0.45,
    'technical_weight': 0.35,
    'sentiment_weight': 0.10,
    'mda_weight': 0.10,
}

SWING_TRADING_PARAMS = {
    'min_holding_period': 3,
    'max_holding_period': 30,
    'risk_per_trade': 0.02,
    'max_portfolio_risk': 0.10,
    'profit_target_multiplier': 2.5,
}

# IMPORTANT: All model download logic has been removed. This file now only
# declares configuration variables to be read from the environment.






