import os

# This automatically finds the root directory of your project.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API KEYS
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "dd33ebe105ea4b02a3b7e77bc4a93d01")

# --- FILE PATHS (Assumes a 'models' folder is inside your project) ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_pipeline_chunking.joblib")
MDA_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_fold_1.pth")

# --- TRADING PARAMETERS ---
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