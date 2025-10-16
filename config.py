import os

# === BASE SETTINGS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === API KEYS ===
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "dd33ebe105ea4b02a3b7e77bc4a93d01")
HF_API_KEY = os.getenv("HF_API_KEY")  # Set this in Render dashboard (Environment → Add Variable)

# === HUGGING FACE MODEL ENDPOINTS ===
# ⚠️ These URLs currently point to Hugging Face’s *web* interface, not raw files.
# For programmatic access (Render fetching models), you’ll need the *raw file URLs* or use Hugging Face Hub download.

HF_SBERT_MODEL_URL = os.getenv(
    "HF_SBERT_MODEL_URL",
    "https://huggingface.co/Brosoverhoes07/financial-model/resolve/main/sentiment_pipeline_chunking.joblib"
)

HF_MDA_MODEL_URL = os.getenv(
    "HF_MDA_MODEL_URL",
    "https://huggingface.co/Brosoverhoes07/financial-model/resolve/main/best_model_fold_1.pth"
)

# === TRADING PARAMETERS ===
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
# === BACKWARD COMPATIBILITY FOR LOCAL PATH VARIABLES ===
SBERT_MODEL_PATH = HF_SBERT_MODEL_URL
MDA_MODEL_PATH = HF_MDA_MODEL_URL



