import os
os.environ['HF_HOME'] = './cache'
os.environ['TRANSFORMERS_CACHE'] = './cache'
os.environ['HF_DATASETS_CACHE'] = './cache'
os.makedirs('./cache', exist_ok=True)

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ================================
# ✅ SAFE TRANSFORMERS CACHE SETUP
# ================================
# Hugging Face Spaces have read-only /.cache directories,
# so we create a custom writable cache for BERT downloads.
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.makedirs('/tmp/hf_cache', exist_ok=True)


# ================================
# ✅ LOGGER SETUP
# ================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================================
# ✅ OPTIONAL SBERT IMPORT
# ================================
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    logger.info("✅ sentence-transformers found. SBERT available.")
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers not found. SBERT will not be functional.")


# ================================
# ✅ SBERT TRANSFORMER CLASS
# ================================
class SBERTTransformer:
    """
    Wrapper class for SentenceTransformer to integrate with scikit-learn pipelines.
    Used for encoding text embeddings with SBERT during sentiment model prediction.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            logger.info(f"SBERT model '{model_name}' loaded successfully.")
        else:
            self.model = None
            logger.warning("SBERT model unavailable — using fallback logic.")

    def fit(self, X, y=None):
        """No-op fit method for sklearn pipeline compatibility."""
        return self

    def transform(self, X, y=None):
        """Encodes text data into embeddings."""
        if self.model and X is not None:
            return self.model.encode(list(X), show_progress_bar=False)
        logger.warning("SBERTTransformer.transform() called without model or data.")
        return []


# ================================
# ✅ MDA SENTIMENT MODEL CLASS
# ================================
class MDASentimentModel:
    """
    Loads and manages a fine-tuned BERT model for analyzing
    Management Discussion & Analysis (MD&A) tone and sentiment.
    """

    def __init__(self, model_path):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            logger.info("Initializing MDA sentiment model...")

            # Inner BERT classifier class
            class BERTSentimentClassifier(nn.Module):
                def __init__(self, n_classes):
                    super(BERTSentimentClassifier, self).__init__()
                    self.bert = AutoModel.from_pretrained('bert-base-uncased')
                    self.drop = nn.Dropout(p=0.3)
                    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=False
                    )
                    output = self.drop(outputs[1])
                    return self.out(output)

            # Instantiate model
            self.model = BERTSentimentClassifier(n_classes=5)

            # Load state dictionary
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.bert.resize_token_embeddings(30873)  # Adjust for vocab mismatch if any
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            logger.info(f"✅ MDA sentiment model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"❌ Failed to load MDA model from {model_path}: {e}")

    def is_available(self):
        """Returns True if model is loaded successfully."""
        return self.model is not None

    def predict(self, texts: list):
        """
        Placeholder inference method. You can extend this for actual predictions.
        """
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ MDA model not loaded; returning empty predictions.")
            return [], []

        # Example inference logic (can be customized)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        return predictions, logits.cpu().numpy().tolist()