import os
from huggingface_hub import hf_hub_download

def download_model_from_hf(repo_id, filename):
    """Download a file from Hugging Face Hub using the HF API token (if available)."""
    token = os.getenv("HF_API_KEY")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return local_path
