import os
from huggingface_hub import hf_hub_download

# This is your unique Hugging Face Repository ID
REPO_ID = "Brosoverhoes07/finance_models"

def download_model_file(filename):
    """
    Downloads a specific file from Hugging Face and returns the local path.
    """
    print(f"--- Initiating download for: {filename} ---")
    
    # We use the HF_TOKEN you will set in Render's dashboard
    token = os.getenv("HF_TOKEN")
    
    try:
        # This downloads the file to a cache folder on Render
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=token
        )
        print(f"--- Successfully downloaded to: {file_path} ---")
        return file_path
    except Exception as e:
        print(f"--- Error downloading {filename}: {e} ---")
        return None



def get_all_models():
    """
    Helper to fetch the main model files needed for the project.
    """
    files_to_download = [
        "best_model_fold_1.pth",
        "sbert_rf_pipeline.pkl",
        "sentiment_pipeline_chunking.joblib"
    ]
    
    paths = {}
    for file in files_to_download:
        paths[file] = download_model_file(file)
    
    return paths
