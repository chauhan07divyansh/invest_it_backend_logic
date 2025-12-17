# hf_utils.py

import os
import requests
import logging

# It's better to create a single, reusable session object for performance
# as it reuses the underlying TCP connection.
session = requests.Session()

# The API Key is automatically read from the environment variable set in Render
HF_API_KEY = os.getenv("HF_API_KEY")
if HF_API_KEY:
    session.headers.update({"Authorization": f"Bearer {HF_API_KEY}"})

def query_hf_api(api_url: str, payload: dict, timeout: int = 20) -> list | None:
    """
    Sends a payload to a specified Hugging Face API endpoint and returns the result.

    Args:
        api_url (str): The full URL of the Hugging Face Space API endpoint.
        payload (dict): The data to send, typically formatted as {"inputs": ...}.
        timeout (int): How many seconds to wait for a response.

    Returns:
        A list containing the model's predictions, or None if an error occurs.
    """
    if not api_url:
        logging.error("Hugging Face API URL is not configured.")
        return None

    try:
        response = session.post(api_url, json=payload, timeout=timeout)
        
        # This will automatically raise an exception for HTTP errors (like 4xx, 5xx)
        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to query Hugging Face API at {api_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the API call: {e}")
        return None
