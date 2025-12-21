# hf_utils.py

import os
import requests
import logging

session = requests.Session()

HF_API_KEY = os.getenv("HF_API_KEY")
if HF_API_KEY:
    session.headers.update({"Authorization": f"Bearer {HF_API_KEY}"})


def query_hf_api(api_url: str, payload: dict, timeout: int = 20) -> dict | None:
    """
    Sends a JSON payload to a Hugging Face Space endpoint.

    Args:
        api_url (str): Full HF Space endpoint URL.
        payload (dict): JSON payload expected by the endpoint.
        timeout (int): Request timeout in seconds.

    Returns:
        dict: Parsed JSON response
        None: If request fails
    """
    if not api_url:
        logging.error("Hugging Face API URL is not configured.")
        return None

    try:
        response = session.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"HF Space request failed [{api_url}]: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected HF client error: {e}")
        return None
