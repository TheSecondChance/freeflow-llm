import json
import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.

    Args:
        provider: Provider name (e.g., 'groq', 'google')

    Returns:
        API key if found, None otherwise
    """
    key_mapping = {
        "groq": "GROQ_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_key = key_mapping.get(provider.lower())
    if env_key:
        return get_env_var(env_key)
    return None


def is_rate_limit_error(status_code: int, error_message: str = "") -> bool:
    """
    Check if an error is a rate limit error.

    Args:
        status_code: HTTP status code
        error_message: Error message text

    Returns:
        True if it's a rate limit error
    """
    if status_code == 429:
        return True

    # Check error message for common rate limit indicators
    rate_limit_keywords = [
        "rate limit",
        "too many requests",
        "quota exceeded",
        "resource exhausted",
    ]
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in rate_limit_keywords)


def parse_sse_line(line: str) -> Optional[dict[str, Any]]:
    """
    Parse a single SSE data line into JSON.

    Args:
        line: SSE data line (without 'data: ' prefix)

    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not line or line == "[DONE]":
        return None

    try:
        result: dict[str, Any] = json.loads(line)
        return result
    except json.JSONDecodeError:
        return None


def extract_error_message(response: httpx.Response) -> str:
    """
    Extract error message from HTTP response.

    Args:
        response: HTTP response object

    Returns:
        Error message string
    """
    try:
        error_data = response.json()

        if "error" in error_data:
            error = error_data["error"]
            if isinstance(error, dict):
                message: str = error.get("message", str(error))
                return message
            return str(error)

        if "message" in error_data:
            msg: str = error_data["message"]
            return msg

        return str(error_data)

    except Exception:
        return response.text or f"HTTP {response.status_code}"
