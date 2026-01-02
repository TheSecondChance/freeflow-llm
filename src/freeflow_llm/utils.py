import os
from typing import Optional

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
        provider: Provider name (groq)

    Returns:
        API key if found, None otherwise
    """
    key_mapping = {
        "groq": "GROQ_API_KEY",
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
