import json
import os
from collections.abc import Iterator
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


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


def create_http_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    """
    Create a configured httpx client for synchronous requests.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Configured httpx.Client instance
    """
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
    )


def make_api_request(
    url: str,
    headers: dict[str, str],
    json_data: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Make a synchronous POST request to an API endpoint.

    Args:
        url: API endpoint URL
        headers: HTTP headers
        json_data: JSON request body
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response

    Raises:
        httpx.HTTPStatusError: For HTTP error responses
        httpx.TimeoutException: For timeout errors
    """
    with create_http_client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def stream_api_request(
    url: str,
    headers: dict[str, str],
    json_data: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[str]:
    """
    Make a streaming POST request to an API endpoint using SSE.

    Args:
        url: API endpoint URL
        headers: HTTP headers
        json_data: JSON request body (should include stream=true)
        timeout: Request timeout in seconds

    Yields:
        SSE data lines (without 'data: ' prefix)

    Raises:
        httpx.HTTPStatusError: For HTTP error responses
        httpx.TimeoutException: For timeout errors
    """
    with (
        create_http_client(timeout=timeout) as client,
        client.stream("POST", url, headers=headers, json=json_data) as response,
    ):
        response.raise_for_status()

        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                yield data


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
