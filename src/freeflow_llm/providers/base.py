import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional, Union

import httpx

from ..config import DEFAULT_MODELS
from ..exceptions import ProviderError, RateLimitError
from ..models import FreeFlowResponse
from ..utils import (
    extract_error_message,
    get_api_keys,
    is_rate_limit_error,
    parse_sse_line,
)

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Base provider with built-in httpx support for all LLM providers.

    Subclasses only need to implement:
    - get_api_base_url() - return the API base URL
    - build_request_headers() - return auth headers
    - build_request_payload() - transform messages to provider format
    - parse_response() - transform provider response to FreeFlowResponse
    - parse_stream_chunk() - transform streaming chunk to FreeFlowResponse
    """

    def __init__(self, api_key: Optional[Union[str, list[str]]] = None):
        """
        Initialize the provider.

        Args:
            api_key: API key(s) for the provider. Can be a single key or list of keys.
                    If None, will try to load from environment.
        """
        self.name = self.__class__.__name__.replace("Provider", "").lower()

        if api_key is None:
            self.api_keys = get_api_keys(self.name)
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            self.api_keys = [api_key]

        self.current_key_index = 0
        self.client = httpx.Client(timeout=30.0)
        self.stream_client = httpx.Client(timeout=60.0)

    @property
    def api_key(self) -> Optional[str]:
        """Get the current API key."""
        if self.api_keys and 0 <= self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return None

    def is_available(self) -> bool:
        """Check if this provider is available (has valid API key)."""
        return len(self.api_keys) > 0

    def rotate_key(self) -> bool:
        """
        Rotate to the next API key.

        Returns:
            True if rotated to a new key, False if no more keys available
        """
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            return True
        return False

    def reset_key_index(self) -> None:
        """Reset to the first API key."""
        self.current_key_index = 0

    def has_more_keys(self) -> bool:
        """Check if there are more API keys to try."""
        return self.current_key_index < len(self.api_keys) - 1

    @abstractmethod
    def get_api_base_url(self) -> str:
        """Return the API base URL for this provider."""
        pass

    @abstractmethod
    def build_request_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests."""
        pass

    @abstractmethod
    def build_request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """
        Build request payload for the provider.

        Returns:
            Tuple of (endpoint_path, json_payload)
        """
        pass

    @abstractmethod
    def parse_response(self, response_data: dict[str, Any], model: str) -> FreeFlowResponse:
        """Parse provider response to FreeFlowResponse."""
        pass

    @abstractmethod
    def parse_stream_chunk(
        self, chunk_data: dict[str, Any], model: str
    ) -> Optional[FreeFlowResponse]:
        """Parse streaming chunk to FreeFlowResponse. Return None to skip chunk."""
        pass

    def _make_request(
        self, endpoint: str, headers: dict[str, str], json_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Internal method to make HTTP requests with error handling."""
        try:
            response = self.client.post(
                endpoint,
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError(self.name, error_msg) from e
            raise ProviderError(self.name, error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or is_rate_limit_error(0, error_str):
                raise RateLimitError(self.name, error_str) from e
            raise ProviderError(self.name, error_str) from e

    def _stream_request(
        self, endpoint: str, headers: dict[str, str], json_data: dict[str, Any]
    ) -> Iterator[str]:
        """Internal method to make streaming HTTP requests with SSE support."""
        try:
            with self.stream_client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=json_data,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield data

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError(self.name, error_msg) from e
            raise ProviderError(self.name, error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or is_rate_limit_error(0, error_str):
                raise RateLimitError(self.name, error_str) from e
            raise ProviderError(self.name, error_str) from e

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> FreeFlowResponse:
        """
        Create a chat completion with automatic key rotation on rate limits.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model: Optional model name (provider-specific)
            **kwargs: Additional provider-specific parameters

        Returns:
            FreeFlowResponse object

        Raises:
            RateLimitError: If rate limit is hit on all API keys
            ProviderError: For other provider errors
        """
        if not self.is_available():
            raise ProviderError(self.name, f"{self.name.capitalize()} API key missing")

        if model is None:
            model = DEFAULT_MODELS.get(self.name, "default")

        # Try all available API keys
        last_error: Optional[Exception] = None
        keys_tried = 0

        while keys_tried < len(self.api_keys):
            try:
                logger.info(
                    f"{self.name}: Trying API key {self.current_key_index + 1}/{len(self.api_keys)}"
                )

                endpoint_path, json_data = self.build_request_payload(
                    messages, temperature, max_tokens, top_p, model, stream=False, **kwargs
                )
                headers = self.build_request_headers()
                url = f"{self.get_api_base_url()}{endpoint_path}"

                response_data = self._make_request(url, headers, json_data)
                self.reset_key_index()

                return self.parse_response(response_data, model)

            except RateLimitError as e:
                last_error = e
                keys_tried += 1

                logger.warning(
                    f"{self.name}: Rate limit hit on key {self.current_key_index + 1}/{len(self.api_keys)}"
                )

                if self.rotate_key():
                    continue
                else:
                    self.reset_key_index()
                    raise RateLimitError(
                        self.name, f"Rate limit hit on all {len(self.api_keys)} API key(s)"
                    ) from e

            except (ProviderError, Exception):
                self.reset_key_index()
                raise

        # Shouldn't reach here, but just in case
        self.reset_key_index()
        if last_error:
            raise last_error
        raise ProviderError(self.name, "Unknown error occurred")

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[FreeFlowResponse]:
        """
        Create a streaming chat completion with automatic key rotation on rate limits.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model: Optional model name (provider-specific)
            **kwargs: Additional provider-specific parameters

        Yields:
            FreeFlowResponse objects with partial content

        Raises:
            RateLimitError: If rate limit is hit on all API keys
            ProviderError: For other provider errors
        """
        if not self.is_available():
            raise ProviderError(self.name, f"{self.name.capitalize()} API key missing")

        if model is None:
            model = DEFAULT_MODELS.get(self.name, "default")

        # Try all available API keys
        last_error: Optional[Exception] = None
        keys_tried = 0

        while keys_tried < len(self.api_keys):
            try:
                logger.info(
                    f"{self.name}: Trying API key {self.current_key_index + 1}/{len(self.api_keys)} (streaming)"
                )

                endpoint_path, json_data = self.build_request_payload(
                    messages, temperature, max_tokens, top_p, model, stream=True, **kwargs
                )
                headers = self.build_request_headers()
                url = f"{self.get_api_base_url()}{endpoint_path}"

                for line in self._stream_request(url, headers, json_data):
                    chunk_data = parse_sse_line(line)
                    if chunk_data is None:
                        continue

                    chunk = self.parse_stream_chunk(chunk_data, model)
                    if chunk is not None:
                        yield chunk

                self.reset_key_index()
                return

            except RateLimitError as e:
                last_error = e
                keys_tried += 1

                logger.warning(
                    f"{self.name}: Rate limit hit on key {self.current_key_index + 1}/{len(self.api_keys)} (streaming)"
                )

                if self.rotate_key():
                    continue
                else:
                    self.reset_key_index()
                    raise RateLimitError(
                        self.name, f"Rate limit hit on all {len(self.api_keys)} API key(s)"
                    ) from e

            except (ProviderError, Exception):
                self.reset_key_index()
                raise

        # Shouldn't reach here, but just in case
        self.reset_key_index()
        if last_error:
            raise last_error
        raise ProviderError(self.name, "Unknown error occurred")

    def close(self) -> None:
        """
        Close HTTP clients and clean up resources.

        This method should be called when the provider is no longer needed
        to ensure proper cleanup of HTTP connections.
        """
        try:
            self.client.close()
            self.stream_client.close()
        except Exception as e:
            logger.warning(f"Error closing {self.name} provider clients: {e}")

    def __enter__(self) -> "BaseProvider":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and clean up resources."""
        self.close()

    def __del__(self):
        """Clean up HTTP clients."""
        try:
            self.client.close()
            self.stream_client.close()
        except Exception:
            pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
