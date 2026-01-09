import logging
from collections.abc import Iterator
from typing import Any, Optional

from .exceptions import NoProvidersAvailableError, ProviderError, RateLimitError
from .models import FreeFlowResponse
from .providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FreeFlowClient:
    """
    Main FreeFlow LLM client with automatic provider fallback.

    This client automatically tries multiple free-tier LLM providers in sequence,
    switching to the next provider when rate limits are hit.

    Example:
        ```python
        from freeflow_llm import FreeFlowClient
        # Use context manager for automatic cleanup
        with FreeFlowClient() as client:
            response = client.chat(
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            print(response.content)
        # Resources are automatically cleaned up when exiting the 'with' block

        # Or manually Basic usage
        client = FreeFlowClient()
        try:
            response = client.chat(
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            print(response.content)
        finally:
            client.close()  # Clean up resources
        ```
    """

    providers: list[BaseProvider]

    def __init__(
        self,
        providers: Optional[list[BaseProvider]] = None,
        verbose: bool = True,
    ):
        """
        Initialize FreeFlow client.

        Args:
            providers: Custom list of providers (defaults to all available providers)
            verbose: Whether to log provider switches and errors
        """
        self.verbose = verbose

        if providers is None:
            default_providers: list[BaseProvider] = [
                GroqProvider(),
                GeminiProvider(),
            ]
            self.providers = [p for p in default_providers if p.is_available()]
        else:
            self.providers = providers

        if not self.providers:
            logger.warning("No providers available. Please set API keys in environment variables.")

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
        Create a chat completion with automatic provider fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model: Optional model name (provider-specific)
            **kwargs: Additional parameters

        Returns:
            FreeFlowResponse with the completion result

        Raises:
            NoProvidersAvailableError: If all providers fail
        """
        if not self.providers:
            raise NoProvidersAvailableError(
                "No providers configured. Please set API keys in environment variables."
            )

        last_error: Optional[Exception] = None
        attempts: list[str] = []

        for provider in self.providers:
            try:
                num_keys = len(provider.api_keys) if hasattr(provider, "api_keys") else 1
                if self.verbose:
                    logger.info(
                        f"Attempting provider: {provider.name} (with {num_keys} API key(s))"
                    )

                completion = provider.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    model=model,
                    **kwargs,
                )

                if self.verbose:
                    logger.info(f"Success with provider: {provider.name}")

                return completion

            except RateLimitError as e:
                num_keys = len(provider.api_keys) if hasattr(provider, "api_keys") else 1
                attempts.append(f"{provider.name}: rate limited (tried {num_keys} key(s))")
                if self.verbose:
                    logger.warning(
                        f"Rate limit hit on all {num_keys} key(s) for {provider.name}, "
                        f"trying next provider..."
                    )
                last_error = e
                continue

            except ProviderError as e:
                attempts.append(f"{provider.name}: {str(e)}")
                if self.verbose:
                    logger.warning(f"Error with {provider.name}: {str(e)}, trying next provider...")
                last_error = e
                continue

            except Exception as e:
                attempts.append(f"{provider.name}: unexpected error")
                if self.verbose:
                    logger.warning(
                        f"Unexpected error with {provider.name}: {str(e)}, trying next provider..."
                    )
                last_error = e
                continue

        error_summary = "\n".join(f"  - {attempt}" for attempt in attempts)
        raise NoProvidersAvailableError(
            f"All providers exhausted. Attempts:\n{error_summary}\n\nLast error: {last_error}"
        )

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
        Create a streaming chat completion with automatic provider fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model: Optional model name (provider-specific)
            **kwargs: Additional parameters

        Yields:
            FreeFlowResponse objects with partial content

        Raises:
            NoProvidersAvailableError: If all providers fail
        """
        if not self.providers:
            raise NoProvidersAvailableError(
                "No providers configured. Please set API keys in environment variables."
            )

        last_error: Optional[Exception] = None
        attempts: list[str] = []

        for provider in self.providers:
            try:
                num_keys = len(provider.api_keys) if hasattr(provider, "api_keys") else 1
                if self.verbose:
                    logger.info(
                        f"Attempting provider: {provider.name} (with {num_keys} API key(s))"
                    )

                yield from provider.chat_stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    model=model,
                    **kwargs,
                )

                if self.verbose:
                    logger.info(f"Success with provider: {provider.name}")

                return

            except RateLimitError as e:
                num_keys = len(provider.api_keys) if hasattr(provider, "api_keys") else 1
                attempts.append(f"{provider.name}: rate limited (tried {num_keys} key(s))")
                if self.verbose:
                    logger.warning(
                        f"Rate limit hit on all {num_keys} key(s) for {provider.name}, "
                        f"trying next provider..."
                    )
                last_error = e
                continue

            except ProviderError as e:
                attempts.append(f"{provider.name}: {str(e)}")
                if self.verbose:
                    logger.warning(f"Error with {provider.name}: {str(e)}, trying next provider...")
                last_error = e
                continue

            except Exception as e:
                attempts.append(f"{provider.name}: unexpected error")
                if self.verbose:
                    logger.warning(
                        f"Unexpected error with {provider.name}: {str(e)}, trying next provider..."
                    )
                last_error = e
                continue

        error_summary = "\n".join(f"  - {attempt}" for attempt in attempts)
        raise NoProvidersAvailableError(
            f"All providers exhausted. Attempts:\n{error_summary}\n\nLast error: {last_error}"
        )

    def list_providers(self) -> list[str]:
        """
        List all available providers.

        Returns:
            list of provider names
        """
        return [p.name for p in self.providers]

    def close(self) -> None:
        """
        Close all providers and clean up resources.

        This method should be called when the client is no longer needed
        to ensure proper cleanup of HTTP connections and other resources.
        """
        for provider in self.providers:
            try:
                provider.close()
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error closing provider {provider.name}: {e}")

    def __enter__(self) -> "FreeFlowClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and clean up resources."""
        self.close()

    def __repr__(self) -> str:
        providers_str = ", ".join(self.list_providers())
        return f"FreeFlowClient(providers=[{providers_str}])"
