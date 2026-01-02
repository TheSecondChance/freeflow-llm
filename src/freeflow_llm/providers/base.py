from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models import FreeFlowResponse


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider. If None, will try to load from environment.
        """
        self.api_key = api_key
        self.name = self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is available (has valid API key).

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
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
        Create a chat completion.

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
            RateLimitError: If rate limit is hit
            ProviderError: For other provider errors
        """
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
