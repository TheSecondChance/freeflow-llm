"""
FreeFlow LLM - Chain multiple free-tier LLM APIs with automatic rate limit fallback.

This package provides a simple, unified interface to use multiple free LLM providers
(Groq, Gemini, Mistral, GitHub Models, OpenRouter) with automatic fallback when rate limits are hit.

Example:
    ```python
    from freeflow_llm import FreeFlowClient

    # Recommended: Use context manager for automatic resource cleanup
    with FreeFlowClient() as client:
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        )
        print(response.content)
        print(f"Response from: {response.provider}")
    # Resources are automatically cleaned up here

    # Or without context manager (remember to call close())
    client = FreeFlowClient()
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content)
    client.close()  # Clean up resources
    ```
"""

from . import config
from .client import FreeFlowClient
from .exceptions import (
    FreeFlowError,
    InvalidAPIKeyError,
    NoProvidersAvailableError,
    ProviderError,
    RateLimitError,
)
from .models import Choice, FreeFlowResponse, Message, Usage
from .providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
)

__version__ = "0.1.0"
__author__ = "FreeFlow Contributors"
__all__ = [
    "FreeFlowClient",
    "FreeFlowResponse",
    "Choice",
    "Message",
    "Usage",
    "FreeFlowError",
    "RateLimitError",
    "ProviderError",
    "NoProvidersAvailableError",
    "InvalidAPIKeyError",
    "BaseProvider",
    "GroqProvider",
    "GeminiProvider",
    "config",
]
