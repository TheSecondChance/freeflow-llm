from typing import Any, Optional

from groq import Groq
from groq import RateLimitError as GroqRateLimitError

from ..config import DEFAULT_MODELS
from ..exceptions import ProviderError, RateLimitError
from ..models import FreeFlowResponse
from ..utils import get_api_key, is_rate_limit_error
from .base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq LLM provider (free tier: ~14,000 requests/day)."""

    client: Optional[Groq]

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or get_api_key("groq")
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        """Check if Groq is available."""
        return self.client is not None and self.api_key is not None

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
        Create a chat completion using Groq.

        Default model: llama-3.3-70b-versatile (fast and capable)
        """
        if not self.is_available():
            raise ProviderError("groq", "Groq client not initialized or API key missing")

        assert self.client is not None

        if model is None:
            model = DEFAULT_MODELS["groq"]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 1024,
                top_p=top_p,
                **kwargs,
            )

            # Convert to our standard format
            completion = FreeFlowResponse.from_dict(
                {
                    "id": response.id,
                    "object": "chat.completion",
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                    ],
                    "usage": (
                        {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None
                    ),
                },
                provider="groq",
            )
            return completion

        except GroqRateLimitError as e:
            raise RateLimitError("groq", str(e)) from e
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error by status code or message
            if "429" in error_str or is_rate_limit_error(0, error_str):
                raise RateLimitError("groq", error_str) from e
            raise ProviderError("groq", error_str) from e
