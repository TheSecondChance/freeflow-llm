import time
from collections.abc import Iterator
from typing import Any, Optional

import httpx

from ..config import DEFAULT_MODELS
from ..exceptions import ProviderError, RateLimitError
from ..models import FreeFlowResponse
from ..utils import (
    extract_error_message,
    get_api_key,
    is_rate_limit_error,
    make_api_request,
    parse_sse_line,
    stream_api_request,
)
from .base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq LLM provider (free tier: ~14,000 requests/day)."""

    GROQ_API_BASE = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or get_api_key("groq")

    def is_available(self) -> bool:
        """Check if Groq is available."""
        return self.api_key is not None

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Groq API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
            raise ProviderError("groq", "Groq API key missing")

        if model is None:
            model = DEFAULT_MODELS["groq"]

        url = f"{self.GROQ_API_BASE}/chat/completions"
        headers = self._build_headers()

        json_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_tokens is not None:
            json_data["max_tokens"] = max_tokens
        else:
            json_data["max_tokens"] = 1024

        # Add any additional kwargs
        json_data.update(kwargs)

        try:
            response_data = make_api_request(url, headers, json_data)

            # Convert to our standard format
            completion = FreeFlowResponse.from_dict(
                {
                    "id": response_data.get("id", f"chatcmpl-{int(time.time())}"),
                    "object": "chat.completion",
                    "created": response_data.get("created", int(time.time())),
                    "model": response_data.get("model", model),
                    "choices": [
                        {
                            "index": choice.get("index", 0),
                            "message": {
                                "role": choice.get("message", {}).get("role", "assistant"),
                                "content": choice.get("message", {}).get("content", ""),
                            },
                            "finish_reason": choice.get("finish_reason", "stop"),
                        }
                        for choice in response_data.get("choices", [])
                    ],
                    "usage": (
                        {
                            "prompt_tokens": response_data["usage"]["prompt_tokens"],
                            "completion_tokens": response_data["usage"]["completion_tokens"],
                            "total_tokens": response_data["usage"]["total_tokens"],
                        }
                        if response_data.get("usage")
                        else None
                    ),
                },
                provider="groq",
            )
            return completion

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError("groq", error_msg) from e
            raise ProviderError("groq", error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError("groq", f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or is_rate_limit_error(0, error_str):
                raise RateLimitError("groq", error_str) from e
            raise ProviderError("groq", error_str) from e

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
        Create a streaming chat completion using Groq.

        Default model: llama-3.3-70b-versatile (fast and capable)
        """
        if not self.is_available():
            raise ProviderError("groq", "Groq API key missing")

        if model is None:
            model = DEFAULT_MODELS["groq"]

        url = f"{self.GROQ_API_BASE}/chat/completions"
        headers = self._build_headers()

        json_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        if max_tokens is not None:
            json_data["max_tokens"] = max_tokens
        else:
            json_data["max_tokens"] = 1024

        # Add any additional kwargs
        json_data.update(kwargs)

        try:
            for line in stream_api_request(url, headers, json_data):
                chunk_data = parse_sse_line(line)
                if chunk_data is None:
                    continue

                # Convert to our standard format
                chunk = FreeFlowResponse.from_dict(
                    {
                        "id": chunk_data.get("id", f"chatcmpl-{int(time.time())}"),
                        "object": "chat.completion.chunk",
                        "created": chunk_data.get("created", int(time.time())),
                        "model": chunk_data.get("model", model),
                        "choices": [
                            {
                                "index": choice.get("index", 0),
                                "delta": choice.get("delta", {}),
                                "finish_reason": choice.get("finish_reason"),
                            }
                            for choice in chunk_data.get("choices", [])
                        ],
                        "usage": None,
                    },
                    provider="groq",
                )
                yield chunk

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError("groq", error_msg) from e
            raise ProviderError("groq", error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError("groq", f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or is_rate_limit_error(0, error_str):
                raise RateLimitError("groq", error_str) from e
            raise ProviderError("groq", error_str) from e
