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


class GeminiProvider(BaseProvider):
    """Google Gemini provider"""

    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or get_api_key("google")

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.api_key is not None

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Gemini API requests."""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }

    def _convert_messages_to_gemini_format(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """
        Convert standard messages to Gemini API format.

        Returns:
            Tuple of (contents, system_instruction)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": content}]})

        return contents, system_instruction

    def _convert_gemini_response_to_standard(
        self, response_data: dict[str, Any], model: str
    ) -> FreeFlowResponse:
        """Convert Gemini API response to FreeFlowResponse."""
        candidates = response_data.get("candidates", [])

        if not candidates:
            response_text = ""
            finish_reason = "stop"
        else:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            response_text = parts[0].get("text", "") if parts else ""

            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop",
            }
            gemini_finish = candidate.get("finishReason", "STOP")
            finish_reason = finish_reason_map.get(gemini_finish, "stop")

        created_timestamp = int(time.time())

        return FreeFlowResponse.from_dict(
            {
                "id": f"chatcmpl-{created_timestamp}",
                "object": "chat.completion",
                "created": created_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "model",
                            "content": response_text,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": None,  # Gemini doesn't provide usage in basic response
            },
            provider="gemini",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        model: Optional[str] = None,
        **_kwargs: Any,
    ) -> FreeFlowResponse:
        """
        Create a chat completion using Gemini.

        Default model: gemini-2.5-flash
        """
        if not self.is_available():
            raise ProviderError("gemini", "Gemini API key missing")

        if model is None:
            model = DEFAULT_MODELS["gemini"]

        url = f"{self.GEMINI_API_BASE}/models/{model}:generateContent"
        headers = self._build_headers()

        contents, system_instruction = self._convert_messages_to_gemini_format(messages)

        generation_config = {
            "temperature": temperature,
            "topP": top_p,
        }

        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        json_data: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            json_data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            response_data = make_api_request(url, headers, json_data)
            return self._convert_gemini_response_to_standard(response_data, model)

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError("gemini", error_msg) from e
            raise ProviderError("gemini", error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError("gemini", f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if (
                "429" in error_str
                or "quota" in error_str.lower()
                or is_rate_limit_error(0, error_str)
            ):
                raise RateLimitError("gemini", error_str) from e
            raise ProviderError("gemini", error_str) from e

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        model: Optional[str] = None,
        **_kwargs: Any,
    ) -> Iterator[FreeFlowResponse]:
        """
        Create a streaming chat completion using Gemini.

        Default model: gemini-2.5-flash
        """
        if not self.is_available():
            raise ProviderError("gemini", "Gemini API key missing")

        if model is None:
            model = DEFAULT_MODELS["gemini"]

        url = f"{self.GEMINI_API_BASE}/models/{model}:streamGenerateContent"
        headers = self._build_headers()

        contents, system_instruction = self._convert_messages_to_gemini_format(messages)

        generation_config = {
            "temperature": temperature,
            "topP": top_p,
        }

        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        json_data: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            json_data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            for line in stream_api_request(url, headers, json_data):
                chunk_data = parse_sse_line(line)
                if chunk_data is None:
                    continue

                candidates = chunk_data.get("candidates", [])
                if not candidates:
                    continue

                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                delta_text = parts[0].get("text", "") if parts else ""

                finish_reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                    "RECITATION": "content_filter",
                    "OTHER": "stop",
                }
                gemini_finish = candidate.get("finishReason")
                finish_reason = finish_reason_map.get(gemini_finish) if gemini_finish else None

                created_timestamp = int(time.time())

                chunk = FreeFlowResponse.from_dict(
                    {
                        "id": f"chatcmpl-{created_timestamp}",
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text} if delta_text else {},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": None,
                    },
                    provider="gemini",
                )
                yield chunk

        except httpx.HTTPStatusError as e:
            error_msg = extract_error_message(e.response)
            if is_rate_limit_error(e.response.status_code, error_msg):
                raise RateLimitError("gemini", error_msg) from e
            raise ProviderError("gemini", error_msg) from e
        except httpx.TimeoutException as e:
            raise ProviderError("gemini", f"Request timeout: {str(e)}") from e
        except Exception as e:
            error_str = str(e)
            if (
                "429" in error_str
                or "quota" in error_str.lower()
                or is_rate_limit_error(0, error_str)
            ):
                raise RateLimitError("gemini", error_str) from e
            raise ProviderError("gemini", error_str) from e
