import time
from typing import Any, Optional

from ..models import FreeFlowResponse
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider.

    Default model: gemini-2.5-flash
    """

    def get_api_base_url(self) -> str:
        """Return Gemini API base URL."""
        return "https://generativelanguage.googleapis.com/v1beta"

    def build_request_headers(self) -> dict[str, str]:
        """Build HTTP headers with Gemini API key."""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": f"{self.api_key}",
        }

    def _convert_messages_to_gemini_format(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Convert standard messages to Gemini API format."""
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

    def build_request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        model: str,
        stream: bool = False,
        **_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Build Gemini-specific request payload."""
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

        # Determine endpoint based on streaming
        endpoint = (
            f"/models/{model}:streamGenerateContent"
            if stream
            else f"/models/{model}:generateContent"
        )

        return endpoint, json_data

    def parse_response(self, response_data: dict[str, Any], model: str) -> FreeFlowResponse:
        """Parse Gemini response to FreeFlowResponse."""
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

    def parse_stream_chunk(
        self, chunk_data: dict[str, Any], model: str
    ) -> Optional[FreeFlowResponse]:
        """Parse Gemini streaming chunk to FreeFlowResponse."""
        candidates = chunk_data.get("candidates", [])
        if not candidates:
            return None

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

        return FreeFlowResponse.from_dict(
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
