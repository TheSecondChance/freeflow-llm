from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """Represents a chat message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Choice:
    """Represents a completion choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Choice":
        return cls(
            index=data.get("index", 0),
            message=Message(
                role=data.get("message", {}).get("role", "assistant"),
                content=data.get("message", {}).get("content", ""),
            ),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class Usage:
    """Represents token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class FreeFlowResponse:
    """Represents a chat completion response from any provider."""

    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    provider: Optional[str] = None

    @property
    def content(self) -> str:
        """Convenience property to get the response content directly."""
        if self.choices and len(self.choices) > 0:
            return self.choices[0].message.content
        return ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], provider: Optional[str] = None) -> "FreeFlowResponse":
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=[Choice.from_dict(choice) for choice in data.get("choices", [])],
            usage=Usage.from_dict(data.get("usage", {})) if data.get("usage") else None,
            provider=provider,
        )
