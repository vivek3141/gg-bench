from typing import Literal, TypedDict


class Message(TypedDict):
    """Represents a chat message."""

    role: Literal["system", "user", "assistant"]
    content: str
