"""
Atlas conversation state.

Stores and manages messages for one conversation.
"""

from app.types.message import Message, MessageRole


class Conversation:
    """Represents one Atlas conversation."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Add a message to the conversation."""

        if not message.content.strip():
            raise ValueError("Message content cannot be empty.")

        self._messages.append(message)

    def add_system(self, content: str) -> Message:
        """Add a system message."""

        return self._add_role_message(
            MessageRole.SYSTEM,
            content,
        )

    def add_user(self, content: str) -> Message:
        """Add a user message."""

        return self._add_role_message(
            MessageRole.USER,
            content,
        )

    def add_assistant(self, content: str) -> Message:
        """Add an assistant message."""

        return self._add_role_message(
            MessageRole.ASSISTANT,
            content,
        )

    def add_tool(self, content: str) -> Message:
        """Add a tool message."""

        return self._add_role_message(
            MessageRole.TOOL,
            content,
        )

    def all(self) -> tuple[Message, ...]:
        """Return all messages in insertion order."""

        return tuple(self._messages)

    def last(self) -> Message | None:
        """Return the most recent message."""

        if not self._messages:
            return None

        return self._messages[-1]

    def __len__(self) -> int:
        return len(self._messages)

    def _add_role_message(
            self,
            role: MessageRole,
            content: str,
    ) -> Message:
        message = Message(
            role=role,
            content=content,
        )

        self.add(message)

        return message