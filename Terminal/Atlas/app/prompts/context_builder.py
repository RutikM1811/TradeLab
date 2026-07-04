"""
Atlas conversation context builder.

Converts structured conversation history into model-ready context.
"""

from app.memory.conversation import Conversation
from app.types.message import MessageRole


class ContextBuilder:
    """Build model-ready context from an Atlas conversation."""

    _ROLE_LABELS = {
        MessageRole.SYSTEM: "System",
        MessageRole.USER: "User",
        MessageRole.ASSISTANT: "Assistant",
        MessageRole.TOOL: "Tool",
    }

    def build(self, conversation: Conversation) -> str:
        """Build context from all conversation messages."""

        lines: list[str] = []

        for message in conversation.all():
            label = self._ROLE_LABELS[message.role]

            lines.append(
                f"{label}: {message.content}"
            )

        return "\n".join(lines)