from __future__ import annotations


class ToolRouter:
    """Routes user messages to Atlas tools."""

    _SYSTEM_INFO_KEYWORDS = (
        "cpu",
        "ram",
        "memory",
        "system",
        "hardware",
        "specs",
    )

    def route(
            self,
            message: str,
    ) -> str | None:
        message = message.strip().lower()

        if not message:
            return None

        if any(keyword in message for keyword in self._SYSTEM_INFO_KEYWORDS):
            return "system_info"

        return None