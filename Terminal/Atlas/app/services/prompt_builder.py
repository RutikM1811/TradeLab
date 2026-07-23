from __future__ import annotations

from app.tools.tool_schema import ToolSchema


class PromptBuilder:
    """Builds prompts for the language model."""

    def build(
            self,
            user_message: str,
            tools: tuple[ToolSchema, ...],
    ) -> str:
        lines: list[str] = []

        if tools:
            lines.append("Available tools:")
            lines.append("")

            for index, tool in enumerate(tools, start=1):
                lines.append(f"{index}.")
                lines.append(f"Name: {tool.name}")
                lines.append(f"Description: {tool.description}")

                if tool.arguments:
                    lines.append("Arguments:")

                    for argument, description in tool.arguments.items():
                        lines.append(
                            f"- {argument}: {description}"
                        )
                else:
                    lines.append("Arguments: None")

                lines.append("")

        lines.append("User message:")
        lines.append(user_message)

        return "\n".join(lines).strip()