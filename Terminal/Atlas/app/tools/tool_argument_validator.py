from __future__ import annotations

from app.tools.tool_call import ToolCall
from app.tools.tool_schema import ToolSchema


class ToolArgumentValidator:
    """
    Validates tool arguments against a tool schema.

    Version 1:
    - Every argument defined in the schema is required.
    - Argument types are not validated.
    - Unknown arguments are allowed.
    """

    def validate(
            self,
            call: ToolCall,
            schema: ToolSchema,
    ) -> list[str]:
        errors: list[str] = []

        for argument in schema.arguments:
            if argument not in call.arguments:
                errors.append(
                    f"Missing required parameter: {argument}"
                )

        return errors