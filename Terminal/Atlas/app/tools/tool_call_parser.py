from __future__ import annotations

from app.tools.tool_call import ToolCall


class ToolCallParser:

    PREFIX = "CALL_TOOL:"

    def parse(
            self,
            response: str,
    ) -> ToolCall | None:

        response = response.strip()

        if not response.startswith(self.PREFIX):
            return None

        lines = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return None

        tool_name = lines[1]

        kwargs: dict[str, str] = {}

        for line in lines[2:]:

            if "=" not in line:
                continue

            key, value = line.split(
                "=",
                1,
            )

            kwargs[key.strip()] = value.strip()

        return ToolCall(
            tool_name,
            kwargs,
        )