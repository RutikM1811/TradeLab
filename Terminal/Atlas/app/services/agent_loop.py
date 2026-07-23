from __future__ import annotations

from typing import Any

from app.memory.conversation import Conversation
from app.models.atlas.atlas_model import AtlasModel
from app.services.prompt_builder import PromptBuilder
from app.tools.tool_call_parser import ToolCallParser
from app.tools.tool_registry import ToolRegistry
from app.types.model_result import ModelResult


class AgentLoop:
    """
    Minimal agent loop.

    Phase 1:
    - Add the user message.
    - Build a prompt.
    - Call the model.
    - Parse the response.
    - Return the model result.
    """

    def __init__(
            self,
            model: AtlasModel,
            tool_registry: ToolRegistry,
            prompt_builder: PromptBuilder | None = None,
            tool_call_parser: ToolCallParser | None = None,
    ) -> None:
        self._model = model
        self._tool_registry = tool_registry
        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder()
        )
        self._tool_call_parser = (
            tool_call_parser
            if tool_call_parser is not None
            else ToolCallParser()
        )

    async def run(
            self,
            conversation: Conversation,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """
        Execute one iteration of the agent loop.
        """

        conversation.add_user(message)

        tools = tuple(
            tool.schema
            for tool in self._tool_registry.values()
        )

        prompt = self._prompt_builder.build(
            user_message=message,
            tools=tools,
        )

        result = await self._model.generate(
            prompt,
            **kwargs,
        )

        if not result.success:
            return result

        if result.content is None:
            return result

        self._tool_call_parser.parse(
            result.content,
        )

        return result