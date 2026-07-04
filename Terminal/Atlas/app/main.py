import asyncio
from typing import Any

from app.core.logger import logger
from app.kernel.bootstrap import Kernel
from app.memory.conversation import Conversation
from app.types.model_result import ModelResult


async def generate_from_conversation(
        kernel: Kernel,
        conversation: Conversation,
        **kwargs: Any,
) -> ModelResult:
    """Generate a response from conversation history."""

    context = kernel._context_builder.build(conversation)

    if not context:
        return ModelResult.fail(
            error="Conversation cannot be empty."
        )

    return await kernel.generate(
        context,
        **kwargs,
    )


async def main() -> None:
    kernel = Kernel()
    kernel.boot()

    tool_result = await kernel.execute_tool("system_info")

    if tool_result.success:
        logger.info(
            "System information: {}",
            tool_result.data,
        )
    else:
        logger.error(
            "System information tool failed: {}",
            tool_result.error,
        )

    model_result = await kernel.generate(
        "echo",
        "Hello Atlas",
    )

    if model_result.success:
        logger.info(
            "Model response: {}",
            model_result.content,
        )
    else:
        logger.error(
            "Model failed: {}",
            model_result.error,
        )


if __name__ == "__main__":
    asyncio.run(main())