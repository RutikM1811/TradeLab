"""
Atlas interactive terminal application.
"""

import asyncio

from app.core.logger import logger
from app.kernel.bootstrap import Kernel
from app.terminal.command_runtime import TerminalCommandRuntime


async def main() -> None:
    """Run the Atlas interactive terminal."""

    kernel = Kernel()
    kernel.boot()

    commands = TerminalCommandRuntime(kernel)

    restored_count = kernel.restore_conversations()

    if restored_count > 0:
        conversation_id, _ = kernel.list_conversations()[-1]

        logger.info(
            "Restored {} conversation(s).",
            restored_count,
        )
        logger.info(
            "Continuing conversation: {}",
            conversation_id,
        )
    else:
        conversation_id, _ = commands.create_conversation()

        logger.info(
            "Created conversation: {}",
            conversation_id,
        )

    print()
    print("Atlas is ready.")
    print("Type /help to see commands.")
    print()

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue

        if message.lower() in {
            "/exit",
            "exit",
            "quit",
        }:
            break

        if message == "/help":
            print()

            for line in commands.help_text():
                print(line)

            print()
            continue

        if message == "/new":
            conversation_id, _ = (
                commands.create_conversation()
            )

            print(
                "Created new conversation: "
                f"{conversation_id}"
            )
            print()
            continue

        if message == "/list":
            conversations = (
                commands.list_conversations()
            )

            print()

            if not conversations:
                print("No conversations.")
            else:
                for index, (
                        listed_id,
                        conversation,
                ) in enumerate(
                    conversations,
                    start=1,
                ):
                    marker = (
                        "*"
                        if listed_id == conversation_id
                        else " "
                    )

                    print(
                        f"{marker} {index}. "
                        f"{conversation.metadata.title} "
                        f"({len(conversation)} messages)"
                    )

            print()
            continue

        if message.startswith("/rename"):
            parts = message.split(maxsplit=1)

            if len(parts) < 2:
                print(
                    "Command error: "
                    "Usage: /rename <title>"
                )
                print()
                continue

            try:
                commands.rename_conversation(
                    conversation_id,
                    parts[1],
                )

                print(
                    "Conversation renamed to: "
                    f"{parts[1].strip()}"
                )

            except ValueError as error:
                print(
                    f"Command error: {error}"
                )

            print()
            continue

        if message.startswith("/switch "):
            try:
                index = int(
                    message.split(maxsplit=1)[1]
                )

                conversation_id, _ = (
                    commands.switch_conversation(
                        index
                    )
                )

                print(
                    "Switched to conversation "
                    f"{index}: {conversation_id}"
                )

            except (ValueError, IndexError) as error:
                print(
                    f"Command error: {error}"
                )

            print()
            continue

        if message == "/history":
            history = commands.history(
                conversation_id
            )

            print()

            if not history:
                print("Conversation is empty.")
            else:
                for line in history:
                    print(line)

            print()
            continue

        if message == "/delete":
            commands.delete_conversation(
                conversation_id
            )

            print(
                "Current conversation deleted."
            )

            conversations = (
                commands.list_conversations()
            )

            if conversations:
                conversation_id, _ = (
                    conversations[-1]
                )

                print(
                    "Switched to conversation: "
                    f"{conversation_id}"
                )
            else:
                conversation_id, _ = (
                    commands.create_conversation()
                )

                print(
                    "Created new conversation: "
                    f"{conversation_id}"
                )

            print()
            continue
            if message == "/info":
                info = commands.conversation_info(
                    conversation_id
                )
                print()
                print(f"Title: {info['title']}")
                print(f"ID: {info['id']}")
                print(f"Created: {info['created_at']}")
                print(f"Updated: {info['updated_at']}")
                print(f"Messages: {info['message_count']}")
                print()
                continue
        if message.startswith("/"):
            print(
                "Unknown command. "
                "Type /help to see available commands."
            )
            print()
            continue

        result = await kernel.chat(
            conversation_id,
            message,
        )

        if result.success and result.content:
            print(
                f"Atlas: {result.content}"
            )
        else:
            print(
                "Atlas error: "
                f"{result.error or 'No response generated.'}"
            )

        kernel.save_conversation(
            conversation_id
        )

        print()

    kernel.save_conversation(
        conversation_id
    )

    print()
    print("Conversation saved.")
    print("Atlas stopped.")


if __name__ == "__main__":
    asyncio.run(main())