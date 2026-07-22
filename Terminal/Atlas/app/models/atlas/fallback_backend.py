from collections.abc import AsyncIterator
from typing import Any

from app.memory.conversation import Conversation
from app.models.atlas.backend import AbstractInferenceBackend


class FallbackBackend(AbstractInferenceBackend):

    @property
    def name(self) -> str:
        return "fallback"

    def __init__(self, backends):
        self.backends = backends

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ):
        last_exception = None

        for backend in self.backends:
            try:
                print(f"Trying {backend.name}")
                result = await backend.generate(
                    prompt,
                    **kwargs,
                )
                print(f"{backend.name} succeeded")
                return result

            except Exception as ex:
                print(
                    f"{backend.name} failed: "
                    f"{type(ex).__name__}: {ex}"
                )
                last_exception = ex

        raise last_exception

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ):
        last_exception = None

        for backend in self.backends:
            try:
                return await backend.generate_from_conversation(
                    conversation,
                    system_prompt=system_prompt,
                    **kwargs,
                )

            except Exception as ex:
                last_exception = ex

        raise last_exception

    async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response using the first backend that succeeds.
        """

        last_exception = None

        for backend in self.backends:
            try:
                async for chunk in backend.generate_stream(
                        prompt,
                        **kwargs,
                ):
                    yield chunk

                return

            except Exception as ex:
                last_exception = ex

        raise last_exception

    async def generate_stream_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response from a conversation using the first
        backend that succeeds.
        """

        last_exception = None

        for backend in self.backends:
            try:
                async for chunk in backend.generate_stream_from_conversation(
                        conversation,
                        system_prompt=system_prompt,
                        **kwargs,
                ):
                    yield chunk

                return

            except Exception as ex:
                last_exception = ex

        raise last_exception