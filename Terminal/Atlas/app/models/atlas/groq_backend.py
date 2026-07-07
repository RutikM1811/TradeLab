"""
Atlas Groq Inference Backend.

Provides real AI inference through the Groq API.
"""

from typing import Any

from openai import AsyncOpenAI

from app.memory.conversation import Conversation
from app.models.atlas.backend import AbstractInferenceBackend
from app.types.model_result import ModelResult


class GroqBackend(AbstractInferenceBackend):
    """Inference backend powered by Groq."""

    def __init__(
            self,
            api_key: str,
            model: str,
            base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "Groq API key cannot be empty."
            )

        if not model.strip():
            raise ValueError(
                "Groq model cannot be empty."
            )

        self._model = model.strip()

        self._client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        """Return the backend name."""

        return "groq"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response using Groq."""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return await self._generate_messages(
            messages,
            **kwargs,
        )

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate using structured conversation history."""

        messages: list[dict[str, str]] = []

        if (
                system_prompt is not None
                and system_prompt.strip()
        ):
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.extend(
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in conversation.all()
        )

        return await self._generate_messages(
            messages,
            **kwargs,
        )

    async def _generate_messages(
            self,
            messages: list[dict[str, str]],
            **kwargs: Any,
    ) -> ModelResult:
        """Generate from model-ready messages."""

        try:
            response = await (
                self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    **kwargs,
                )
            )

            content = response.choices[0].message.content

            if not content:
                return ModelResult.fail(
                    error="Groq returned an empty response."
                )

            return ModelResult.ok(
                content=content
            )

        except Exception as error:
            return ModelResult.fail(
                error=self._format_error(error)
            )

    @staticmethod
    def _format_error(
            error: Exception,
    ) -> str:
        """Convert Groq errors into readable messages."""

        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if status_code == 401:
            return (
                "Groq authentication failed. "
                "Check your API key."
            )

        if status_code == 402:
            return (
                "Groq credits are required "
                "for the selected model."
            )

        if status_code == 429:
            return (
                "Groq is temporarily rate-limited. "
                "Please retry shortly."
            )

        return (
            "Groq generation failed: "
            f"{error}"
        )